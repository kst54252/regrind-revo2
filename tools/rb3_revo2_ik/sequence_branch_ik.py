"""Opt-in offline branch selection around the existing strict pose IK.

No new robot model, realtime controller, angle wrapping, or pose relaxation.
Uses future reference poses: NOT a drop-in causal policy IK replacement.
"""
from __future__ import annotations

import copy
import numpy as np

from tools.rb3_revo2_ik.warm_start_ik import PoseJacobian


def solve_local(kin, p, quat, seed, max_nfev=300):
    """The existing bounded solver, with one explicit seed and verified Jacobian."""
    local = copy.copy(kin)
    seed = np.clip(seed, kin.joint_lower + 1e-10, kin.joint_upper - 1e-10)
    local._candidate_seeds = lambda warm, neutral: [seed.copy()]
    differential = PoseJacobian(kin)
    local._residual = differential.residual
    return local.inverse(p, quat, initial_q=seed, neutral_q=seed,
                         position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3,
                         jacobian=differential.jacobian, max_nfev=max_nfev)


def continue_path(kin, positions, quaternions, seed, reverse=False, max_nfev=300):
    q = np.full((len(positions), 6), np.nan)
    warm = np.asarray(seed).copy()
    order = range(len(q)-1, -1, -1) if reverse else range(len(q))
    for i in order:
        result = solve_local(kin, positions[i], quaternions[i], warm, max_nfev)
        if not result.success:
            break
        q[i] = result.q
        warm = result.q
    return q


def seed_bank(kin, reference, count=24):
    """Deterministic alternatives, including elbow/base/wrist branches; no telemetry."""
    if count < 4:
        raise ValueError('At least four seeds required')
    seeds = [np.asarray(reference).copy(), np.zeros(6)]
    for base_shift in (0., np.pi, -np.pi):
        for elbow_sign in (1., -1.):
            q = np.asarray(reference).copy()
            q[0] += base_shift
            q[1:3] *= elbow_sign
            seeds.append(q)
    rng = np.random.default_rng(730)
    while len(seeds) < count:
        seeds.append(rng.uniform(np.maximum(kin.joint_lower, -np.pi),
                                 np.minimum(kin.joint_upper, np.pi)))
    return [np.clip(q, kin.joint_lower+1e-8, kin.joint_upper-1e-8) for q in seeds[:count]]


def distinct_solutions(kin, p, quat, seeds, max_nfev=300):
    result = []
    for seed in seeds:
        solution = solve_local(kin, p, quat, seed, max_nfev)
        if solution.success and not any(np.max(np.abs(solution.q-q)) < 1e-4 for q in result):
            result.append(solution.q)
    return result


def path_score(q, dt):
    """Raw bounded-coordinate speed then acceleration; no wrap or time shift."""
    if not np.isfinite(q).all():
        return (float('inf'), float('inf'), float('inf'))
    velocity = np.diff(q, axis=0) / dt
    return (float(np.max(np.abs(velocity), initial=0)),
            float(np.max(np.abs(np.diff(velocity, axis=0)/dt), initial=0)),
            float(np.sum(velocity**2)))


def minimax_path(layers, dt):
    """Small finite candidate graph: minimize largest actual coordinate step.

    Every vertex is already strict-FK validated. Edges are not collision checked.
    This does not claim a globally optimal path over all IK solutions.
    """
    if dt <= 0 or not layers or any(len(layer) == 0 for layer in layers):
        raise ValueError('Positive dt and nonempty layers required')
    cost = np.zeros(len(layers[0]))
    parents = []
    for previous, current in zip(layers, layers[1:]):
        steps = np.max(np.abs(current[:, None, :] - previous[None, :, :]), axis=2) / dt
        candidates = np.maximum(steps, cost[None, :])
        parent = np.argmin(candidates, axis=1)
        cost = candidates[np.arange(len(current)), parent]
        parents.append(parent)
    bottleneck = float(np.min(cost))
    # Among paths with that bottleneck, minimize squared increments. A pure
    # minimax tie must not arbitrarily alternate between otherwise valid nodes.
    cost = np.zeros(len(layers[0]))
    parents = []
    for previous, current in zip(layers, layers[1:]):
        velocity = (current[:, None, :] - previous[None, :, :]) / dt
        admissible = np.max(np.abs(velocity), axis=2) <= bottleneck+1e-9
        candidates = np.where(admissible, np.sum(velocity**2, axis=2)+cost[None, :], np.inf)
        parent = np.argmin(candidates, axis=1)
        cost = candidates[np.arange(len(current)), parent]
        parents.append(parent)
    index = int(np.argmin(cost))
    path = [layers[-1][index]]
    for t in range(len(layers)-2, -1, -1):
        index = int(parents[t][index])
        path.append(layers[t][index])
    return np.array(path[::-1])


def segment_intersects_box(start, end, lower, upper):
    """Closed line segment / solid AABB intersection, without coarse sampling."""
    first, last = 0., 1.
    for a, b, lo, hi in zip(start, end, lower, upper):
        delta = b-a
        if abs(delta) < 1e-12:
            if a < lo or a > hi:
                return False
            continue
        t0, t1 = sorted(((lo-a)/delta, (hi-a)/delta))
        first, last = max(first, t0), min(last, t1)
        if first > last:
            return False
    return True


def workcell_boxes(layout):
    """Actual configured solid table/pedestal/leg boxes; not robot mesh collision."""
    pedestal = layout['robot_base']
    center, size = np.array(pedestal['center']), np.array(pedestal['size'])
    boxes = [(center-size/2, center+size/2)]
    table = layout['table']
    center = np.r_[table['center_xy'], table['top_z']-table['top_thickness']/2]
    size = np.r_[table['size_xy'], table['top_thickness']]
    boxes.append((center-size/2, center+size/2))
    for xy in table['leg_centers_xy']:
        boxes.append((np.r_[np.array(xy)-np.array(table['leg_size_xy'])/2, layout['floor_z']],
                      np.r_[np.array(xy)+np.array(table['leg_size_xy'])/2,
                            table['top_z']-table['top_thickness']]))
    return boxes


def centerline_clear(kin, q, boxes, floor_z):
    """Reject obvious arm-axis passage through the workcell; no radius/self test."""
    chain = kin.forward_chain_points(q)[1:]  # Exclude intended fixed base mounting.
    return bool(np.min(chain[:, 2]) > floor_z and not any(
        segment_intersects_box(a, b, lower, upper)
        for a, b in zip(chain, chain[1:]) for lower, upper in boxes))


def improve_paths(kin, positions, quaternions, baseline, dt, seeds=24, max_nfev=300,
                  valid_configuration=None):
    local = continue_path(kin, positions, quaternions, baseline[0], max_nfev=max_nfev)
    bank = seed_bank(kin, baseline[0], seeds)
    starts = distinct_solutions(kin, positions[0], quaternions[0], bank, max_nfev)
    forward = [continue_path(kin, positions, quaternions, q, max_nfev=max_nfev) for q in starts]
    complete_forward = [q for q in forward if np.isfinite(q).all()]
    best_forward = min([baseline] + complete_forward, key=lambda q: path_score(q, dt))
    ends = distinct_solutions(kin, positions[-1], quaternions[-1], bank, max_nfev)
    backward = [continue_path(kin, positions, quaternions, q, reverse=True,
                             max_nfev=max_nfev) for q in ends]
    # Baseline always remains an option in the unconstrained graph. Partial solved prefixes/suffixes may
    # contribute vertices, but never count as successful complete trajectories.
    pool = [baseline, local] + forward + backward
    layers = []
    for t in range(len(positions)):
        nodes = []
        for q in pool:
            if np.isfinite(q[t]).all() and not any(np.max(np.abs(q[t]-old)) < 1e-5 for old in nodes):
                nodes.append(q[t])
        layers.append(np.asarray(nodes))
    graph = minimax_path(layers, dt)
    best = min([baseline, best_forward, graph] + [q for q in backward if np.isfinite(q).all()],
               key=lambda q: path_score(q, dt))
    variants = dict(local=local, multistart=best_forward, graph=graph, best=best)
    if valid_configuration is not None:
        screened = [np.array([q for q in layer if valid_configuration(q)]) for layer in layers]
        variants['screened_graph'] = (minimax_path(screened, dt) if all(len(a) for a in screened)
                                      else np.full_like(baseline, np.nan))
        candidates = [q for q in pool if np.isfinite(q).all() and all(valid_configuration(a) for a in q)]
        candidates.append(variants['screened_graph'])
        variants['screened_best'] = min(candidates, key=lambda q: path_score(q, dt))
    return variants, dict(
        first_pose_solutions=len(starts), last_pose_solutions=len(ends),
        complete_forward_paths=len(complete_forward),
        complete_backward_paths=int(sum(np.isfinite(q).all() for q in backward)),
        nodes_per_frame=[len(layer) for layer in layers])
