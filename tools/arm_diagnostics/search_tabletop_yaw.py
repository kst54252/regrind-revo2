"""Offline approach-yaw selection for a fixed can position, using existing IK.

Rotate a whole task reference about the initial can, NOT only wrist orientation.
No runtime policy adapter, object symmetry reward, or physics setting is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools.arm_diagnostics.compare_tabletop_branches import metrics
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.reference_trajectory import load_reference_trajectory
from tools.rb3_revo2_ik.sequence_branch_ik import (
    centerline_clear, continue_path, distinct_solutions, path_score, seed_bank, workcell_boxes,
)

ROOT = Path(__file__).resolve().parents[2]


def rotate_task(positions, quaternions_xyzw, original_center, desired_center, yaw_deg):
    """World vertical yaw, XYZW; same fixed transform for every trajectory frame."""
    positions = np.asarray(positions, dtype=float)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=float)
    original_center, desired_center = np.asarray(original_center), np.asarray(desired_center)
    if (positions.ndim != 2 or positions.shape[1] != 3 or
            quaternions_xyzw.shape != (len(positions), 4) or
            original_center.shape != (3,) or desired_center.shape != (3,) or
            not all(np.isfinite(x).all() for x in (positions, quaternions_xyzw,
                                                   original_center, desired_center, yaw_deg))):
        raise ValueError('Expected finite poses (T,3)/(T,4), centers (3,), and yaw')
    yaw = Rotation.from_euler('z', yaw_deg, degrees=True)
    return (desired_center + yaw.apply(positions-original_center),
            (yaw*Rotation.from_quat(quaternions_xyzw)).as_quat())


def dense_reference(reference):
    source_t = np.arange(reference.frames)*reference.dt
    t = np.arange(4*(reference.frames-1)+1)*reference.dt/4
    positions = np.column_stack([np.interp(t, source_t, reference.wrist_pos[:, j]) for j in range(3)])
    quaternions = Slerp(source_t, Rotation.from_quat(reference.wrist_quat_xyzw))(t).as_quat()
    return t, positions, quaternions


def angle_key(degrees):
    # Only canonicalize the SEARCH ANGLE label, never wrap robot joint paths.
    return float((degrees+180.) % 360. - 180.)


def search_one(job):
    index, xy, zero_q, reference_data, layout, vertices, seed_count, budget, coarse_step, refine_step, dense_angles = job
    source_pos, source_quat, dense_pos, dense_quat, center, dt = reference_data
    target_center = np.r_[xy, center[2]]
    kin = RB3730Kinematics(base_position=layout['robot_mount']['position'],
                          base_quaternion_xyzw=layout['robot_mount']['quaternion_xyzw'])
    boxes = workcell_boxes(layout)
    clear = lambda q: centerline_clear(kin, q, boxes, layout['floor_z'])
    table = layout['table']
    lower = np.array(table['center_xy'])-np.array(table['size_xy'])/2
    upper = np.array(table['center_xy'])+np.array(table['size_xy'])/2
    records, candidates = [], {}
    started = time.monotonic()

    def evaluate_angle(theta):
        theta = angle_key(theta)
        if theta in candidates:
            return
        candidates[theta] = []
        footprint = Rotation.from_euler('z', theta, degrees=True).apply(vertices)[:, :2]+xy
        if np.any(footprint.min(0) < lower) or np.any(footprint.max(0) > upper):
            records.append(dict(yaw_deg=theta, reason='initial_can_footprint_outside_table'))
            return
        pos, quat = rotate_task(source_pos, source_quat, center, target_center, theta)
        neutral = zero_q[0].copy()
        neutral[0] += np.deg2rad(theta)
        starts = distinct_solutions(kin, pos[0], quat[0], seed_bank(kin, neutral, seed_count), budget)
        valid_starts = [q for q in starts if clear(q)]
        for q0 in valid_starts:
            q = continue_path(kin, pos, quat, q0, max_nfev=budget)
            if np.isfinite(q).all() and all(clear(qi) for qi in q):
                candidates[theta].append(q)
        candidates[theta].sort(key=lambda q: path_score(q, dt))
        q = candidates[theta][0] if candidates[theta] else None
        records.append(dict(yaw_deg=theta, reason='complete' if q is not None else 'no_complete_clear_path',
                            first_pose_solutions=len(starts), clear_first_solutions=len(valid_starts),
                            clear_complete_paths=len(candidates[theta]),
                            max_speed=path_score(q, dt)[0] if q is not None else None))

    for theta in range(-180, 180, coarse_step):
        evaluate_angle(theta)
    feasible = [(path_score(paths[0], dt), angle) for angle, paths in candidates.items() if paths]
    if feasible:
        best_coarse = min(feasible)[1]
        for offset in np.array([-3, -2, -1, 1, 2, 3])*refine_step:
            evaluate_angle(best_coarse+offset)
    # Validate two branches from each of the three best angles, at 120Hz with
    # no mid-motion resets. A coarse result alone never becomes a success.
    ranked = sorted((path_score(paths[0], dt), angle) for angle, paths in candidates.items() if paths)[:dense_angles]
    p0, r0 = rotate_task(dense_pos, dense_quat, center, target_center, 0.)
    baseline_stats = metrics(kin, zero_q, p0, r0, dt/4, boxes, layout['floor_z'], 10.)
    dense_candidates = []
    if baseline_stats['strict_pose'] and baseline_stats['coarse_clear']:
        dense_candidates.append((zero_q, 0., baseline_stats, 'prior_zero_yaw_selected_path'))
    for _, theta in ranked:
        pos, quat = rotate_task(dense_pos, dense_quat, center, target_center, theta)
        for coarse_q in candidates[theta][:2]:
            q = continue_path(kin, pos, quat, coarse_q[0], max_nfev=budget)
            result = metrics(kin, q, pos, quat, dt/4, boxes, layout['floor_z'], 10.)
            if result['strict_pose'] and result['coarse_clear']:
                dense_candidates.append((q, theta, result, 'dense_local_continuation'))
    if not dense_candidates:
        raise RuntimeError(f'No valid candidate including previous zero yaw at {xy}')
    best_q, best_yaw, best_stats, method = min(dense_candidates, key=lambda item: path_score(item[0], dt/4))
    pos, quat = rotate_task(dense_pos, dense_quat, center, target_center, best_yaw)
    return dict(index=index, xy=xy, q=best_q, yaw=best_yaw, pos=pos, quat=quat,
                before=baseline_stats, after=best_stats, method=method, angles=records,
                dense_candidates=len(dense_candidates), elapsed_s=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', type=Path, required=True)
    p.add_argument('--zero-yaw', type=Path, required=True, help='Previous optimized120 result directory')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seeds', type=int, default=12)
    p.add_argument('--max-nfev', type=int, default=300)
    p.add_argument('--coarse-step', type=int, default=30)
    p.add_argument('--refine-step', type=float, default=5.)
    p.add_argument('--dense-angles', type=int, default=3)
    p.add_argument('--only-indices', type=int, nargs='+', help='Original scan indices for a bounded targeted retry')
    p.add_argument('--merge-previous', type=Path, help='Preserve other placements from a prior yaw search in this new output')
    args = p.parse_args()
    if args.workers < 1 or args.seeds < 4 or args.max_nfev < 1:
        p.error('Positive workers/budget and at least four seeds required')
    if args.coarse_step <= 0 or 360 % args.coarse_step or args.refine_step <= 0 or args.dense_angles < 1:
        p.error('coarse-step must divide 360; refine-step/dense-angles must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.scan/'summary.json').read_text())
    zero_meta = json.loads((args.zero_yaw/'summary.json').read_text())
    if zero_meta['reference_sha256'] != meta['reference_sha256'] or zero_meta['model_sha256'] != meta['model_sha256']:
        raise ValueError('Previous result input mismatch')
    ref_path = Path(meta['reference'])
    if hashlib.sha256(ref_path.read_bytes()).hexdigest() != meta['reference_sha256']:
        raise ValueError('Reference changed since scan')
    if hashlib.sha256((ROOT/'tools/rb3_revo2_ik/rb3_model.json').read_bytes()).hexdigest() != meta['model_sha256']:
        raise ValueError('Robot model changed since scan')
    layout = json.loads((ROOT/'config/workcell/rb3_revo2_table.json').read_text())
    if layout != meta['workcell']:
        raise ValueError('Workcell changed since scan')
    reference = load_reference_trajectory(ref_path)
    t, dense_pos, dense_quat = dense_reference(reference)
    zero = np.load(args.zero_yaw/'paths.npz')
    scan = np.load(args.scan/'solutions.npz')
    indices = np.flatnonzero(scan['full_sequence_ik'] & ~scan['speed_screen_pass'])
    np.testing.assert_array_equal(indices, zero['indices'])
    np.testing.assert_allclose(zero['xy'], scan['xy'][indices], atol=0, rtol=0)
    if zero['selected'].shape != (len(indices), len(t), 6) or not np.isclose(zero['dt'], reference.dt/4):
        raise ValueError('Expected 120Hz zero-yaw baseline')
    import trimesh
    mesh = trimesh.load(str(ROOT/'007_tuna_fish_can/textured_simple.obj'), force='mesh', process=False)
    vertices = Rotation.from_quat(reference.object_quat_xyzw[0]).apply(mesh.vertices)
    data = (reference.wrist_pos, reference.wrist_quat_xyzw, dense_pos, dense_quat,
            reference.object_pos[0], reference.dt)
    if args.only_indices and not set(args.only_indices).issubset(set(indices.tolist())):
        raise ValueError('Retry indices must belong to the frozen orange set')
    jobs = [(int(i), scan['xy'][i], zero['selected'][j], data, layout, vertices, args.seeds, args.max_nfev,
             args.coarse_step, args.refine_step, args.dense_angles)
            for j, i in enumerate(indices) if not args.only_indices or int(i) in args.only_indices]
    results = []
    started = time.monotonic()
    print(f'Fixed placements={len(jobs)}, {360//args.coarse_step} coarse + up to 6 refined yaws each, {args.seeds} seeds', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(search_one, jobs, chunksize=1):
            results.append(result)
            if len(results)%10 == 0 or len(results)==len(jobs):
                print(f'{len(results)}/{len(jobs)}, 120Hz screened passes='
                      f'{sum(r["after"]["screened_pass"] for r in results)}, '
                      f'elapsed={time.monotonic()-started:.1f}s', flush=True)
    if args.merge_previous:
        prior_meta = json.loads((args.merge_previous/'summary.json').read_text())
        if prior_meta['reference_sha256'] != meta['reference_sha256'] or prior_meta['model_sha256'] != meta['model_sha256']:
            raise ValueError('Prior yaw result inputs differ')
        prior = np.load(args.merge_previous/'candidates.npz')
        np.testing.assert_array_equal(prior['indices'], indices)
        np.testing.assert_allclose(prior['time'], t, atol=1e-12, rtol=0)
        np.testing.assert_allclose(prior['xy'], zero['xy'], atol=0, rtol=0)
        prior_search = {r['index']: r for r in json.loads((args.merge_previous/'search_records.json').read_text())}
        replacements = {r['index']: r for r in results}
        combined = []
        kin = RB3730Kinematics(base_position=layout['robot_mount']['position'],
                              base_quaternion_xyzw=layout['robot_mount']['quaternion_xyzw'])
        boxes = workcell_boxes(layout)
        for j, index in enumerate(indices):
            index = int(index)
            xy = zero['xy'][j]; destination = np.r_[xy, reference.object_pos[0, 2]]
            p0, r0 = rotate_task(dense_pos, dense_quat, reference.object_pos[0], destination, 0.)
            item = dict(index=index, xy=xy, q=prior['q'][j], yaw=float(prior['yaw_deg'][j]),
                        pos=prior['target_wrist_pos'][j], quat=prior['target_wrist_quat_xyzw'][j],
                        before=metrics(kin, zero['selected'][j], p0, r0, reference.dt/4, boxes, layout['floor_z'], 10.),
                        after=metrics(kin, prior['q'][j], prior['target_wrist_pos'][j], prior['target_wrist_quat_xyzw'][j],
                                      reference.dt/4, boxes, layout['floor_z'], 10.),
                        **{k: v for k, v in prior_search[index].items() if k in ('method', 'angles', 'dense_candidates', 'elapsed_s')})
            if index in replacements and path_score(replacements[index]['q'], reference.dt/4) < path_score(item['q'], reference.dt/4):
                item = replacements[index]
            combined.append(item)
        results = combined
    else:
        indices = np.array([r['index'] for r in results])
    rows = []
    for r in results:
        for condition in ('before', 'after'):
            rows.append(dict(scan_index=r['index'], x_m=float(r['xy'][0]), y_m=float(r['xy'][1]),
                             condition=condition, yaw_deg=0. if condition=='before' else r['yaw'], **r[condition]))
    with (args.out/'comparison.csv').open('w') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    accepted = np.array([r['after']['screened_pass'] for r in results])
    arrays = dict(indices=indices, xy=scan['xy'][indices], time=t,
                  yaw_deg=np.array([r['yaw'] for r in results]), q=np.array([r['q'] for r in results]),
                  target_wrist_pos=np.array([r['pos'] for r in results]),
                  target_wrist_quat_xyzw=np.array([r['quat'] for r in results]), accepted=accepted)
    np.savez_compressed(args.out/'candidates.npz', **arrays)
    # A separate screened subset avoids accidentally replaying rejected candidates.
    np.savez_compressed(args.out/'screened_candidates.npz', time=t,
                        **{k: v[accepted] for k, v in arrays.items() if k!='time'})
    report = dict(reference=str(ref_path), reference_sha256=meta['reference_sha256'], model_sha256=meta['model_sha256'],
                  zero_yaw=str(args.zero_yaw), seed_count=args.seeds, max_nfev=args.max_nfev,
                  samples=len(t), dt=reference.dt/4, duration_s=float(t[-1]), placements=len(results),
                  newly_searched_indices=[int(job[0]) for job in jobs],
                  merge_previous=str(args.merge_previous) if args.merge_previous else None,
                  coarse_step_deg=args.coarse_step, refine_step_deg=args.refine_step, dense_angles=args.dense_angles,
                  workers=args.workers, elapsed_s=time.monotonic()-started, methods={},
                  limitations=['Offline reference planning, not closed-loop RL or grasp evaluation',
                               'Coarse centerline/AABB screen: no mesh thickness, self-collision or continuous collision certificate',
                               'Future path used for selection; approach to the first pose not tested',
                               'Yaw rotates the whole task frame; RL object-orientation/keypoint symmetry is NOT implemented',
                               'Sampled velocity screen is not acceleration/jerk or actuator validation'])
    for condition in ('before', 'after'):
        stats = [r[condition] for r in results]
        report['methods'][condition] = dict(
            strict_pose=sum(s['strict_pose'] for s in stats), coarse_clear=sum(s['coarse_clear'] for s in stats),
            screened_pass=sum(s['screened_pass'] for s in stats),
            max_speed=max(s['max_speed'] for s in stats),
            median_placement_max_speed=float(np.median([s['max_speed'] for s in stats])),
            max_step=max(s['max_step'] for s in stats), max_acceleration=max(s['max_acceleration'] for s in stats),
            max_position_error_m=max(s['max_position_error_m'] for s in stats),
            max_rotation_error_rad=max(s['max_rotation_error_rad'] for s in stats))
    report['worse_speed_count'] = sum(r['after']['max_speed'] > r['before']['max_speed']+1e-6 for r in results)
    report['worse_acceleration_count'] = sum(r['after']['max_acceleration'] > r['before']['max_acceleration']+1e-6 for r in results)
    report['lost_pass_indices'] = [r['index'] for r in results if r['before']['screened_pass'] and not r['after']['screened_pass']]
    report['new_pass_indices'] = [r['index'] for r in results if not r['before']['screened_pass'] and r['after']['screened_pass']]
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.out/'search_records.json').write_text(json.dumps([
        {k: v for k, v in r.items() if k in ('index', 'yaw', 'method', 'angles', 'dense_candidates', 'elapsed_s')}
        for r in results], indent=2)+'\n')
    plot_results(args.out, scan['xy'], scan['speed_screen_pass'], indices, results)
    print(json.dumps(report, indent=2))


def plot_results(out, all_xy, original_green, indices, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xy = all_xy[indices]
    passed = np.array([r['after']['screened_pass'] for r in results])
    old_pass = np.array([r['before']['screened_pass'] for r in results])
    angles = np.array([r['yaw'] for r in results])
    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    ax.scatter(*all_xy[original_green].T, s=9, color='#ddd', label='Other original green: not re-tested')
    ax.scatter(*xy[~passed].T, c='#e59628', s=18, label=f'Remain rejected: {(~passed).sum()}')
    dots = ax.scatter(*xy[passed].T, c=angles[passed], cmap='twilight', vmin=-180, vmax=180, s=23)
    wrist_vectors = np.array([r['pos'][0, :2]-r['xy'] for r in results])[passed]
    wrist_vectors /= np.maximum(np.linalg.norm(wrist_vectors, axis=1, keepdims=True), 1e-12)
    ax.quiver(xy[passed, 0], xy[passed, 1], wrist_vectors[:, 0], wrist_vectors[:, 1],
              angles='xy', scale_units='xy', scale=55, width=.002, alpha=.6)
    ax.scatter(*xy[old_pass].T, facecolors='none', edgecolors='black', s=40, linewidths=.5,
               label=f'Already passed with zero yaw: {old_pass.sum()}')
    fig.colorbar(dots, ax=ax, label='Whole-task yaw relative to original [deg, CCW]')
    ax.set(aspect='equal', xlabel='World X [m]', ylabel='World Y [m]',
           title=f'Approach-yaw search: {passed.sum()}/{len(xy)} original orange points pass\n120Hz IK + coarse workcell + <=10 rad/s; not physics validation\nArrows: can center toward initial wrist (XY)')
    ax.grid(alpha=.2); ax.legend(fontsize=8, loc='lower right')
    fig.savefig(out/'yaw_region.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    b = np.array([r['before']['max_speed'] for r in results]); a = np.array([r['after']['max_speed'] for r in results])
    ax.scatter(b, a, s=12); ax.plot([.1, b.max()], [.1, b.max()], '--', color='gray')
    ax.axhline(10, linestyle=':', color='green')
    ax.set(xscale='log', yscale='log', xlabel='Optimized zero-yaw max joint speed [rad/s]',
           ylabel='Yaw-selected max joint speed [rad/s]', title='Same 249 placements, same timing and strict pose tolerances')
    ax.grid(alpha=.2); fig.savefig(out/'speed_comparison.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
