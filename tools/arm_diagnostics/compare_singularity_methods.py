"""Opt-in, recorded-input IK comparison; no policy, physics, or default changes.

Reuse the verified mounted-wrist model, strict solver, offline branch search,
and command-bounded solver. Rotated tasks and different reset branches are
labelled separately from identical-target/identical-reset comparisons.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from tools.arm_diagnostics.search_tabletop_yaw import rotate_task
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.sequence_branch_ik import (
    centerline_clear, continue_path, distinct_solutions, improve_paths,
    path_score, seed_bank, workcell_boxes,
)
from tools.rb3_revo2_ik.velocity_bounded_ik import VelocityBoundedIK
from tools.rb3_revo2_ik.warm_start_ik import PoseJacobian

ROOT = Path(__file__).resolve().parents[2]


def load_episode(source, episode):
    """Read one episode, preserving reset and command/state timestamps."""
    source = Path(source)
    meta = json.loads((source / 'metadata.json').read_text())
    initial = next(s for s in meta['initial_states'] if s['episode'] == episode)
    rows = []
    with (source / 'physics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['episode'] == episode:
                rows.append(row)
    dt = float(meta['physics_dt'])
    if not rows or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Missing episode or invalid physics dt')
    with Path(meta['state_bank']).open() as stream:
        header = json.loads(next(stream))
    kin = RB3730Kinematics(base_position=header['base_position'],
                          base_quaternion_xyzw=header['base_quaternion_xyzw'])
    ids = [meta['joint_names'].index(n) for n in kin.joint_names]
    if [meta['joint_names'][i] for i in meta['arm_ids']] != list(kin.joint_names):
        raise ValueError('Raw IK joint indexing does not match model order')
    q0 = np.array(initial['all_q'])[ids]
    # Exact FK of saved reset joints removes only sub-micron reset serialization
    # noise. All real IK targets below are read unmodified from the trace.
    p0, r0 = kin.forward(q0)
    t = np.r_[0., [r['time_s'] for r in rows]]
    policy_t = np.array([r['command_time_s'] for r in rows])
    command_t = t[1:]-dt
    decimation = round(float(meta['control_dt'])/dt)
    expected_policy_t = np.arange(len(rows))//decimation * float(meta['control_dt'])
    if (not np.allclose(np.diff(t), dt, rtol=0, atol=1e-9) or
            not np.allclose(policy_t, expected_policy_t, rtol=0, atol=1e-9) or
            not np.array_equal([r['episode_step'] for r in rows], np.arange(1, len(t)))):
        raise ValueError('Missing/reordered timesteps or unexpected command timing')
    if meta['ik_policy_rate'] or not meta['arm_response_physics']:
        raise ValueError('This benchmark expects the recorded per-physics-step IK path')
    p = np.vstack([p0, [r['ik_input_pos'] for r in rows]])
    quat = np.vstack([r0, [r['ik_input_quat'] for r in rows]])
    q = np.vstack([q0, [r['raw_solve']['q'] for r in rows]])
    if (not all(np.isfinite(a).all() for a in (p, quat, q)) or
            not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-5) or
            not all(r['raw_solve']['success'] for r in rows)):
        raise ValueError('Expected finite successful original IK and unit XYZW quaternions')
    return kin, dict(t=t, command_t=command_t, policy_t=policy_t, p=p, quat=quat, q=q,
                     q_cmd=np.vstack([q0, [r['q_cmd'] for r in rows]]),
                     hand_target=np.array([r['hand_target'] for r in rows]),
                     object_initial=np.array(initial['object_state']),
                     speed=np.array(meta['recovery_runtime']['velocity_limit'])[ids],
                     dt=dt, meta=meta)


def bounded_path(kin, p, quat, q0, dt, speed, budget, acceleration):
    """Best feasible command, with explicit pose failures; never silently hold."""
    q = np.full((len(p), 6), np.nan)
    q[0] = q0
    previous_v = np.zeros(6)
    solver = VelocityBoundedIK(kin)
    failures = []
    for i in range(1, len(p)):
        try:
            result = solver.inverse(p[i], quat[i], command_q=q[i-1],
                velocity_limit=speed, dt=dt, position_lower=kin.joint_lower,
                position_upper=kin.joint_upper, max_nfev=budget,
                previous_velocity=previous_v, acceleration_limit=acceleration)
        except ValueError as error:
            failures.append(dict(frame=i, error=str(error)))
            break
        if not result.finite or not result.optimizer_success:
            failures.append(dict(frame=i, error=result.message))
            break
        q[i] = result.q
        previous_v = (q[i]-q[i-1])/dt
    return q, failures


def evaluate_path(kin, q, p, quat, t, speed, boxes, floor_z):
    dt = float(t[1]-t[0])
    if not np.isfinite(q).all():
        return dict(complete=False, strict_pose=False, screened_pass=False,
                    incomplete_frame_indices=np.where(~np.isfinite(q).all(axis=1))[0].tolist()), {}
    fk_p, fk_r = kin.forward_batch(q)
    pe = np.linalg.norm(fk_p-p, axis=1)
    oe = (Rotation.from_quat(quat).inv()*Rotation.from_quat(fk_r)).magnitude()
    v = np.diff(q, axis=0)/dt  # bounded joint coordinates, deliberately NO wrap
    a = np.diff(np.vstack([np.zeros(6), v]), axis=0)/dt  # reset velocity = 0
    step = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    sv = []
    for qi, pi, ri in zip(q, fk_p, fk_r):
        j = PoseJacobian(kin).jacobian(qi, pi, Rotation.from_quat(ri).as_matrix(), 10.)
        sv.append(np.linalg.svd(j, compute_uv=False)[-1])
    bad_clear = [i for i, qi in enumerate(q) if not centerline_clear(kin, qi, boxes, floor_z)]
    bad_edges = []
    for i in range(1, len(q)):
        # Additional centerline screen along each joint-linear edge, <= .02 rad
        # per sample. Still NOT full mesh, self, hand, or swept collision proof.
        count = max(2, int(np.ceil(step[i-1]/.02))+1)
        if any(not centerline_clear(kin, qi, boxes, floor_z)
               for qi in np.linspace(q[i-1], q[i], count)[1:-1]):
            bad_edges.append(i)
    limit_ok = bool(np.all(q >= kin.joint_lower-1e-9) and np.all(q <= kin.joint_upper+1e-9))
    pose_ok = (pe <= 1e-4) & (oe <= 1e-3)
    velocity_ok = bool(np.all(np.abs(v) <= speed+1e-8))
    stats = dict(complete=True, strict_pose=bool(pose_ok.all()),
        strict_failure_frames=np.where(~pose_ok)[0].tolist(),
        approximation_budget_met=bool(np.all(pe <= .005) and np.all(oe <= .05)),
        joint_limit_ok=limit_ok, velocity_ok=velocity_ok,
        coarse_clear=not bad_clear and not bad_edges,
        coarse_collision_frames=bad_clear, coarse_collision_edges=bad_edges,
        screened_pass=bool(pose_ok.all() and velocity_ok and limit_ok and not bad_clear and not bad_edges),
        max_step_rad=float(step.max()), max_step_sample_time_s=float(t[1:][step.argmax()]),
        max_step_command_time_s=float(t[1:][step.argmax()]-dt),
        max_speed_rad_s=float(np.abs(v).max()), max_acceleration_rad_s2=float(np.abs(a).max()),
        max_position_error_mm=float(pe.max()*1000), p95_position_error_mm=float(np.percentile(pe,95)*1000),
        max_rotation_error_deg=float(np.rad2deg(oe.max())),
        min_weighted_jacobian_sigma=float(min(sv)),
        # Axis alignment modulo pi is diagnostic only; q/v are never wrapped.
        min_wrist_axis_separation_deg=float(np.rad2deg(np.arcsin(np.abs(np.sin(q[:,4])))).min()),
        min_joint_margin_rad=float(np.minimum(q-kin.joint_lower, kin.joint_upper-q).min()),
        per_joint_max_speed_rad_s=np.abs(v).max(0).tolist(),
        per_joint_max_acceleration_rad_s2=np.abs(a).max(0).tolist())
    return stats, dict(fk_position=fk_p, fk_quat_xyzw=fk_r, position_error_m=pe,
                      rotation_error_rad=oe, path_velocity=v, path_acceleration=a,
                      min_weighted_jacobian_sigma=np.array(sv))


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite prior evidence
    kin, data = load_episode(args.source, args.episode)
    p, quat, q, dt = (data[k] for k in ('p', 'quat', 'q', 'dt'))
    layout = json.loads(args.layout.read_text())
    np.testing.assert_allclose(kin.base_position, layout['robot_mount']['position'], atol=1e-9)
    boxes = workcell_boxes(layout)
    clear = lambda qi: centerline_clear(kin, qi, boxes, layout['floor_z'])
    # Freeze all budgets before solving; selection never increases speed or time.
    experiment = dict(episode=args.episode, source=str(args.source.resolve()),
        checkpoint=data['meta']['checkpoint'], checkpoint_sha256=data['meta']['checkpoint_sha256'],
        dt=dt, sample_count=len(p), initial_can_xyz=data['object_initial'][:3].tolist(),
        joint_names=list(kin.joint_names), speed_limits=data['speed'].tolist(),
        seeds=args.seeds, max_nfev=args.max_nfev, yaws=args.yaws,
        acceleration_bound=250., joint_margin_rad=.1,
        strict_position_m=1e-4, strict_orientation_rad=1e-3,
        source_metadata_sha256=hashlib.sha256((args.source/'metadata.json').read_bytes()).hexdigest(),
        source_files={}, physics_executed=False, defaults_changed=False,
        timing='reset t=0; command_t=state_time-dt for per-physics IK; source command_time_s is POLICY time, saved as policy_t; no time shifting',
        limits_note='native speed from original runtime; 250 rad/s^2 is an experimental command bound',
        selection='strict FK + native speed + coarse workcell screen, then peak raw joint speed',
        unsupported='actual tracking, contact, grasp, full mesh/self collision, approach-to-reset, drive torque')
    for path in (ROOT/'tools/rb3_revo2_ik/rb3_model.json', args.layout):
        experiment['source_files'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Archive the exact selected numerical input, not the entire multi-episode log.
    np.savez_compressed(out/'inputs.npz', **{k:v for k,v in data.items() if k != 'meta'})
    experiment['inputs_sha256'] = hashlib.sha256((out/'inputs.npz').read_bytes()).hexdigest()
    (out/'experiment.json').write_text(json.dumps(experiment, indent=2)+'\n')
    results, plot_data = {}, {}

    def save(name, path, pos=p, rot=quat, **extra):
        stats, arrays = evaluate_path(kin, path, pos, rot, data['t'], data['speed'], boxes, layout['floor_z'])
        stats.update(extra)
        stats['same_reset_joints'] = bool(np.allclose(path[0], q[0], rtol=0, atol=1e-6))
        stats['same_target'] = bool(np.array_equal(pos, p) and np.array_equal(rot, quat))
        stats['initial_joint_change_rad'] = (float(np.max(np.abs(path[0]-q[0])))
                                             if np.isfinite(path[0]).all() else None)
        np.savez_compressed(out/f'{name}.npz', q=path, target_position=pos,
                            target_quat_xyzw=rot, sample_time=data['t'], **arrays)
        results[name] = stats
        if arrays:
            plot_data[name] = dict(q=path, **arrays)
        print(name, json.dumps({k:stats.get(k) for k in ('complete','screened_pass','max_step_rad',
            'max_position_error_mm','max_rotation_error_deg','coarse_clear','min_wrist_axis_separation_deg')}), flush=True)
        (out/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')

    save('recorded_raw_ik', q, category='original raw solver output, not actuator motion')
    save('recorded_command', data['q_cmd'], category='original postprocessed command, not actual')
    start = time.monotonic()
    local = continue_path(kin, p, quat, q[0], max_nfev=args.max_nfev)
    save('warm_start', local, category='same reset, same exact target')
    margin_model = copy.copy(kin)
    margin_model.joint_lower = kin.joint_lower+.1
    margin_model.joint_upper = kin.joint_upper-.1
    margin = continue_path(margin_model, p, quat, q[0], max_nfev=args.max_nfev)
    save('end_margin_01', margin, category='IK search bounds only; native limits unchanged')
    variants, branch_search = improve_paths(kin, p, quat, q, dt, args.seeds, args.max_nfev, clear)
    save('initial_branch', variants['multistart'], category='offline initial branch selection; may collide')
    save('screened_branch', variants['screened_best'], category='offline look-ahead + different reset branch')
    for acceleration in (0., 250.):
        bounded, failures = bounded_path(kin, p, quat, q[0], dt, data['speed'], args.max_nfev, acceleration)
        save('bounded_speed' if acceleration == 0 else 'bounded_speed_accel', bounded,
             category='same reset/target, explicit pose approximation', solver_failures=failures)
    # Entire task rotates about the stationary INITIAL can center; do not rotate
    # wrist alone or imply the unchanged frozen policy is yaw-equivariant.
    center = data['object_initial'][:3]
    for yaw in args.yaws:
        pos, rot = rotate_task(p, quat, center, center, yaw)
        seed = q[0].copy()
        seed[0] += np.deg2rad(yaw)
        starts = distinct_solutions(kin, pos[0], rot[0], seed_bank(kin, seed, args.seeds), args.max_nfev)
        paths = [continue_path(kin, pos, rot, s, max_nfev=args.max_nfev) for s in starts if clear(s)]
        paths = [a for a in paths if np.isfinite(a).all() and all(clear(qi) for qi in a)]
        selected = min(paths, key=lambda a:path_score(a, dt)) if paths else np.full_like(q, np.nan)
        save(f'yaw_{yaw:+g}', selected, pos, rot, category='rotated TASK + different initial branch',
             task_yaw_deg=yaw, first_pose_solutions=len(starts), clear_paths=len(paths),
             rotated_object_initial_quat_xyzw=(Rotation.from_euler('z',yaw,degrees=True)*
                 Rotation.from_quat(data['object_initial'][3:7])).as_quat().tolist())
    experiment.update(elapsed_s=time.monotonic()-start, branch_search=branch_search)
    (out/'experiment.json').write_text(json.dumps(experiment, indent=2)+'\n')
    fields = ['method','same_target','same_reset_joints','complete','strict_pose','coarse_clear',
              'velocity_ok','screened_pass','max_step_rad','max_speed_rad_s','max_acceleration_rad_s2',
              'max_position_error_mm','max_rotation_error_deg','min_wrist_axis_separation_deg']
    with (out/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, stats in results.items():
            writer.writerow(dict(method=name, **stats))
    plot(out, data['t'], plot_data, results)
    print(f'Finished: {out} ({experiment["elapsed_s"]:.1f} s)', flush=True)


def plot(out, t, paths, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in ('recorded_raw_ik','screened_branch','bounded_speed_accel') if n in paths]
    good_yaws = [n for n,s in results.items() if n.startswith('yaw_') and s.get('screened_pass')]
    if good_yaws:
        names.append(min(good_yaws, key=lambda n: results[n]['max_speed_rad_s']))
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for name in names:
        d = paths[name]
        for ax, j in zip(axes.flat[:3], (3,4,5)):
            ax.plot(t, np.rad2deg(d['q'][:,j]), label=name)
        axes[1,1].plot(t[1:], np.max(np.abs(np.diff(d['q'],axis=0)),axis=1), label=name)
        axes[2,0].plot(t, 1000*d['position_error_m'], label=name)
        axes[2,1].plot(t, np.rad2deg(d['rotation_error_rad']), label=name)
    for ax, title in zip(axes.flat, ('wrist1 [deg]','wrist2 [deg]','wrist3 [deg]',
                                    'Maximum raw joint step [rad / 1/120 s]',
                                    'Target-to-FK position error [mm]', 'Target-to-FK orientation error [deg]')):
        ax.set_title(title)
        ax.set_xlabel('Sample-aligned time [s]; command issued one dt earlier')
        ax.grid(alpha=.25)
    axes[1,1].axhline(10/120, color='black', linestyle=':', label='10 rad/s at 120 Hz')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Recorded singularity episode: offline IK methods\nYaw changes the whole task; alternative branches change reset. Not a physics/grasp evaluation.')
    fig.savefig(out/'comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=11)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layout', type=Path, default=ROOT/'config/workcell/rb3_revo2_table.json')
    parser.add_argument('--seeds', type=int, default=24)
    parser.add_argument('--max-nfev', type=int, default=300)
    parser.add_argument('--yaws', nargs='+', type=float, default=[-60.,-30.,-15.,15.,30.,60.,90.])
    args = parser.parse_args()
    if args.seeds < 4 or args.max_nfev < 1 or not np.isfinite(args.yaws).all():
        parser.error('Invalid seed/iteration/yaw budget')
    run(args)


if __name__ == '__main__':
    main()
