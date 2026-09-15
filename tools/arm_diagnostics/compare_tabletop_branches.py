"""Bounded, offline comparison on the previously measured orange placements."""
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

from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.reference_trajectory import load_reference_trajectory
from tools.rb3_revo2_ik.sequence_branch_ik import (
    centerline_clear, continue_path, improve_paths, path_score, workcell_boxes,
)
from tools.rb3_revo2_ik.velocity_bounded_ik import VelocityBoundedIK

ROOT = Path(__file__).resolve().parents[2]


def metrics(kin, q, pos, quat, dt, boxes, floor, speed):
    if not np.isfinite(q).all():
        return dict(strict_pose=False, coarse_clear=False, screened_pass=False,
                    max_speed=None, max_step=None, max_acceleration=None,
                    max_position_error_m=None, max_rotation_error_rad=None)
    fk, fkr = kin.forward_batch(q)
    pe = np.linalg.norm(fk-pos, axis=1)
    oe = (Rotation.from_quat(quat).inv()*Rotation.from_quat(fkr)).magnitude()
    limit_ok = bool(np.all(q >= kin.joint_lower-1e-9) and np.all(q <= kin.joint_upper+1e-9))
    strict = bool(pe.max() <= 1e-4 and oe.max() <= 1e-3 and limit_ok)
    clear = bool(all(centerline_clear(kin, qi, boxes, floor) for qi in q))
    v, a, _ = path_score(q, dt)
    return dict(strict_pose=strict, coarse_clear=clear, screened_pass=bool(strict and clear and v <= speed+1e-8),
                max_speed=v, max_step=v*dt, max_acceleration=a,
                max_position_error_m=float(pe.max()), max_rotation_error_rad=float(oe.max()),
                joint_limit_ok=limit_ok,
                max_speed_joint=int(np.unravel_index(np.abs(np.diff(q, axis=0)).argmax(), (len(q)-1, 6))[1]),
                min_joint_margin_rad=float(np.min(np.minimum(q-kin.joint_lower, kin.joint_upper-q))))


def run_one(job):
    index, xy, baseline, pos, quat, dt, layout, seeds, budget, speed = job
    kin = RB3730Kinematics(base_position=layout['robot_mount']['position'],
                          base_quaternion_xyzw=layout['robot_mount']['quaternion_xyzw'])
    boxes = workcell_boxes(layout)
    clear = lambda q: centerline_clear(kin, q, boxes, layout['floor_z'])
    start = time.monotonic()
    variants, search = improve_paths(kin, pos, quat, baseline, dt, seeds, budget, clear)
    variants = dict(baseline=baseline, **variants)
    seed = variants['screened_best'][0]
    bounded_q = np.full_like(baseline, np.nan)
    bounded_converged = True
    if np.isfinite(seed).all():
        bounded_q[0] = seed
        solver = VelocityBoundedIK(kin)
        for frame in range(1, len(pos)):
            result = solver.inverse(pos[frame], quat[frame], command_q=bounded_q[frame-1],
                velocity_limit=np.full(6, speed), dt=dt,
                position_lower=kin.joint_lower, position_upper=kin.joint_upper,
                position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3, max_nfev=budget)
            if not result.finite:
                bounded_converged = False
                break
            bounded_converged = bounded_converged and result.optimizer_success
            # Save the attempted bounded solution even if pose tolerance fails.
            # FK metrics below explicitly reject such a trajectory.
            bounded_q[frame] = result.q
    variants['bounded_attempt'] = bounded_q
    stats = {name: metrics(kin, q, pos, quat, dt, boxes, layout['floor_z'], speed)
             for name, q in variants.items()}
    stats['bounded_attempt']['optimizer_success'] = bool(bounded_converged)
    if not bounded_converged:
        stats['bounded_attempt']['strict_pose'] = False
        stats['bounded_attempt']['screened_pass'] = False
    eligible = [name for name in ('screened_best', 'bounded_attempt', 'baseline')
                if stats[name]['strict_pose'] and stats[name]['coarse_clear']]
    winner = min(eligible, key=lambda name: path_score(variants[name], dt)) if eligible else 'baseline'
    variants['selected'] = variants[winner].copy()
    stats['selected'] = stats[winner].copy()
    search.update(selected_method=winner, elapsed_s=time.monotonic()-start)
    return index, xy, variants, stats, search


def run_dense_one(job):
    index, xy, starts, knots, pos, quat, dt, layout, budget, speed = job
    kin = RB3730Kinematics(base_position=layout['robot_mount']['position'],
                          base_quaternion_xyzw=layout['robot_mount']['quaternion_xyzw'])
    boxes = workcell_boxes(layout)
    paths, stats = {}, {}
    for name, q0 in starts.items():
        q = continue_path(kin, pos, quat, q0, max_nfev=budget)
        paths[name] = q
        stats[name] = metrics(kin, q, pos, quat, dt, boxes, layout['floor_z'], speed)
        stats[name]['max_original_knot_q_difference_rad'] = (float(np.max(np.abs(q[::4]-knots[name])))
                                                          if np.isfinite(q).all() else None)
    return index, xy, paths, stats


def dense_validation(args, scan_summary, reference, layout):
    source = np.load(args.candidates)
    scan = np.load(args.scan/'solutions.npz')
    expected = np.flatnonzero(scan['full_sequence_ik'] & ~scan['speed_screen_pass'])
    np.testing.assert_array_equal(source['indices'], expected)
    np.testing.assert_allclose(source['baseline'], scan['q'][expected], atol=0, rtol=0)
    times = np.arange(reference.frames)*reference.dt
    dense_t = np.arange(4*(reference.frames-1)+1)*reference.dt/4
    dense_p = np.column_stack([np.interp(dense_t, times, reference.wrist_pos[:, j]) for j in range(3)])
    dense_q = Slerp(times, Rotation.from_quat(reference.wrist_quat_xyzw))(dense_t).as_quat()
    jobs = [(int(i), xy, {name: source[name][j, 0] for name in ('baseline', 'selected')},
             {name: source[name][j] for name in ('baseline', 'selected')},
             dense_p+np.r_[xy-reference.object_pos[0, :2], 0.], dense_q, reference.dt/4,
             layout, args.max_nfev, scan_summary['speed_screen_rad_s'])
            for j, (i, xy) in enumerate(zip(source['indices'], source['xy']))]
    rows, paths, results = [], {'baseline': [], 'selected': []}, []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, xy, configs, stats in pool.map(run_dense_one, jobs):
            results.append(stats)
            for name in paths:
                paths[name].append(configs[name])
                rows.append(dict(scan_index=index, x_m=float(xy[0]), y_m=float(xy[1]), method=name, **stats[name]))
            if len(results)%25 == 0 or len(results)==len(jobs):
                print(f'120Hz continuation {len(results)}/{len(jobs)}; selected pass='
                      f'{sum(r["selected"]["screened_pass"] for r in results)}', flush=True)
    with (args.out/'comparison.csv').open('w') as f:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.out/'paths.npz', indices=source['indices'], xy=source['xy'], time=dense_t,
                        **{name: np.array(q) for name, q in paths.items()})
    report = dict(mode='120Hz single-seed continuation from selected first pose, no state resets between knots',
                  interpolation='world position linear / quaternion SLERP, identical for both conditions',
                  candidate_file=str(args.candidates.resolve()),
                  reference_sha256=scan_summary['reference_sha256'],
                  samples=len(dense_t), dt=reference.dt/4, duration_s=float(dense_t[-1]),
                  elapsed_s=time.monotonic()-started, methods={})
    for name in paths:
        valid = [r[name] for r in results if r[name]['max_speed'] is not None]
        report['methods'][name] = dict(
            finite_complete_paths=len(valid), strict_pose=sum(r[name]['strict_pose'] for r in results),
            coarse_clear=sum(r[name]['coarse_clear'] for r in results),
            screened_pass=sum(r[name]['screened_pass'] for r in results),
            median_placement_max_speed=float(np.median([r['max_speed'] for r in valid])),
            max_speed=max(r['max_speed'] for r in valid), max_step=max(r['max_step'] for r in valid),
            max_acceleration=max(r['max_acceleration'] for r in valid),
            max_position_error_m=max(r['max_position_error_m'] for r in valid),
            max_rotation_error_rad=max(r['max_rotation_error_rad'] for r in valid))
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seeds', type=int, default=24)
    p.add_argument('--max-nfev', type=int, default=300)
    p.add_argument('--candidates', type=Path,
                   help='Validate saved paths.npz initial configurations on identical 120Hz targets instead of searching')
    p.add_argument('--dense-search-baseline', type=Path,
                   help='Search at 120Hz using the baseline from an earlier --candidates validation')
    args = p.parse_args()
    if args.workers < 1 or args.seeds < 4 or args.max_nfev < 1:
        p.error('workers/budget must be positive; seeds must be >=4')
    args.out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.scan/'summary.json').read_text())
    reference_path = Path(summary['reference'])
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != summary['reference_sha256']:
        raise ValueError('Reference differs from original scan')
    if hashlib.sha256((ROOT/'tools/rb3_revo2_ik/rb3_model.json').read_bytes()).hexdigest() != summary['model_sha256']:
        raise ValueError('Robot model differs from original scan')
    reference = load_reference_trajectory(reference_path)
    scan = np.load(args.scan/'solutions.npz')
    layout = json.loads((ROOT/'config/workcell/rb3_revo2_table.json').read_text())
    if layout != summary['workcell']:
        raise ValueError('Workcell differs from original scan')
    if args.candidates:
        dense_validation(args, summary, reference, layout)
        return
    orange = np.flatnonzero(scan['full_sequence_ik'] & ~scan['speed_screen_pass'])
    target_pos, target_quat, target_dt = reference.wrist_pos, reference.wrist_quat_xyzw, reference.dt
    baselines = scan['q'][orange]
    if args.dense_search_baseline:
        dense = np.load(args.dense_search_baseline)
        np.testing.assert_array_equal(dense['indices'], orange)
        np.testing.assert_allclose(dense['xy'], scan['xy'][orange], atol=0, rtol=0)
        np.testing.assert_allclose(dense['baseline'][:, 0], baselines[:, 0], atol=1e-5, rtol=0)
        dense_meta = json.loads((args.dense_search_baseline.parent/'summary.json').read_text())
        if dense_meta['reference_sha256'] != summary['reference_sha256']:
            raise ValueError('Dense baseline reference mismatch')
        t = np.arange(reference.frames)*reference.dt
        dense_t = np.arange(4*(reference.frames-1)+1)*reference.dt/4
        np.testing.assert_allclose(dense['time'], dense_t, atol=1e-12, rtol=0)
        target_pos = np.column_stack([np.interp(dense_t, t, reference.wrist_pos[:, j]) for j in range(3)])
        target_quat = Slerp(t, Rotation.from_quat(reference.wrist_quat_xyzw))(dense_t).as_quat()
        target_dt = reference.dt/4
        baselines = dense['baseline']
    jobs = [(int(i), scan['xy'][i], baselines[j],
             target_pos + np.r_[scan['xy'][i]-reference.object_pos[0, :2], 0.],
             target_quat, target_dt, layout, args.seeds, args.max_nfev,
             summary['speed_screen_rad_s']) for j, i in enumerate(orange)]
    started = time.monotonic()
    rows, results, paths, searches = [], [], {}, []
    print(f'Frozen orange placements={len(jobs)}, first/last seed attempts={args.seeds} each', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, xy, variants, stats, search in pool.map(run_one, jobs, chunksize=1):
            results.append(stats); searches.append(search)
            for name, q in variants.items():
                paths.setdefault(name, []).append(q)
                rows.append(dict(scan_index=index, x_m=float(xy[0]), y_m=float(xy[1]), method=name, **stats[name]))
            if len(results) % 10 == 0 or len(results) == len(jobs):
                print(f'{len(results)}/{len(jobs)}: screened pass={sum(r["selected"]["screened_pass"] for r in results)}, '
                      f'elapsed={time.monotonic()-started:.1f}s', flush=True)
    with (args.out/'comparison.csv').open('w') as f:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.out/'paths.npz', indices=orange, xy=scan['xy'][orange], dt=target_dt,
                        **{name: np.asarray(q) for name, q in paths.items()})
    report = dict(scan=str(args.scan.resolve()), reference_sha256=summary['reference_sha256'],
                  model_sha256=summary['model_sha256'], placements=len(jobs), seeds=args.seeds,
                  max_nfev=args.max_nfev, dt=target_dt, samples=len(target_pos),
                  dense_search_baseline=str(args.dense_search_baseline) if args.dense_search_baseline else None,
                  speed_limit=summary['speed_screen_rad_s'],
                  elapsed_s=time.monotonic()-started, methods={}, searches=searches,
                  limitations=['Offline full-reference look-ahead, NOT closed-loop policy execution',
                               'Finite target samples at the recorded dt; between-sample motion is not certified',
                               'Centerline/AABB screen only: no robot thickness, self-collision or PhysX validation',
                               'Different first arm configurations need a separate collision-free approach',
                               'No time scaling, target modification, angle wrapping or native limit changes',
                               'Best among finite seed/branch candidates, not proof of global optimum'])
    for name in paths:
        data = [r[name] for r in results]
        valid = [r for r in data if r['max_speed'] is not None]
        report['methods'][name] = dict(
            finite_complete_paths=len(valid),
            strict_pose=sum(r['strict_pose'] for r in data), coarse_clear=sum(r['coarse_clear'] for r in data),
            screened_pass=sum(r['screened_pass'] for r in data),
            max_speed=float(max(r['max_speed'] for r in valid)),
            median_placement_max_speed=float(np.median([r['max_speed'] for r in valid])),
            max_step=float(max(r['max_step'] for r in valid)),
            max_acceleration=float(max(r['max_acceleration'] for r in valid)),
            max_position_error_m=float(max(r['max_position_error_m'] for r in valid)),
            max_rotation_error_rad=float(max(r['max_rotation_error_rad'] for r in valid)))
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    before = np.array([r['baseline']['max_speed'] if r['baseline']['max_speed'] is not None else np.nan for r in results])
    after = np.array([r['selected']['max_speed'] if r['selected']['max_speed'] is not None else np.nan for r in results])
    passed = np.array([r['selected']['screened_pass'] for r in results])
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for ax, heading in zip(axs, ('Before: frozen orange placements', 'After: strict pose + coarse workcell screen')):
        green = scan['speed_screen_pass']
        ax.scatter(*scan['xy'][green].T, s=8, color='#cccccc', label='Original green (not changed)')
        ax.set(xlabel='World X [m]', ylabel='World Y [m]', title=heading, aspect='equal')
        ax.grid(alpha=.2)
    axs[0].scatter(*scan['xy'][orange].T, s=14, color='#e39820', label=f'Original orange: {len(orange)}')
    axs[1].scatter(*scan['xy'][orange[passed]].T, s=14, color='#21885d', label=f'Now <=10 rad/s: {passed.sum()}')
    axs[1].scatter(*scan['xy'][orange[~passed]].T, s=14, color='#e39820', label=f'Remain: {(~passed).sum()}')
    for ax in axs: ax.legend(fontsize=8)
    fig.savefig(args.out/'region_comparison.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    ax.scatter(before, after, s=14); lim=max(np.nanmax(before), np.nanmax(after))*1.03
    ax.plot([0, lim], [0, lim], '--', color='gray'); ax.axhline(10, color='#21885d', linestyle=':')
    ax.set(xlabel='Baseline maximum joint speed [rad/s]', ylabel='Selected maximum joint speed [rad/s]',
           title=f'Same target poses, same {1/target_dt:g}Hz timing; {len(orange)} placements')
    ax.grid(alpha=.2); fig.savefig(args.out/'speed_comparison.png', dpi=180); plt.close(fig)
    worst = int(np.nanargmax(before))
    t = np.arange(len(target_pos))*target_dt
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), layout='constrained')
    for j, ax in enumerate(axs.flat):
        ax.plot(t, paths['baseline'][worst][:, j], label='Baseline')
        ax.plot(t, paths['selected'][worst][:, j], label='Selected')
        ax.set(title=RB3730Kinematics().joint_names[j], xlabel='Time [s]', ylabel='Angle [rad]')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f'Worst original speed placement XY={scan["xy"][orange[worst]].round(2)}; raw joint coordinates')
    fig.savefig(args.out/'worst_placement_joints.png', dpi=180); plt.close(fig)
    print(json.dumps(report['methods'], indent=2))


if __name__ == '__main__':
    main()
