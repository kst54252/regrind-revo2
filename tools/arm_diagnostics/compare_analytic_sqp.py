"""Recorded-input analytic-all-branch vs constrained SLSQP experiment (no Isaac).

Preserves target poses, reset/time conventions and native limits. Free-reset
paths are labelled separately. Failed SQP iterates remain diagnostic data only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy

from tools.arm_diagnostics.compare_singularity_methods import load_episode, evaluate_path
from tools.rb3_revo2_ik.analytic_branch_ik import AnalyticBranchIK
from tools.rb3_revo2_ik.sequence_branch_ik import centerline_clear, minimax_path, workcell_boxes
from tools.rb3_revo2_ik.sqp_pose_ik import solve_sqp

ROOT = Path(__file__).resolve().parents[2]


def sqp_path(kin, data, initial_q, *, stop_on_infeasible=False, **settings):
    q = np.full_like(data['q'], np.nan)
    q[0] = initial_q
    velocity = np.zeros(6)
    records = []
    for i in range(1, len(q)):
        try:
            candidate, info = solve_sqp(kin, data['p'][i], data['quat'][i], q[i-1], velocity,
                                        data['dt'], data['speed'], **settings)
        except ValueError as error:
            records.append(dict(frame=i, accepted=False, error=str(error)))
            break
        info.update(frame=i, command_time_s=float(data['command_t'][i-1]))
        records.append(info)
        if not np.isfinite(candidate).all() or not info['box_feasible']:
            break
        # For the offline comparison continue from bounded failed iterates so
        # later failures are visible. NOT a live command acceptance/fallback.
        q[i] = candidate
        velocity = (q[i]-q[i-1])/data['dt']
        if stop_on_infeasible and not info['feasible']:
            break
    return q, records


def branch_layers(kin, data, boxes, floor):
    solver = AnalyticBranchIK(kin)
    all_layers, screened, info, ids = [], [], [], []
    for i, (p, r) in enumerate(zip(data['p'], data['quat'])):
        start = time.perf_counter()
        result = solver.inverse_all(p, r)
        solve_s = time.perf_counter()-start
        all_layers.append(result.q)
        ids.append(result.branch_ids)
        clear_by_branch = {}
        keep = []
        for q, branch in zip(result.q, result.branch_ids):
            key = tuple(branch)
            if key not in clear_by_branch:
                clear_by_branch[key] = centerline_clear(kin, q, boxes, floor)
            keep.append(clear_by_branch[key])
        screened.append(result.q[np.asarray(keep, bool)])
        info.append(dict(frame=i, geometric_count=result.geometric_count,
                         bounded_coordinate_count=len(result.q),
                         coarse_clear_count=len(screened[-1]),
                         exhaustive_isolated=result.exhaustive_isolated,
                         singular_families=result.singular_families, solve_s=solve_s))
    return all_layers, screened, ids, info


def run(args):
    kin, data = load_episode(args.source, args.episode)
    layout = json.loads(args.layout.read_text())
    np.testing.assert_allclose(kin.base_position, layout['robot_mount']['position'], atol=1e-9)
    boxes, floor = workcell_boxes(layout), layout['floor_z']
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = dict(source=str(args.source.resolve()), episode=args.episode,
        checkpoint=data['meta']['checkpoint'], checkpoint_sha256=data['meta']['checkpoint_sha256'],
        dt=data['dt'], samples=len(data['q']), joint_names=list(kin.joint_names),
        native_speed_limits=data['speed'].tolist(), acceleration_bound_rad_s2=250.,
        strict_position_m=1e-4, strict_orientation_rad=1e-3,
        relaxed_position_m=.005, relaxed_orientation_rad=.05,
        optional_wrist_axis_margin_deg=5., sqp_maxiter=args.maxiter, sqp_ftol=1e-9,
        scipy_version=scipy.__version__, physics_executed=False, defaults_changed=False,
        target_changed=False, time_scaled=False,
        reset='same-reset and freely selected reset are separate; initial path velocity zero',
        timing='inputs command_t = recorded post-step time minus physics_dt; policy_t is separate',
        analytic='all isolated shoulder/elbow/wrist solutions plus ALL legal 2*pi representatives; exact continua reported incomplete',
        graph='offline minimax raw coordinate step, tie-break sum squared steps; screens vertices then validates selected edges at <=.02 rad',
        sqp='causal per-step q/v/a box, nonlinear position/rotation ball constraints; failed bounded iterates retained for diagnostics ONLY',
        unsupported='physics tracking, grasp, full mesh/self collision, approach to alternative reset, solver torque',
        source_hashes={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            args.source/'metadata.json', kin.model_config_path, args.layout,
            ROOT/'tools/rb3_revo2_ik/analytic_branch_ik.py', ROOT/'tools/rb3_revo2_ik/sqp_pose_ik.py')})
    np.savez_compressed(out/'inputs.npz', **{k:v for k,v in data.items() if k != 'meta'})
    frozen['input_sha256'] = hashlib.sha256((out/'inputs.npz').read_bytes()).hexdigest()
    (out/'experiment.json').write_text(json.dumps(frozen, indent=2)+'\n')
    results, plots = {}, {}

    def save(name, q, records=None, **extra):
        stats, arrays = evaluate_path(kin, q, data['p'], data['quat'], data['t'],
                                     data['speed'], boxes, floor)
        stats.update(extra)
        stats['same_reset'] = bool(np.allclose(q[0], data['q'][0], atol=1e-6, rtol=0))
        stats['initial_joint_difference_rad'] = (q[0]-data['q'][0]).tolist()
        if records is not None:
            failed = [row['frame'] for row in records if not row['accepted']]
            feasible = [row['frame'] for row in records if row.get('feasible', False)]
            timing = np.array([row['solve_s'] for row in records if 'solve_s' in row])
            stats.update(sqp_unaccepted_frames=failed, sqp_feasible_frames=feasible,
                         sqp_solver_all_accepted=not failed and len(records)==len(q)-1,
                         sqp_p95_solve_ms=float(np.percentile(timing,95)*1000) if len(timing) else None,
                         sqp_max_solve_ms=float(timing.max()*1000) if len(timing) else None)
            (out/f'{name}_solver.json').write_text(json.dumps(records, indent=2)+'\n')
        if stats.get('complete'):
            stats['acceleration_bound_ok'] = stats['max_acceleration_rad_s2'] <= 250.+1e-5
        np.savez_compressed(out/f'{name}.npz', q=q, **arrays)
        results[name] = stats
        plots[name] = dict(q=q, **arrays)
        (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(name, json.dumps({k:stats[k] for k in (
            'strict_pose','screened_pass','max_step_rad','max_speed_rad_s','max_acceleration_rad_s2',
            'max_position_error_mm','max_rotation_error_deg','coarse_clear','same_reset') if k in stats}), flush=True)

    save('recorded_raw_ik', data['q'])
    start = time.perf_counter()
    layers, screened, ids, enumeration = branch_layers(kin, data, boxes, floor)
    (out/'branch_enumeration.json').write_text(json.dumps(enumeration, indent=2)+'\n')
    np.savez_compressed(out/'all_branches.npz', q=np.concatenate(layers), branch_ids=np.concatenate(ids),
                        offsets=np.r_[0, np.cumsum([len(layer) for layer in layers])])
    print(f'Analytic enumeration: {time.perf_counter()-start:.3f}s, '
          f'geometric counts {sorted(set(r["geometric_count"] for r in enumeration))}, '
          f'periodic counts {sorted(set(r["bounded_coordinate_count"] for r in enumeration))}', flush=True)
    # Exact continua cannot be silently dropped from a claimed exhaustive graph.
    exhaustive = all(row['exhaustive_isolated'] for row in enumeration)
    paths = {}
    for name, nodes in (
            ('analytic_free_unscreened', layers),
            ('analytic_same_reset', [data['q'][:1]]+screened[1:]),
            ('analytic_free_screened', screened)):
        start = time.perf_counter()
        path = (minimax_path(nodes, data['dt']) if exhaustive and all(len(x) for x in nodes)
                else np.full_like(data['q'], np.nan))
        paths[name] = path
        save(name, path, exhaustive_isolated=exhaustive, graph_s=time.perf_counter()-start)

    specs = [('sqp_strict_same_reset', data['q'][0], 1e-4, 1e-3, 0.),
             ('sqp_relaxed_same_reset', data['q'][0], .005, .05, 0.),
             ('sqp_relaxed_axis5_same_reset', data['q'][0], .005, .05, 5.)]
    if np.isfinite(paths['analytic_free_screened']).all():
        specs.extend([('sqp_strict_free_reset', paths['analytic_free_screened'][0], 1e-4, 1e-3, 0.),
                      ('sqp_relaxed_free_reset', paths['analytic_free_screened'][0], .005, .05, 0.)])
    for name, initial, pt, rt, axis in specs:
        start = time.perf_counter()
        path, records = sqp_path(kin, data, initial, position_tolerance=pt,
                                orientation_tolerance=rt, axis_margin_deg=axis, maxiter=args.maxiter)
        save(name, path, records, position_constraint_m=pt, orientation_constraint_rad=rt,
             axis_constraint_deg=axis, elapsed_s=time.perf_counter()-start)
    plot(out, data, plots)
    return results


def plot(out, data, paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['recorded_raw_ik', 'analytic_same_reset', 'analytic_free_screened',
             'sqp_strict_same_reset', 'sqp_relaxed_same_reset', 'sqp_strict_free_reset']
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    t = data['command_t']
    for name in names:
        item = paths.get(name, {})
        if 'path_velocity' not in item:
            continue
        label = name.replace('_', ' ')
        axes[0,0].plot(t, item['position_error_m'][1:]*1000, label=label)
        axes[0,1].plot(t, np.rad2deg(item['rotation_error_rad'][1:]))
        axes[1,0].plot(t, item['path_velocity'][:,3])
        axes[1,1].plot(t, item['path_velocity'][:,5])
        axes[2,0].plot(t, np.max(np.abs(item['path_acceleration']),axis=1))
        axes[2,1].plot(t, np.rad2deg(np.arcsin(np.abs(np.sin(item['q'][1:,4])))))
    for ax, ylabel in zip(axes.flat, ['FK position error [mm]', 'FK rotation error [deg]',
                                      'wrist1 path speed [rad/s]', 'wrist3 path speed [rad/s]',
                                      'max path acceleration [rad/s^2]', 'wrist axis separation [deg]']):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.axvline(1.0333333333, color='gray', linestyle=':', alpha=.5)
    for ax in axes[1]:
        ax.axhline(10, color='red', linestyle='--', alpha=.4)
        ax.axhline(-10, color='red', linestyle='--', alpha=.4)
    for ax in axes[-1]:
        ax.set_xlabel('Actual IK command time [s]; no retiming')
    axes[2,0].axhline(250, color='red', linestyle='--', alpha=.4)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Same recorded target: closed-form all branches / constrained SLSQP\n'
                 'Offline kinematics only; failed SQP iterates are NOT accepted runtime commands')
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=11)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layout', type=Path, default=ROOT/'config/workcell/rb3_revo2_table.json')
    parser.add_argument('--maxiter', type=int, default=150)
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error('--maxiter must be positive')
    run(args)


if __name__ == '__main__':
    main()
