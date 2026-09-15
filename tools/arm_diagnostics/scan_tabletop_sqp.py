"""Whole-table XY screening with the existing experimental SQP pose solver.

One recorded IK-input trajectory, fixed yaw/timing. Not a policy rollout or
grasp-success map. One deterministic, clear analytic reset per XY for both
tolerances; no all-start search or runtime defaults are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from tools.arm_diagnostics.compare_analytic_sqp import sqp_path
from tools.arm_diagnostics.compare_singularity_methods import load_episode
from tools.rb3_revo2_ik.analytic_branch_ik import AnalyticBranchIK
from tools.rb3_revo2_ik.sequence_branch_ik import centerline_clear, workcell_boxes

ROOT = Path(__file__).resolve().parents[2]
MODES = (('strict', 1e-4, 1e-3), ('relaxed', .005, .05))
LABELS = {
    'pass': 'Full SQP + constraints + coarse workcell screen',
    'constraint_failure': 'SQP constraint unsolved (not proof of unreachable)',
    'optimizer_failure': 'Constraints met, optimizer convergence not confirmed',
    'workcell_intersection': 'Arm centerline intersects workcell',
    'reset_unsolved': 'No legal isolated reset IK',
    'reset_singular_uncertain': 'Reset has an unenumerated continuous IK family',
}
COLORS = dict(pass_color='#198754', constraint_failure='#df783c', optimizer_failure='#e0b832',
              workcell_intersection='#916aa9', reset_unsolved='#c4c8cd',
              reset_singular_uncertain='#5287a4')


def table_grid(layout, vertices, quaternion, spacing):
    """World-anchored lattice with the *whole initial mesh footprint* on table."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be positive and finite')
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Expected nonempty finite mesh vertices (N,3)')
    lower = np.asarray(layout['table']['center_xy'])-np.asarray(layout['table']['size_xy'])/2
    upper = np.asarray(layout['table']['center_xy'])+np.asarray(layout['table']['size_xy'])/2
    footprint = Rotation.from_quat(quaternion).apply(vertices)[:, :2]
    fmin, fmax = footprint.min(0), footprint.max(0)
    axes = [np.arange(np.ceil((lower[j]-fmin[j]-1e-9)/spacing),
                      np.floor((upper[j]-fmax[j]+1e-9)/spacing)+1)*spacing for j in range(2)]
    xy = np.array([(x,y) for y in axes[1] for x in axes[0]]).reshape(-1,2)
    if not len(xy):
        raise ValueError('No mesh-safe table grid positions')
    return xy, lower, upper, fmin, fmax


def select_reset(kin, p, quat, original_q, boxes, floor):
    """Nearest raw bounded-coordinate reset, after coarse centerline screening."""
    result = AnalyticBranchIK(kin).inverse_all(p, quat)
    if not result.exhaustive_isolated:
        return None, dict(status='reset_singular_uncertain', reset_geometric_count=result.geometric_count)
    if not len(result.q):
        return None, dict(status='reset_unsolved', reset_geometric_count=0)
    clear = {}
    keep = []
    for q, branch in zip(result.q, result.branch_ids):
        key = tuple(branch)
        if key not in clear:
            clear[key] = centerline_clear(kin, q, boxes, floor)
        keep.append(clear[key])
    valid = np.flatnonzero(keep)
    if not len(valid):
        return None, dict(status='workcell_intersection', reset_geometric_count=result.geometric_count)
    index = valid[np.argmin(np.sum((result.q[valid]-original_q)**2, axis=1))]
    return result.q[index], dict(status='reset_ready', reset_geometric_count=result.geometric_count,
                                reset_branch=result.branch_ids[index].tolist())


def assess_path(kin, q, p, quat, records, dt, speed, boxes, floor, pt, rt):
    """Independent FK/finite-difference check; never pass an untested suffix."""
    measured = np.isfinite(q).all(axis=1)
    count = int(measured.sum())
    errors = np.full((len(q),2), np.nan)
    if not np.array_equal(measured, np.arange(len(q)) < count):
        raise ValueError('Expected a contiguous evaluated prefix')
    pe, oe, v, acc, axis = np.empty(0), np.empty(0), np.empty((0,6)), np.empty((0,6)), np.empty(0)
    bad_clear, bad_edge = [], []
    if count:
        actual_p, actual_r = kin.forward_batch(q[:count])
        pe = np.linalg.norm(actual_p-p[:count],axis=1)
        oe = (Rotation.from_quat(quat[:count]).inv()*Rotation.from_quat(actual_r)).magnitude()
        errors[:count] = np.column_stack([pe,oe])
        v = np.diff(q[:count],axis=0)/dt
        acc = np.diff(np.vstack([np.zeros(6),v]),axis=0)/dt
        axis = np.rad2deg(np.arcsin(np.abs(np.sin(q[:count,4]))))
        bad_clear = [i for i in range(count) if not centerline_clear(kin,q[i],boxes,floor)]
        for i in range(1,count):
            samples = max(2,int(np.ceil(np.max(np.abs(q[i]-q[i-1]))/.02))+1)
            if any(not centerline_clear(kin,qi,boxes,floor)
                   for qi in np.linspace(q[i-1],q[i],samples)[1:-1]):
                bad_edge.append(i)
    pose_ok = (pe <= pt) & (oe <= rt)
    bounds_ok = (not count or bool(np.all(q[:count] >= kin.joint_lower-1e-8)
                                  and np.all(q[:count] <= kin.joint_upper+1e-8)))
    velocity_ok = bool(np.all(np.abs(v) <= speed+1e-6))
    acceleration_ok = bool(np.all(np.abs(acc) <= 250.+1e-5))
    all_frames = count == len(q)
    constraints_ok = bool(all_frames and pose_ok.all() and bounds_ok and velocity_ok and acceleration_ok
                          and len(records)==len(q)-1 and all(r.get('feasible',False) for r in records))
    accepted = len(records)==len(q)-1 and all(r.get('accepted',False) for r in records)
    if bad_clear or bad_edge:
        status = 'workcell_intersection'
    elif not constraints_ok:
        status = 'constraint_failure'
    elif not accepted:
        status = 'optimizer_failure'
    else:
        status = 'pass'
    rejected = [r['frame'] for r in records if not r.get('accepted',False)]
    infeasible = [r['frame'] for r in records if not r.get('feasible',False)]
    summary = dict(status=status, full_path_evaluated=all_frames, evaluated_poses=count,
        constraints_met=constraints_ok, optimizer_all_accepted=bool(accepted),
        first_unaccepted_frame=rejected[0] if rejected else -1,
        first_constraint_failure_frame=infeasible[0] if infeasible else -1,
        unaccepted_frames=rejected, constraint_failure_frames=infeasible,
        coarse_collision_frames=bad_clear, coarse_collision_edges=bad_edge,
        joint_limit_ok=bounds_ok, velocity_limit_ok=velocity_ok, acceleration_limit_ok=acceleration_ok,
        max_evaluated_position_error_mm=float(pe.max()*1000) if count else None,
        max_evaluated_orientation_error_deg=float(np.rad2deg(oe.max())) if count else None,
        max_evaluated_speed_rad_s=float(np.abs(v).max(initial=0)),
        max_evaluated_acceleration_rad_s2=float(np.abs(acc).max(initial=0)),
        max_evaluated_step_rad=float(np.abs(v).max(initial=0)*dt),
        min_evaluated_axis_separation_deg=float(axis.min()) if count else None,
        solver_status_counts=dict(Counter(str(r.get('status','exception')) for r in records)))
    return summary, errors


def solve_grid_point(job):
    index, xy, kin, data, layout = job
    started = time.perf_counter()
    delta = np.r_[xy-data['object_initial'][:2],0.]
    shifted = dict(data, p=data['p']+delta)
    boxes, floor = workcell_boxes(layout), layout['floor_z']
    q0, reset = select_reset(kin, shifted['p'][0],data['quat'][0],data['q'][0],boxes,floor)
    paths = np.full((len(MODES),len(data['p']),6),np.nan)
    errors = np.full((len(MODES),len(data['p']),2),np.nan)
    rows = []
    for mode_index, (mode, pt, rt) in enumerate(MODES):
        base = dict(index=index,x_m=float(xy[0]),y_m=float(xy[1]),mode=mode,**reset)
        if q0 is None:
            rows.append(dict(base, evaluated_poses=0, full_path_evaluated=False,
                             constraints_met=False, optimizer_all_accepted=False))
            continue
        q, records = sqp_path(kin, shifted, q0, stop_on_infeasible=True,
                              position_tolerance=pt, orientation_tolerance=rt, maxiter=150)
        stats, error = assess_path(kin,q,shifted['p'],data['quat'],records,data['dt'],
                                  data['speed'],boxes,floor,pt,rt)
        rows.append(dict(base, **stats, initial_q=q0.tolist(),
                         solver_s=sum(r.get('solve_s',0) for r in records),
                         solver_records=records))
        paths[mode_index], errors[mode_index] = q, error
    return index, rows, paths, errors, time.perf_counter()-started


def extent(points):
    return dict(x=[float(points[:,0].min()),float(points[:,0].max())],
                y=[float(points[:,1].min()),float(points[:,1].max())]) if len(points) else None


def plot_map(out, mode, rows, xy, layout, grid_info, source_center, spacing, commands):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    lower, upper = grid_info[:2]
    fig, ax = plt.subplots(figsize=(9,10),layout='constrained')
    ax.add_patch(Rectangle(lower,*(upper-lower),fill=False,lw=2,label='Tabletop: Z=0'))
    pedestal = layout['robot_base']
    ax.add_patch(Rectangle(np.array(pedestal['center'][:2])-np.array(pedestal['size'][:2])/2,
                           *pedestal['size'][:2],color='#dddddd',label='Robot pedestal'))
    status = np.array([r['status'] for r in rows])
    for key,label in LABELS.items():
        mask=status==key
        if mask.any():
            ax.scatter(*xy[mask].T,s=18,marker='s',color=COLORS['pass_color' if key=='pass' else key],
                       linewidths=0,label=f'{label}: {mask.sum()}')
    near = np.array([r['status']=='pass' and r['min_evaluated_axis_separation_deg']<5 for r in rows])
    if near.any():
        ax.scatter(*xy[near].T,s=23,facecolors='none',edgecolors='black',linewidths=.65,
                   label=f'Pass, but wrist axes approach <5 deg: {near.sum()}')
    ax.scatter(*layout['robot_mount']['position'][:2],marker='+',color='black',s=130)
    ax.text(-.17,.04,'RB3 base',fontsize=9)
    ax.scatter(*source_center[:2],marker='*',color='#1669bd',s=160,label='Recorded source can XY')
    ax.annotate('+X',xy=(.16,-.64),xytext=(-.02,-.64),arrowprops=dict(arrowstyle='->'),ha='center')
    budget='0.1 mm / 0.057 deg' if mode=='strict' else '5 mm / 2.865 deg'
    ax.set(xlabel='World X [m]',ylabel='World Y [m]',aspect='equal',
           xlim=(-.28,upper[0]+.06),ylim=(lower[1]-.04,upper[1]+.05),
           title=f'SQP whole-table screen — {mode}: {budget}\n'
                 f'{spacing*100:g} cm grid; fixed yaw; {commands} commands at 120 Hz\n'
                 'Offline recorded targets, NOT a policy/grasp-success map')
    ax.grid(alpha=.15)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.24),fontsize=8,framealpha=.95)
    fig.savefig(out/f'tabletop_sqp_{mode}.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--episode',type=int,default=11)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--spacing',type=float,default=.02)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1 or not np.isfinite(args.spacing) or args.spacing<=0:
        parser.error('Positive spacing and workers required')
    kin,data=load_episode(args.source,args.episode)
    layout_path=ROOT/'config/workcell/rb3_revo2_table.json'
    mesh_path=ROOT/'007_tuna_fish_can/textured_simple.obj'
    layout=json.loads(layout_path.read_text())
    np.testing.assert_allclose(kin.base_position,layout['robot_mount']['position'],atol=1e-9)
    np.testing.assert_allclose(kin.base_rotation,Rotation.from_quat(layout['robot_mount']['quaternion_xyzw']).as_matrix(),atol=1e-9)
    import trimesh
    mesh=trimesh.load(str(mesh_path),force='mesh',process=False)
    xy,*grid_info=table_grid(layout,mesh.vertices,data['object_initial'][3:7],args.spacing)
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'inputs.npz',**{k:v for k,v in data.items() if k!='meta'})
    frozen=dict(source=str(args.source.resolve()),episode=args.episode,
        checkpoint=data['meta']['checkpoint'],checkpoint_sha256=data['meta']['checkpoint_sha256'],
        dt=data['dt'],samples=len(data['p']),source_object_pose=data['object_initial'][:7].tolist(),
        table_bounds_xy=[x.tolist() for x in grid_info[:2]],
        initial_mesh_footprint_xy=[x.tolist() for x in grid_info[2:]],
        spacing_m=args.spacing,placements=len(xy),workers=args.workers,
        tolerances={name:[pt,rt] for name,pt,rt in MODES},native_speed=data['speed'].tolist(),
        acceleration_bound_rad_s2=250.,maxiter=150,ftol=1e-9,
        initialization='same nearest clear isolated analytic reset for both modes; no start branch re-search on SQP failure',
        stop='first infeasible SQP iterate retained; suffix untested/NaN; feasible nonconvergence continues only for diagnostic classification',
        limits='one recorded IK-input motion, fixed yaw, XY translation only; sampled lattice is not continuous-area proof',
        unsupported='actual policy / actuator / contact / grasp; full mesh/self collision; initial approach; current 10k deployment; torque saturation',
        physics_executed=False,defaults_changed=False,
        hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            kin.model_config_path,kin.source_usd_path,layout_path,mesh_path,
            args.source/'metadata.json',args.out/'inputs.npz',Path(__file__),
            ROOT/'tools/rb3_revo2_ik/sqp_pose_ik.py',ROOT/'tools/arm_diagnostics/compare_analytic_sqp.py')})
    (args.out/'experiment.json').write_text(json.dumps(frozen,indent=2)+'\n')
    # The source's initial_q, hand commands and metadata are stored once; workers
    # do not open the original 27 MB physics log or alter its files.
    worker_data={k:v for k,v in data.items() if k!='meta'}
    jobs=[(i,point,kin,worker_data,layout) for i,point in enumerate(xy)]
    all_q=np.full((len(xy),len(MODES),len(data['p']),6),np.nan)
    all_errors=np.full((len(xy),len(MODES),len(data['p']),2),np.nan)
    all_rows=[]
    started=time.perf_counter()
    print(f'Scan {len(xy)} XY placements, {len(data["p"])-1} commands, 2 modes, {args.workers} workers',flush=True)
    with (args.out/'progress.jsonl').open('x') as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,rows,q,error,elapsed in pool.map(solve_grid_point,jobs,chunksize=1):
                all_q[i],all_errors[i]=q,error
                all_rows.extend(rows)
                stream.write(json.dumps(dict(index=i,rows=rows,elapsed_s=elapsed))+'\n')
                stream.flush()
                if (i+1)%50==0 or i+1==len(xy):
                    counts={m:sum(r['status']=='pass' and r['mode']==m for r in all_rows) for m,_,_ in MODES}
                    print(f'{i+1}/{len(xy)}: {counts}, wall={time.perf_counter()-started:.1f}s',flush=True)
    np.savez_compressed(args.out/'paths.npz',xy=xy,q=all_q,errors_m_rad=all_errors,
                        evaluated=np.isfinite(all_q).all(axis=-1),modes=np.array([m for m,_,_ in MODES]),
                        command_t=data['command_t'],dt=data['dt'])
    compact=[{k:v for k,v in row.items() if k!='solver_records'} for row in all_rows]
    fields=sorted({k for row in compact for k in row})
    with (args.out/'placements.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(compact)
    summary=dict(placements=len(xy),elapsed_s=time.perf_counter()-started,by_mode={},
                 note='Failure is specific to these reset/solver budgets; bounds of passing points are not an entirely feasible rectangle')
    for mode_index,(mode,_,_) in enumerate(MODES):
        rows=[r for r in compact if r['mode']==mode]
        passed=np.array([r['status']=='pass' for r in rows])
        good=[r for r in rows if r['status']=='pass']
        summary['by_mode'][mode]=dict(counts=dict(Counter(r['status'] for r in rows)),
            pass_extent=extent(xy[passed]),
            pass_near_wrist_singularity_count=sum(r['min_evaluated_axis_separation_deg']<5 for r in good),
            pass_max_position_error_mm=max((r['max_evaluated_position_error_mm'] for r in good),default=None),
            pass_max_orientation_error_deg=max((r['max_evaluated_orientation_error_deg'] for r in good),default=None),
            pass_max_speed_rad_s=max((r['max_evaluated_speed_rad_s'] for r in good),default=None),
            pass_max_acceleration_rad_s2=max((r['max_evaluated_acceleration_rad_s2'] for r in good),default=None))
        plot_map(args.out,mode,rows,xy,layout,grid_info,data['object_initial'],args.spacing,len(data['p'])-1)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
