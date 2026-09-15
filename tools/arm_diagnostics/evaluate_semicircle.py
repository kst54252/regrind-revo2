"""Frozen semicircle placement banks and actual closed-loop grasp comparisons.

Preparation uses existing analytic IK only to select yaw/reset branch. Runtime
continues using the maintained policy -> existing IK -> actuators evaluator.
Screened/unreachable placements stay in the denominator; no grasp-result search.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation
from tools.arm_diagnostics.search_tabletop_yaw import rotate_task, dense_reference
from tools.rb3_revo2_ik.analytic_branch_ik import AnalyticBranchIK
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.reference_trajectory import load_reference_trajectory
from tools.rb3_revo2_ik.sequence_branch_ik import centerline_clear, workcell_boxes

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5'
CHECKPOINT=ROOT/'logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt'


def semicircle_grid(radius=.75):
    # Freeze sampling independently of IK and grasp outcomes. Table footprint
    # margin 5 cm, world anchored 10 x 15 cm grid, base XY=(0,0).
    if not np.isfinite(radius) or radius<=.30:
        raise ValueError('Finite radius > .30 m required for the fixed table region')
    return np.array([(x,y) for y in np.arange(-.60,.601,.15)
                     for x in np.arange(.30,.751,.10) if x*x+y*y <= radius*radius+1e-10])


def heldout_points(radius):
    if not np.isfinite(radius) or radius<=.30:
        raise ValueError('Finite radius > .30 m required')
    rng=np.random.default_rng(20260915);points=[]
    while len(points)<20:
        p=rng.uniform([.30,-.70],[.75,.70])
        if np.dot(p,p)<=radius*radius:points.append(p)
    return np.array(points)


def continuous_branches(analytic, p, quat, seed):
    paths={}
    for i,(pi,qi) in enumerate(zip(p,quat)):
        result=analytic.inverse_all(pi,qi)
        if not result.exhaustive_isolated:return []
        groups={tuple(b) for b in result.branch_ids}
        if i==0:paths={key:[] for key in groups}
        for key in list(paths):
            if key not in groups:
                del paths[key];continue
            options=result.q[np.all(result.branch_ids==key,axis=1)]
            previous=paths[key][-1] if paths[key] else seed
            chosen=options[np.argmin(np.sum((options-previous)**2,axis=1))]
            paths[key].append(chosen)
    return [np.array(path) for path in paths.values()]


def select(job):
    index,xy,ref_path,layout,clearance=job
    ref=load_reference_trajectory(ref_path)
    kin=RB3730Kinematics(base_position=layout['robot_mount']['position'],
                        base_quaternion_xyzw=layout['robot_mount']['quaternion_xyzw'])
    analytic=AnalyticBranchIK(kin);boxes=workcell_boxes(layout)
    clear=lambda q:centerline_clear(kin,q,boxes,layout['floor_z'])
    center=ref.object_pos[0];destination=np.r_[xy,center[2]]
    pp,qq=rotate_task(ref.wrist_pos,ref.wrist_quat_xyzw,center,destination,0)
    baseline=kin.inverse(pp[0],qq[0],initial_q=ref.rb3_joints[0],neutral_q=ref.rb3_joints[0])
    fixed=None
    if baseline.success and clear(baseline.q):
        fixed=dict(yaw_deg=0.,reset_q=baseline.q.tolist(),selection='unchanged numerical reset')
    # Optional geometric guard, not a changed collision setting or a mesh
    # certificate. Inflating workcell boxes excludes elbow-down branches whose
    # link centerline barely clears the tabletop/pedestal.
    guarded_boxes=[(lower-clearance,upper+clearance) for lower,upper in boxes]
    clear=lambda q:centerline_clear(kin,q,guarded_boxes,layout['floor_z']+clearance)
    # Rotationally covariant search: radial approach plus twelve offsets.
    radial=np.degrees(np.arctan2(xy[1],xy[0]));candidates=[]
    angles=[radial+a for a in range(-180,180,30)]
    for yaw in angles:
        p,q=rotate_task(ref.wrist_pos,ref.wrist_quat_xyzw,center,destination,yaw)
        seed=ref.rb3_joints[0].copy();seed[0]+=np.radians(yaw)
        for path in continuous_branches(analytic,p,q,seed):
            if not all(clear(qi) for qi in path):continue
            speed=float(np.abs(np.diff(path,axis=0)).max()/ref.dt)
            axis=float(np.degrees(np.arcsin(np.abs(np.sin(path[:,4])))).min())
            # Fixed, outcome-independent criterion; axis margin penalizes
            # singular approaches, not a claim that 15 degrees guarantees safety.
            score=speed+.5*max(0.,15.-axis)
            candidates.append((score,yaw,path,speed,axis))
    chosen=None
    t,dp,dq=dense_reference(ref)
    for _,yaw,path,_,_ in sorted(candidates,key=lambda c:c[0])[:3]:
        p,q=rotate_task(dp,dq,center,destination,yaw)
        dense=continuous_branches(analytic,p,q,path[0])
        if not dense:continue
        path=min(dense,key=lambda a:np.linalg.norm(a[0]-path[0]))
        if not all(clear(qi) for qi in path):continue
        speed=float(np.abs(np.diff(path,axis=0)).max()/(ref.dt/4))
        axis=float(np.degrees(np.arcsin(np.abs(np.sin(path[:,4])))).min())
        record=dict(yaw_deg=float(yaw),reset_q=path[0].tolist(),max_reference_speed=speed,
                    min_wrist_axis_deg=axis,score=speed+.5*max(0.,15.-axis),
                    selection='full-reference analytic branch and yaw; 120 Hz checked; coarse collision only')
        if chosen is None or record['score']<chosen['score']:chosen=record
    return dict(placement_id=index,xy=xy.tolist(),fixed=fixed,selected=chosen,
                unscreened_grasp=True,candidate_paths=len(candidates))


def prepare(args):
    args.out.mkdir(parents=True,exist_ok=False)
    layout=json.loads((ROOT/'config/workcell/rb3_revo2_table.json').read_text())
    ref=load_reference_trajectory(REFERENCE)
    xy=heldout_points(args.radius) if args.heldout else semicircle_grid(args.radius)
    definition=dict(radius_m=args.radius,xy=xy.tolist(),reference=str(REFERENCE),
                    checkpoint=str(CHECKPOINT),reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    selection_rule='12 radial-relative yaws; minimize max reference speed + .5*max(0,15-min wrist-axis degrees)',
                    placement_count=len(xy),heldout=args.heldout,arm_clearance_m=args.arm_clearance,
                    policy_hz=30,physics_hz=120,workcell=layout,
                    success='existing terminations AND final .2s lift >=.10m AND filtered can-robot normal contact >.01N for >=80% samples',
                    scope='discrete grid, no continuous-area or real-robot safety guarantee')
    (args.out/'definition.json').write_text(json.dumps(definition,indent=2)+'\n')
    started=time.monotonic();rows=[]
    jobs=[(i,p,REFERENCE,layout,args.arm_clearance) for i,p in enumerate(xy)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(select,jobs):
            rows.append(r)
            print('placement',r['placement_id'],r['xy'],'fixed',r['fixed'] is not None,
                  'selected',None if r['selected'] is None else round(r['selected']['yaw_deg'],1),flush=True)
    (args.out/'selection.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl').open() as f:
        old_header=json.loads(next(f));template=next(json.loads(line) for line in f if '"initialization"' in line)
    kin=RB3730Kinematics(base_position=layout['robot_mount']['position'],base_quaternion_xyzw=layout['robot_mount']['quaternion_xyzw'])
    manifest=[]
    for mode in ('fixed','selected'):
        states=[]
        for r in rows:
            option=r[mode]
            if option is None:continue
            state=copy.deepcopy(template);center=ref.object_pos[0];dest=np.r_[r['xy'],center[2]]
            op,oq=rotate_task(ref.object_pos,ref.object_quat_xyzw,center,dest,option['yaw_deg'])
            q=np.array(option['reset_q']);wp,wq=kin.forward(q)
            state.update(placement_id=r['placement_id'],task_yaw_deg=option['yaw_deg'],
                         reset_arm_q=q.tolist(),actual_base_pos=wp.tolist(),actual_base_quat_xyzw=wq.tolist(),
                         placement_offset=(dest-center).tolist(),object_root_state=np.r_[op[0],oq[0],np.zeros(6)].tolist(),
                         reference_frame=0,applied_arm_target=q.tolist())
            for name,value in zip(kin.joint_names,q):
                state['all_joint_pos'][old_header['user_joint_names'].index(name)]=float(value)
            states.append(state)
        for batch,start in enumerate(range(0,len(states),20)):
            block=copy.deepcopy(states[start:start+20]);n=len(block)
            block.append(copy.deepcopy(block[0])) # extra reset state, never scored
            for ep,state in enumerate(block):state['episode']=ep
            header=dict(event='metadata',bank_kind='semicircle_task_placements',seed=42,reference=str(REFERENCE),
                        joint_names=kin.joint_names,user_joint_names=old_header['user_joint_names'],
                        base_position=layout['robot_mount']['position'],base_quaternion_xyzw=layout['robot_mount']['quaternion_xyzw'],
                        definition=str((args.out/'definition.json').resolve()))
            path=args.out/f'{mode}_states_{batch}.jsonl'
            with path.open('x') as f:
                for record in [header]+block:f.write(json.dumps(record)+'\n')
            manifest.append(dict(mode=mode,batch=batch,episodes=n,states=str(path.resolve()),
                                 placements=[s['placement_id'] for s in block[:-1]]))
    (args.out/'batches.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared',len(rows),'placements;',len(manifest),'batches;',time.monotonic()-started,'seconds',flush=True)


def run(args):
    batches=json.loads((args.out/'batches.json').read_text())
    mode='fixed' if args.method=='fixed' else 'selected'
    for b in batches:
        if b['mode']!=mode or (args.batch is not None and b['batch']!=args.batch):continue
        output=args.out/f'{args.method}_{b["batch"]}{args.tag}'
        cmd=['bash',str(ROOT/'scripts/evaluate_mounted_interface.sh'),'--mode','simple','--checkpoint',str(CHECKPOINT),
             '--states',b['states'],'--episodes',str(b['episodes'] if args.episodes is None else args.episodes),
             '--task-placement-bank','--output',str(output),'--headless']
        if args.method=='bounded':
            cmd+=['--transfer-config',str(ROOT/'config/experiments/rb3_smooth_bounded_ik.json'),'--fast-ik',
                  '--fingertip-contact-config',str(ROOT/'config/experiments/revo2_rubber_contact.json')]
        else:cmd+=['--arm-controller','video']
        print('Executing:', ' '.join(cmd),flush=True)
        with (args.out/f'{args.method}_{b["batch"]}{args.tag}.log').open('x') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        if not (output/'metadata.json').exists():
            raise RuntimeError(f'Isaac process returned without a completed report: {output}; inspect its log')


def analyze(args):
    from tools.arm_diagnostics.analyze_transfer_recovery import summarize
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    definition=json.loads((args.out/'definition.json').read_text())
    selection=json.loads((args.out/'selection.json').read_text())
    batches=json.loads((args.out/'batches.json').read_text())
    results={};runtime_baseline=None;initials={};rows=[];matched_initials=0
    methods=['fixed','selected','bounded'] + (['clearance'] if args.guarded else [])
    for method in methods:
        source=args.guarded if method=='clearance' else args.out
        run_method='selected' if method=='clearance' else method
        source_batches=json.loads((source/'batches.json').read_text())
        source_selection=json.loads((source/'selection.json').read_text())
        np.testing.assert_allclose([r['xy'] for r in selection],[r['xy'] for r in source_selection],rtol=0,atol=0)
        values={};bmode='fixed' if method=='fixed' else 'selected'
        expected=[b for b in source_batches if b['mode']==bmode]
        for b in expected:
            path=source/f'{run_method}_{b["batch"]}{args.tag}'
            if not (path/'metadata.json').exists():continue
            meta=json.loads((path/'metadata.json').read_text())
            assert meta['frozen_policy_verified'] and meta['policy_loaded']
            assert meta['checkpoint_sha256']==definition['checkpoint_sha256']
            profile={k:meta[k] for k in ('physics_dt','control_dt','gains','limits','gravity','robot_spawn','fingertip_contact')}
            if runtime_baseline is None:runtime_baseline=profile
            assert profile==runtime_baseline, 'Uncontrolled physical/configuration difference'
            summary=summarize(path)
            (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            for e in summary['episodes']:
                s=meta['initial_states'][e['episode']];i=s['placement_id']
                if method=='bounded':
                    previous=initials.get(i)
                    if previous is None:raise ValueError('Analyze selected before bounded')
                    for key in ('all_q','all_v','wrist_pos','wrist_quat','object_state','phase'):
                        np.testing.assert_allclose(s[key],previous[key],atol=2e-6,rtol=0)
                    matched_initials+=1
                if method=='selected':initials[i]=s
                e.update(placement_id=i,yaw_deg=s['task_yaw_deg'],
                         grasp_proxy=bool(e['success'] and not e['simultaneous_failure_with_success'] and e['lift_proxy']))
                values[i]=e
        results[method]=dict(evaluated=len(values),total=len(selection),
                             task_success=sum(e['success'] for e in values.values()),
                             lift_contact_success=sum(e['grasp_proxy'] for e in values.values()),
                             episodes=values)
        for r in source_selection:
            e=values.get(r['placement_id'])
            row=dict(method=method,placement_id=r['placement_id'],x=r['xy'][0],y=r['xy'][1],
                     state='not_evaluated' if e is None else 'grasp' if e['grasp_proxy'] else 'failed',
                     yaw_deg=None if r[bmode] is None else r[bmode]['yaw_deg'])
            if e is None and r[bmode] is None:row['state']='initialization_screen_failed'
            if e is not None:
                row.update(task_success=e['success'],lift_contact_success=e['grasp_proxy'],
                           termination=json.dumps(e['termination']),final_lift_m=e['final_lift_m'],
                           ik_failures=e['ik_failures'],
                           wrist_p95_m=e['metrics']['all']['wrist_position_error_m']['p95'],
                           max_actual_accel=max(e['max_actual_accel_rad_s2']))
            rows.append(row)
    pairs={}
    for method in methods[1:]:
        a=results['fixed']['episodes'];b=results[method]['episodes']
        common=set(a)&set(b)
        pairs[method]=dict(common_evaluated=len(common),
            failure_to_success=sorted(i for i in common if not a[i]['grasp_proxy'] and b[i]['grasp_proxy']),
            success_to_failure=sorted(i for i in common if a[i]['grasp_proxy'] and not b[i]['grasp_proxy']),
            new_screened_initialization_success=sorted(i for i in b if i not in a and b[i]['grasp_proxy']))
    report=dict(definition=definition,methods=results,paired=pairs,runtime_equal_verified=True,
                selected_bounded_actual_initial_states_matched=matched_initials,
                effort_saturation='UNKNOWN; implicit solver drive-only torque unavailable')
    (args.out/f'comparison{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.out/f'comparison{args.tag}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,len(methods),figsize=(4.2*len(methods),6),sharex=True,sharey=True)
    colors=dict(grasp='#23935f',failed='#e57437',initialization_screen_failed='#555b63',not_evaluated='#c9cdd1')
    names=['Fixed yaw / original controller','Selected yaw + initial branch','Selected + bounded IK (250 rad/s²)',
           'Selected + 55 mm workcell guard']
    theta=np.linspace(-np.pi/2,np.pi/2,250);radius=definition['radius_m']
    for ax,method,title in zip(axes,results,names):
        ax.add_patch(Rectangle((.25,-.8),.8,1.6,fc='#f2f4f7',ec='#999'))
        ax.plot(radius*np.cos(theta),radius*np.sin(theta),'k--',lw=1)
        ax.plot(0,0,'ks',ms=7)
        for r in [v for v in rows if v['method']==method]:
            ax.scatter(r['x'],r['y'],s=55,c=colors[r['state']],zorder=4)
            if r['yaw_deg'] is not None:
                a=np.radians(r['yaw_deg']);ax.arrow(r['x'],r['y'],.035*np.cos(a),.035*np.sin(a),
                    width=.0015,head_width=.014,color='#333',zorder=5,length_includes_head=True)
        n=results[method]['lift_contact_success'];ax.set_title(f'{title}\nLift/contact proxy: {n}/{len(selection)}')
        ax.set_aspect('equal');ax.set_xlim(-.04,.82);ax.set_ylim(-.78,.78)
        ax.set_xlabel('World X [m]');ax.grid(alpha=.2)
    axes[0].set_ylabel('World Y [m]')
    for name,color in colors.items():axes[-1].scatter([],[],c=color,label=name.replace('_',' '))
    axes[-1].legend(loc='lower right',fontsize=7)
    fig.suptitle(f'Actual closed-loop REGRIND grasps — R={radius:g} m, {len(selection)} fixed placements\nGreen requires task success + final 0.2 s lift/contact; not an IK-only map',fontsize=11)
    fig.tight_layout();fig.savefig(args.out/f'semicircle_grasp_comparison{args.tag}.png',dpi=190);plt.close(fig)
    print(json.dumps({k:{n:v[n] for n in ('evaluated','total','task_success','lift_contact_success')} for k,v in results.items()},indent=2))
    print(json.dumps(pairs,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run','analyze'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--radius',type=float,default=.75)
    p.add_argument('--heldout',action='store_true',help='Separate 20-point fixed random bank; never select based on grasp outcomes')
    p.add_argument('--arm-clearance',type=float,default=0.,
                   help='Opt-in workcell AABB inflation [m] for branch selection only, not simulator collision geometry')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--method',choices=('fixed','selected','bounded'),default='fixed')
    p.add_argument('--batch',type=int)
    p.add_argument('--episodes',type=int,help='Pilot only; never label a prefix as a complete grid')
    p.add_argument('--tag',default='',help='New-run suffix to preserve all previous results')
    p.add_argument('--guarded',type=Path,help='Analyze: additional geometry-guarded selection results at identical XY')
    args=p.parse_args()
    if not np.isfinite(args.arm_clearance) or not 0<=args.arm_clearance<=.1:p.error('arm clearance must be in [0,.1] m')
    {'prepare':prepare,'run':run,'analyze':analyze}[args.command](args)


if __name__=='__main__':main()
