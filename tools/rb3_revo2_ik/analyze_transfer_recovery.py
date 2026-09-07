"""Summarize only completed recovery runs, separating replay from live policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.rb3_revo2_ik.analyze_mounted_interface import stats


def lag_samples(target,actual,dt):
    # Exploratory derivative correlation, not a correction to any error metric.
    x=np.diff(np.asarray(target),axis=0).reshape(len(target)-1,-1)
    y=np.diff(np.asarray(actual),axis=0).reshape(len(actual)-1,-1)
    if np.linalg.norm(x)<1e-8 or np.linalg.norm(y)<1e-8:return None
    scores=[]
    for lag in range(min(round(.2/dt),len(x)//3)+1):
        a=x[:len(x)-lag or None];b=y[lag:]
        scores.append(float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-20)))
    best=int(np.argmax(scores))
    return dict(delay_s=best*dt,correlation=scores[best],method='derivative correlation over [0,0.2]s; not time-shifting errors')


def summarize(path):
    meta=json.loads((path/'metadata.json').read_text())
    rows=[json.loads(l) for l in (path/'physics.jsonl').open()]
    # Existing verified hand link FK, not semantic keypoints. Include physical
    # follower tracking in the error by comparing ideal drive-target FK to runtime.
    from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics
    from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
    from scipy.spatial.transform import Rotation
    hand_fk=Revo2Kinematics()
    np.testing.assert_array_equal(hand_fk.joint_names,[meta['joint_names'][i] for i in meta['hand_ids']])
    tip_names=[f'right_{finger}_touch_link' for finger in ('thumb','index','middle','ring','pinky')]
    s0=meta['initial_states'][0];links=hand_fk._forward_links(np.asarray(s0['hand_q']))
    init_tips=Rotation.from_quat(s0['wrist_quat']).apply([links[n][1] for n in tip_names])+s0['wrist_pos']
    np.testing.assert_allclose(init_tips,s0['fingertips'],atol=2e-6,rtol=0)
    arm_fk=None
    if meta['arm_ids']:
        with Path(meta['state_bank']).open() as stream:bank=json.loads(next(stream))
        arm_fk=RB3730Kinematics(base_position=bank['base_position'],base_quaternion_xyzw=bank['base_quaternion_xyzw'])
    for row in rows:
        pos,quat=arm_fk.forward(row['q_cmd']) if arm_fk else (row['desired_wrist_pos'],row['desired_wrist_quat'])
        links=hand_fk._forward_links(np.asarray(row['hand_target']))
        tips=Rotation.from_quat(quat).apply([links[n][1] for n in tip_names])+pos
        row['_fingertip_command_error_m']=np.linalg.norm(tips-row['state']['fingertips'],axis=-1)
    result=dict(stage=meta['stage'],mode=meta['mode'],gains=meta.get('arm_gains_key','baseline'),
        response_tau=meta.get('response_tau_s',0),source=meta.get('actual_motion_source'),
        state_bank=meta['state_bank'],checkpoint_sha256=meta['checkpoint_sha256'],
        no_can=meta.get('recovery_no_can',False),slow=meta.get('recovery_speed',1),
        holds=meta.get('recovery_holds',False),episodes=[],effort_saturation='UNKNOWN')
    limits=np.asarray(meta['limits']['position']);vlimit=np.asarray(meta['limits']['velocity'])
    for end in meta['ends']:
        ep=end['episode'];s=[r for r in rows if r['episode']==ep];dt=meta['physics_dt']
        t=np.array([r['time_s'] for r in s]);q=np.array([r['state']['all_q'] for r in s]);v=np.array([r['state']['all_v'] for r in s])
        if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Nonfinite state')
        contact=np.array([r['can_robot_contact_N']>.01 for r in s]);ix=np.flatnonzero(contact)
        onset=float(t[ix[0]]) if len(ix) else None
        before=t<onset if onset is not None else np.ones(len(t),bool)
        rise=np.array([r['state']['object_state'][2]-meta['initial_states'][ep]['object_state'][2] for r in s])
        last=t>t[-1]-.2
        term=end['termination'];success=term['success']  # Existing task flag, unchanged.
        metrics={}
        for region,mask in [('all',np.ones(len(t),bool)),('pre_contact',before),('post_contact',~before)]:
            metrics[region]={k:stats([r[k] for i,r in enumerate(s) if mask[i]]) for k in
                ('wrist_position_error_m','wrist_rotation_error_rad','hand_error_rad','object_keypoint_error_m','_fingertip_command_error_m')}
            subset=[r for i,r in enumerate(s) if mask[i] and 'source_state' in r]
            if subset:
                metrics[region]['fingertip_source_error_m']=stats([np.linalg.norm(np.asarray(r['state']['fingertips'])-r['source_state']['fingertips'],axis=-1) for r in subset])
                metrics[region]['hand_source_error_rad']=stats([np.asarray(r['state']['hand_q'])-r['source_state']['hand_q'] for r in subset])
        violation=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0)
        entry=dict(episode=ep,success=bool(success),termination=term,contact_s=onset,
            simultaneous_failure_with_success=bool(success and any(value for key,value in term.items() if key not in ('success','demo_end_reached'))),
            lift_proxy=bool(rise[last].min()>=.1 and contact[last].mean()>=.8),
            final_lift_m=float(rise[-1]),ik_failures=sum(r.get('ik_failed',False) for r in s),
            slew_steps=sum(bool(r.get('command_rate_limited')) for r in s),
            max_position_violation_rad=float(violation.max()),
            joint_near_position_fraction=dict(zip(meta['joint_names'],(np.minimum(q-limits[:,0],limits[:,1]-q)<.01).mean(0).tolist())),
            joint_near_speed_fraction=dict(zip(meta['joint_names'],(abs(v)>=.95*vlimit).mean(0).tolist())),
            metrics=metrics)
        if meta['arm_ids']:
            ids=meta['arm_ids'];qc=np.array([r['q_cmd'] for r in s]);qi=np.array([r['q_ik'] for r in s]);qa=q[:,ids]
            step=np.diff(qc,axis=0);speed=step/dt;acc=np.diff(speed,axis=0)/dt
            entry.update(joint_error_rad=[stats(qa[:,i]-qc[:,i]) for i in range(6)],
                arm_joint_names=[meta['joint_names'][i] for i in ids],
                max_joint_step_rad=np.max(abs(step),axis=0).tolist(),
                max_path_speed_rad_s=np.max(abs(speed),axis=0).tolist(),
                max_path_accel_rad_s2=np.max(abs(acc),axis=0).tolist(),
                max_actual_speed_rad_s=np.max(abs(v[:,ids]),axis=0).tolist(),
                max_actual_accel_rad_s2=np.max(abs(np.diff(v[:,ids],axis=0)/dt),axis=0).tolist(),
                ik_jump_steps=np.flatnonzero(np.max(abs(np.diff(qi,axis=0)),axis=1)>.1).tolist(),
                arm_delay=lag_samples(qc,qa,dt),hand_delay=lag_samples([r['hand_target'] for r in s],[r['state']['hand_q'] for r in s],dt),
                max_ik_error_m=max(r['ik_position_error_m'] for r in s),
                max_command_tracking_error_m=max(r['tracking_position_m'] for r in s))
        if result['holds']:
            entry['holds']=[]
            for hold in range(5):
                h=[r for r in s if r.get('diagnostic_hold')==hold];endtime=h[-1]['time_s'];h=[r for r in h if r['time_s']>endtime-1]
                entry['holds'].append(dict(index=hold,position_m=stats([r['wrist_position_error_m'] for r in h]),
                                          rotation_rad=stats([r['wrist_rotation_error_rad'] for r in h])))
        result['episodes'].append(entry)
    result['successes']=sum(e['success'] for e in result['episodes'])
    result['lift_proxy_count']=sum(e['lift_proxy'] for e in result['episodes'])
    result['samples']=len(rows)
    return result


def main(root):
    output={}
    # First held-out attempt had PLAY XY randomization disabled: twenty repeats,
    # not twenty independent placements. Preserve raw logs but exclude inference.
    excluded={'live_new20_baseline','live_new20_candidate'}
    for path in sorted(root.iterdir()):
        if path.name not in excluded and path.is_dir() and (path/'metadata.json').is_file():
            if json.loads((path/'metadata.json').read_text()).get('realtime_view',False):
                continue  # Lightweight viewer has no per-physics diagnostics.
            output[path.name]=summarize(path)
    (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    for name,r in output.items():
        es=r['episodes'];p=np.mean([e['metrics']['all']['wrist_position_error_m']['mean'] for e in es])*1000
        print(name,r['stage'],f"{r['successes']}/{len(es)}",f'{p:.2f} mm','DIAGNOSTIC' if r['no_can'] else '')
    pairs={}
    for label,left,right,fixed in [
        ('hand_input','R0_full','R1_full',False),('arm_control','R1_full','R2_full',True),
        ('live_old20','live_old20_baseline','live_old20_candidate',False),
        ('live_new20','live_new20_v2_baseline','live_new20_v2_candidate',False)]:
        if left not in output or right not in output:continue
        a=json.loads((root/left/'metadata.json').read_text());b=json.loads((root/right/'metadata.json').read_text())
        if label=='live_new20':
            from tools.rb3_revo2_ik.recovery_replay import validate_unique_placements
            validate_unique_placements(a['initial_states'][:20])
            previous=json.loads((root/'live_old20_baseline'/'metadata.json').read_text())
            xy=np.asarray([s['object_state'][:2] for s in a['initial_states'][:20]])
            old_xy=np.asarray([s['object_state'][:2] for s in previous['initial_states'][:20]])
            if np.linalg.norm(xy[:,None]-old_xy[None,:],axis=-1).min()<1e-5:
                raise ValueError('Held-out placement overlaps selection bank')
        for key in ('checkpoint_sha256','reference','physics_dt','control_dt','limits','gravity','robot_spawn'):
            if a[key]!=b[key]:raise ValueError(f'{label}: changed invariant {key}')
        for ep in range(20):
            for key in ('all_q','all_v','robot_root','object_state','wrist_pos','wrist_quat','hand_q','hand_v','phase'):
                np.testing.assert_allclose(a['initial_states'][ep][key],b['initial_states'][ep][key],rtol=0,atol=2e-6,
                                           err_msg=f'{label} {ep} {key}')
        if fixed:
            ar=[json.loads(l) for l in (root/left/'physics.jsonl').open()]
            br=[json.loads(l) for l in (root/right/'physics.jsonl').open()]
            if len(ar)!=len(br):raise ValueError('Replay length changed')
            hand_ids=[i for i,n in enumerate(a['joint_names']) if n.startswith('right_')]
            for x,y in zip(ar,br):
                for key in ('time_s','source_time_s','desired_wrist_pos','desired_wrist_quat'):
                    np.testing.assert_array_equal(x[key],y[key])
                for key in ('position','velocity','feedforward'):
                    np.testing.assert_array_equal(np.asarray(x['controller'][key])[hand_ids],np.asarray(y['controller'][key])[hand_ids])
        old=output[left]['episodes'];new=output[right]['episodes']
        pairs[label]=dict(initial_states_verified=True,identical_replay_inputs_verified=fixed,
            old_success=sum(e['success'] for e in old),new_success=sum(e['success'] for e in new),
            success_to_failure=[x['episode'] for x,y in zip(old,new) if x['success'] and not y['success']],
            failure_to_success=[x['episode'] for x,y in zip(old,new) if not x['success'] and y['success']])
    (root/'paired_comparison.json').write_text(json.dumps(pairs,indent=2)+'\n')
    if all(name in output for name in ('R0_full','R1_full','R2_full','live_new20_v2_candidate')):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,2,figsize=(13,8))
        for name in ('R0_full','R1_full','R2_full'):
            es=output[name]['episodes'];x=[e['episode'] for e in es]
            ax[0,0].plot(x,[e['metrics']['all']['wrist_position_error_m']['mean']*1000 for e in es],'o-',label=name)
            ax[0,1].plot(x,[e['final_lift_m']*1000 for e in es],'o-',label=name)
        names=['live_old20_baseline','live_old20_candidate','live_new20_v2_baseline','live_new20_v2_candidate']
        ax[1,0].bar(['old/base','old/new','held/base','held/new'],[output[n]['successes'] for n in names])
        ax[1,0].set_ylim(0,21);ax[1,0].set_ylabel('Live-policy successes / 20')
        for label,name in [('Existing placements','live_old20_baseline'),('Held-out placements','live_new20_v2_baseline')]:
            m=json.loads((root/name/'metadata.json').read_text());xy=np.asarray([s['object_state'][:2] for s in m['initial_states'][:20]])
            ax[1,1].scatter(xy[:,0],xy[:,1],label=label)
        ax[0,0].set_ylabel('Replay wrist mean error [mm]');ax[0,1].set_ylabel('Replay final can lift [mm]')
        ax[0,0].set_xlabel('Placement');ax[0,1].set_xlabel('Placement')
        ax[1,1].set_xlabel('Initial can X [m]');ax[1,1].set_ylabel('Initial can Y [m]')
        for axis in ax.flat:axis.grid(alpha=.3)
        for axis in (ax[0,0],ax[0,1],ax[1,1]):axis.legend(fontsize=8)
        fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
