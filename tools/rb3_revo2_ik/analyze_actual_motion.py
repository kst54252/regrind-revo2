"""Compare actual-motion arm replay to its timestamp-matched floating source."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.rb3_revo2_ik.analyze_mounted_interface import stats


def analyze(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    if meta['stage']!='actual' or meta['policy_loaded']:raise ValueError('Not a policy-free actual-motion replay')
    source=Path(meta['actual_motion_source'])
    sm=json.loads((source/'metadata.json').read_text())
    rows=[json.loads(l) for l in (directory/'physics.jsonl').open()]
    sr=[json.loads(l) for l in (source/'physics.jsonl').open()]
    output=dict(source=str(source),replay=str(directory),closed_loop_policy_evaluation=False,
                effort_saturation='UNKNOWN',episodes=[])
    for end in meta['ends']:
        ep=end['episode'];a=[r for r in rows if r['episode']==ep];b=[r for r in sr if r['episode']==ep]
        if len(a)!=len(b):raise ValueError('Incomplete or extended motion')
        np.testing.assert_allclose([r['time_s'] for r in a],[r['time_s'] for r in b],atol=1e-10,rtol=0)
        for x,y in zip(a,b):
            np.testing.assert_allclose(x['desired_wrist_pos'],y['state']['wrist_pos'],atol=5e-8,rtol=0)
            np.testing.assert_allclose(x['desired_wrist_quat'],y['state']['wrist_quat'],atol=5e-8,rtol=0)
            np.testing.assert_allclose(x['recorded_hand_q'],y['state']['hand_q'],atol=0,rtol=0)
            if x['policy_called']:raise ValueError('Unexpected policy call')
            if not np.isfinite(np.r_[x['state']['all_q'],x['state']['all_v'],x['state']['wrist_pos'],x['state']['wrist_quat'],x['state']['object_state']]).all():
                raise ValueError('Nonfinite measured state')
        t=np.array([r['time_s'] for r in a]);contact=np.array([r['can_robot_contact_N']>.01 for r in a])
        onset=next((r['time_s'] for r in a if r['can_robot_contact_N']>.01),None)
        before=t<onset if onset is not None else np.ones(len(t),bool)
        z0=meta['initial_states'][ep]['object_state'][2]
        rise=np.array([r['state']['object_state'][2]-z0 for r in a]);last=t>t[-1]-.2
        q=np.array([r['state']['all_q'] for r in a]);limits=np.array(meta['limits']['position'])
        violations=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0)
        metrics={}
        for label,mask in [('all',np.ones(len(t),bool)),('before_contact',before),('after_contact',~before)]:
            metrics[label]={k:stats([r[k] for i,r in enumerate(a) if mask[i]]) for k in
                ('wrist_position_error_m','wrist_rotation_error_rad','recorded_hand_error_rad',
                 'recorded_follower_error_rad','object_keypoint_error_m')}
        entry=dict(episode=ep,duration_s=float(t[-1]),termination=end['termination'],
            first_termination_time_s=end['first_termination_time_s'],
            success_without_earlier_failure=bool(end['termination']['success']) and not any(
                value for key,value in end['termination'].items() if key not in ('success','demo_end_reached')),
            lift_contact_proxy=bool(rise[last].min()>=.1 and contact[last].mean()>=.8),
            final_lift_m=float(rise[-1]),source_final_lift_m=b[-1]['state']['object_state'][2]-sm['initial_states'][ep]['object_state'][2],
            contact_onset_s=onset,source_contact_onset_s=next((r['time_s'] for r in b if r['can_robot_contact_N']>.01),None),
            object_actual_path_error_m=stats(np.linalg.norm(np.array([r['state']['object_state'][:3] for r in a])-np.array([r['state']['object_state'][:3] for r in b]),axis=1)),
            ik_failed_steps=sum(r['ik_failed'] for r in a),
            max_ik_position_error_m=max(r['ik_position_error_m'] for r in a),
            arm_position_violation_steps=int((violations[:,meta['arm_ids']]>1e-6).any(axis=1).sum()),
            max_joint_limit_excursion_rad=float(violations.max()),
            velocity_limit_steps=sum(r['velocity_limit_violation'] for r in a),
            recorded_hand_clipped_steps=sum(r['recorded_hand_clipped'] for r in a),
            rate_limited_steps=sum(r['command_rate_limited'] for r in a),metrics=metrics)
        output['episodes'].append(entry)
    output['completed_episodes']=len(output['episodes'])
    output['successes']=sum(e['success_without_earlier_failure'] for e in output['episodes'])
    output['lift_contact_count']=sum(e['lift_contact_proxy'] for e in output['episodes'])
    output['failed_episodes']=[e['episode'] for e in output['episodes'] if not e['success_without_earlier_failure']]
    failure_time={e['episode']:e['first_termination_time_s'] for e in output['episodes']}
    output['pooled_metrics']={}
    for label,selected in [('full_recording',rows),('through_first_termination',[
            r for r in rows if r['time_s']<=failure_time[r['episode']]])]:
        output['pooled_metrics'][label]={k:stats([r[k] for r in selected]) for k in
            ('wrist_position_error_m','wrist_rotation_error_rad','recorded_hand_error_rad')}
    output['max_target_leader_clamp_rad']=max(float(np.max(abs(
        np.asarray(r['hand_target'])-r['recorded_hand_q']))) for r in rows)
    output['max_error_events']={}
    for key in ('wrist_position_error_m','wrist_rotation_error_rad'):
        row=max(rows,key=lambda r:r[key])
        output['max_error_events'][key]={k:row[k] for k in ('episode','time_s',key)}
    (directory/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ep=4 if len(output['episodes'])>4 else 0
    for label,collection in [('Floating measured source',sr),('Mounted actual-motion replay',rows)]:
        a=[r for r in collection if r['episode']==ep];t=[r['time_s'] for r in a]
        for axis,key in zip(ax.flat[:2],['wrist_pos','object_state']):
            axis.plot(t,[r['state'][key][2]*1000 for r in a],label=label)
        ax[1,0].plot(t,[r['can_robot_contact_N'] for r in a],label=label)
    entries=output['episodes'];x=[e['episode'] for e in entries]
    ax[1,1].plot(x,[e['source_final_lift_m']*1000 for e in entries],'o-',label='Floating source')
    ax[1,1].plot(x,[e['final_lift_m']*1000 for e in entries],'o-',label='Arm replay')
    for axis,title in zip(ax.flat,[f'Placement {ep}: actual wrist Z [mm]',f'Placement {ep}: can Z [mm]',
                                  'Peak body can contact [N]','Final can lift [mm]']):
        axis.set_ylabel(title);axis.grid(alpha=.3);axis.legend(fontsize=8)
    for axis in [ax[0,0],ax[0,1],ax[1,0]]:axis.set_xlabel('Recorded time [s], no shift')
    ax[1,1].set_xlabel('Placement');fig.tight_layout();fig.savefig(directory/'comparison.png',dpi=150)
    plt.close(fig)
    print(json.dumps({k:output[k] for k in ('completed_episodes','successes','lift_contact_count','failed_episodes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    analyze(p.parse_args().directory)
