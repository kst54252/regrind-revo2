"""Three live-policy outcomes and contact-split metrics; no success-rule edits."""
import argparse
import json
from pathlib import Path
import numpy as np


def stats(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return None
    return dict(mean=float(np.mean(abs(x))),p95=float(np.percentile(abs(x),95)),max=float(np.max(abs(x))))


def analyze(root, simple_directory=None, output_directory=None):
    destination=output_directory or root
    destination.mkdir(parents=True,exist_ok=True)
    result={'contact_threshold_N':.01,'note':'Contact onset = first sampled filtered can-robot normal force >0.01 N. Task success is unchanged reference-end success, not a contact-sensor definition.', 'modes':{}}
    metas={}; traces={}; sources={}
    for mode in ('floating','legacy','simple'):
        directory=simple_directory if mode=='simple' and simple_directory else root/(mode+'20')
        sources[mode]=str(directory)
        meta=json.loads((directory/'metadata.json').read_text());metas[mode]=meta
        rows=[json.loads(l) for l in (directory/'physics.jsonl').open()]
        traces[mode]=rows
        if len(meta['ends'])!=20:raise ValueError('Incomplete evaluation '+mode)
        summaries=[]
        for end in meta['ends']:
            ep=end['episode'];s=[r for r in rows if r['episode']==ep]
            t=np.array([r['time_s'] for r in s]);contact=np.array([r['can_robot_contact_N']>.01 for r in s])
            ix=np.flatnonzero(contact);onset=float(t[ix[0]]) if len(ix) else None
            initial=meta['initial_states'][ep];initialz=initial['object_state'][2]
            rise=np.array([r['state']['object_state'][2]-initialz for r in s])
            last=t>t[-1]-.2
            q=np.array([r['state']['all_q'] for r in s]);v=np.array([r['state']['all_v'] for r in s])
            limits=np.array(meta['limits']['position']);vlimits=np.array(meta['limits']['velocity'])
            violation=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0)
            part=dict(episode=ep,initial_object_xyz=initial['object_state'][:3],termination=end['termination'],
                task_success=bool(end['termination']['success']),duration_s=float(t[-1]),
                contact_onset_s=onset,peak_contact_N=max(r['can_robot_contact_N'] for r in s),
                max_lift_m=float(rise.max()),final_lift_m=float(rise[-1]),last_02s_min_lift_m=float(rise[last].min()),
                last_02s_contact_fraction=float(contact[last].mean()),
                lift_contact_proxy=bool(rise[last].min()>=.1 and contact[last].mean()>=.8),
                ik_failed_steps=sum(r.get('ik_failed',False) for r in s),
                command_rate_limited_steps=sum(bool(r.get('command_rate_limited')) for r in s),
                position_limit_steps=sum(r['position_limit_violation'] for r in s),
                velocity_limit_steps=sum(r['velocity_limit_violation'] for r in s),effort_saturation='UNKNOWN',metrics={})
            part['joint_max_position_violation_rad']=dict(zip(meta['joint_names'],violation.max(0).tolist()))
            part['joint_speed_near_limit_fraction']=dict(zip(meta['joint_names'],(abs(v)>=.95*vlimits).mean(0).tolist()))
            part['arm_position_violation_steps']=int(np.any(violation[:,meta['arm_ids']]>1e-6,axis=1).sum()) if meta['arm_ids'] else 0
            part['hand_error_mean_by_joint_rad']=np.abs([r['hand_error_rad'] for r in s]).mean(0).tolist()
            part['hand_error_max_by_joint_rad']=np.abs([r['hand_error_rad'] for r in s]).max(0).tolist()
            masks={'all':np.ones(len(s),bool),'before_contact':t<onset if onset is not None else np.ones(len(s),bool),
                   'after_contact':t>=onset if onset is not None else np.zeros(len(s),bool)}
            for region,mask in masks.items():
                part['metrics'][region]={k:stats([r[k] for i,r in enumerate(s) if mask[i]]) for k in
                    ('wrist_position_error_m','wrist_rotation_error_rad','hand_error_rad','object_keypoint_error_m')}
            summaries.append(part)
        result['modes'][mode]=dict(task_successes=sum(r['task_success'] for r in summaries),
            lift_contact_proxy_count=sum(r['lift_contact_proxy'] for r in summaries),episodes=summaries,
            frozen=meta['frozen_policy_verified'],observation_parity_checks=meta['observation_action_parity_checks'])
    for other in ('floating','simple'):
        for key in ('checkpoint_sha256','reference','state_bank','physics_dt','control_dt'):
            if metas['legacy'][key]!=metas[other][key]:raise ValueError('Mismatch '+key)
        for i in range(20):
            a,b=metas['legacy']['initial_states'][i],metas[other]['initial_states'][i]
            keys=['wrist_pos','wrist_quat','wrist_velocity','hand_q','hand_v','follower_q','follower_v','object_state','phase']
            if other=='simple':keys+=['all_q','all_v','robot_root']
            for key in keys:np.testing.assert_allclose(a[key],b[key],atol=2e-6,rtol=0,err_msg=f'{other} initial {i} {key}')
    for key in ('gains','limits','robot_spawn'):
        if metas['legacy'][key]!=metas['simple'][key]:raise ValueError('Changed arm physics '+key)
    result['initial_states_verified']=True
    old=result['modes']['legacy']['episodes'];new=result['modes']['simple']['episodes']
    result['old_success_new_failure']=[a['episode'] for a,b in zip(old,new) if a['task_success'] and not b['task_success']]
    result['old_failure_new_success']=[a['episode'] for a,b in zip(old,new) if not a['task_success'] and b['task_success']]
    result['proxy_old_success_new_failure']=[a['episode'] for a,b in zip(old,new) if a['lift_contact_proxy'] and not b['lift_contact_proxy']]
    result['proxy_old_failure_new_success']=[a['episode'] for a,b in zip(old,new) if not a['lift_contact_proxy'] and b['lift_contact_proxy']]
    result['simple_response_tau_s']=metas['simple'].get('response_tau_s',0.)
    result['source_directories']=sources
    (destination/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for mode,r in result['modes'].items():
        label=f"simple tau={result['simple_response_tau_s']:g}s" if mode=='simple' else mode
        ep=r['episodes'];x=[e['episode'] for e in ep]
        for a,y,title in zip(ax.flat,[[e['final_lift_m']*1000 for e in ep],
            [e['metrics']['all']['wrist_position_error_m']['mean']*1000 for e in ep],
            [e['metrics']['all']['hand_error_rad']['mean'] for e in ep],
            [e['metrics']['all']['object_keypoint_error_m']['mean']*1000 for e in ep]],
            ['Final can lift [mm]','Mean desired-actual wrist error [mm]','Mean finger error [rad]','Mean object keypoint error [mm]']):
            a.plot(x,y,'o-',label=label);a.set_ylabel(title);a.set_xlabel('Placement');a.grid(alpha=.3)
    ax[0,0].legend();fig.tight_layout();fig.savefig(destination/'comparison.png',dpi=140);plt.close(fig)
    # Placement 4 was selected from the original failure, not after the trial.
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for mode,rows in traces.items():
        label=f"simple tau={result['simple_response_tau_s']:g}s" if mode=='simple' else mode
        s=[r for r in rows if r['episode']==4];t=[r['time_s'] for r in s]
        series=[[r['state']['wrist_pos'][2]*1000 for r in s],
                [r['state']['object_state'][2]*1000 for r in s],
                [r['can_robot_contact_N'] for r in s],
                [np.linalg.norm(r['hand_error_rad']) for r in s]]
        for a,y,title in zip(ax.flat,series,['Actual wrist Z [mm]','Can Z [mm]',
                                           'Peak body can-contact [N]','Finger error norm [rad]']):
            a.plot(t,y,label=label);a.set_xlabel('Time [s]');a.set_ylabel(title);a.grid(alpha=.3)
    ax[0,0].legend();fig.suptitle('Placement 4: same initial state, actual closed-loop policy')
    fig.tight_layout();fig.savefig(destination/'placement4.png',dpi=140);plt.close(fig)
    print(json.dumps({m:{k:v[k] for k in ('task_successes','lift_contact_proxy_count')} for m,v in result['modes'].items()},indent=2))
    print('old success -> new failure',result['old_success_new_failure'])
    print('old failure -> new success',result['old_failure_new_success'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--simple-directory',type=Path)
    p.add_argument('--output-directory',type=Path)
    args=p.parse_args();analyze(args.root,args.simple_directory,args.output_directory)
