"""Offline 120 Hz comparison; reuse existing FK, scoring and effort provenance.

python -m tools.rb3_revo2_ik.analyze_ik120 ROOT RUN [RUN ...]
Pair adjacent runs with --paired. Output stays under ROOT; no simulator calls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tools.rb3_revo2_ik.analyze_mounted_interface import stats
from tools.rb3_revo2_ik.analyze_transfer_recovery import summarize
from tools.rb3_revo2_ik.analyze_arm_execution import pose_errors


def differences(commands, actual_velocity, initial_command, initial_velocity, dt):
    """Backward differences at post-step timestamps, including first reset sample.

    Bounded RB3 coordinates are NOT wrapped. No cross-episode differentiation.
    Path derivative and native velocity targets are distinct measured columns.
    """
    path=np.diff(np.vstack([initial_command,commands]),axis=0)/dt
    acceleration=np.diff(np.vstack([initial_velocity,actual_velocity]),axis=0)/dt
    path_acc=np.diff(np.vstack([np.zeros_like(initial_command),path]),axis=0)/dt
    return path,path_acc,acceleration


def paired_initials(a,b):
    for key in ('checkpoint_sha256','reference','state_bank','physics_dt','control_dt','limits',
                'gravity','robot_spawn','response_tau_s','arm_velocity_path','arm_response_physics','ik_policy_rate'):
        if a[key]!=b[key]:raise ValueError('Changed comparison invariant: '+key)
    if len(a['ends'])!=len(b['ends']):raise ValueError('Different episode counts')
    keep=[i for i in range(len(a['joint_names'])) if i!=a['arm_ids'][5]]
    for key in ('kp','kd'):
        np.testing.assert_array_equal(np.asarray(a['gains'][key])[keep],np.asarray(b['gains'][key])[keep])
    for left,right in zip(a['initial_states'],b['initial_states']):
        for key in ('wrist_pos','wrist_quat','wrist_velocity','all_q','all_v','hand_q','hand_v',
                    'follower_q','follower_v','object_state','robot_root','phase'):
            np.testing.assert_array_equal(left[key],right[key],err_msg='Actual initial '+key)
        if left.get('reset_buffers')!=right.get('reset_buffers'):
            raise ValueError('Reset/history mismatch')


def analyze(root,names,paired=False,plot_episode=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result={};metas={}
    for name in names:
        path=root/name
        meta=json.loads((path/'metadata.json').read_text());metas[name]=meta
        rows=[json.loads(line) for line in (path/'physics.jsonl').open()]
        for row in rows:
            # Existing IKResult computes these with the verified FK on its raw
            # float64 output. Do not substitute a held/float32 accepted goal.
            row['raw_ik_position_error_m']=row['raw_solve']['position_error_m']
            row['raw_ik_rotation_error_rad']=row['raw_solve']['orientation_error_rad']
        fp,fr=pose_errors(np.array([r['ik_input_pos'] for r in rows]),np.array([r['ik_input_quat'] for r in rows]),
                         np.array([r['state']['wrist_pos'] for r in rows]),np.array([r['state']['wrist_quat'] for r in rows]))
        for row,p_error,r_error in zip(rows,fp,fr):
            row['ik_input_actual_position_m']=float(p_error);row['ik_input_actual_rotation_rad']=float(r_error)
        base=summarize(path);result[name]=base
        ids=meta['arm_ids'];dt=meta['physics_dt']
        from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
        from tools.rb3_revo2_ik.warm_start_ik import PoseJacobian
        from scipy.spatial.transform import Rotation
        with Path(meta['state_bank']).open() as stream:bank=json.loads(next(stream))
        kin=RB3730Kinematics(base_position=bank['base_position'],base_quaternion_xyzw=bank['base_quaternion_xyzw'])
        metrics={k:[] for k in ('A_m','A_rad','B_m','B_rad','C_m','C_rad','D_m','D_rad','E_m','E_rad','F_m','F_rad')}
        joint_errors=[];path_speeds=[];path_accels=[];actual_speeds=[];actual_accels=[];submitted_speeds=[]
        fig,axes=plt.subplots(3,2,figsize=(12,9))
        for episode in base['episodes']:
            ep=episode['episode'];r=[x for x in rows if x['episode']==ep]
            t=np.array([x['time_s'] for x in r]);init=meta['initial_states'][ep]
            q=np.array([x['q_cmd'] for x in r]);qa=np.array([x['state']['all_q'] for x in r])[:,ids]
            v=np.array([x['state']['all_v'] for x in r])[:,ids]
            np.testing.assert_array_equal(q,np.array([x['controller']['position'] for x in r])[:,ids])
            np.testing.assert_array_equal([x['hand_target'] for x in r],
                np.array([x['controller']['position'] for x in r])[:,meta['hand_ids']])
            np.testing.assert_allclose([x['controller']['state_time_s'] for x in r],t,atol=1e-12,rtol=0)
            np.testing.assert_allclose([x['controller']['command_time_s'] for x in r],t-dt,atol=1e-12,rtol=0)
            if any(x['controller']['overwrite_detected'] for x in r):raise ValueError('Native overwrite')
            pre=np.asarray(r[0]['controller']['pre_state']['all_q'])[ids]
            # At reset native arm target equals the reset joint configuration.
            speed,path_acc,acc=differences(q,v,pre,np.asarray(init['all_v'])[ids],dt)
            submitted=np.array([x['controller']['velocity'] for x in r])[:,ids]
            error=qa-q;joint_errors.extend(error);path_speeds.extend(speed);path_accels.extend(path_acc)
            actual_speeds.extend(v);actual_accels.extend(acc);submitted_speeds.extend(submitted)
            columns={'A_m':'raw_ik_position_error_m','A_rad':'raw_ik_rotation_error_rad',
                     'B_m':'postprocess_position_m','B_rad':'postprocess_rotation_rad',
                     'C_m':'tracking_fk_position_m','C_rad':'tracking_fk_rotation_rad',
                     'D_m':'runtime_fk_position_m','D_rad':'runtime_fk_rotation_rad',
                     'E_m':'wrist_position_error_m','E_rad':'wrist_rotation_error_rad',
                     'F_m':'ik_input_actual_position_m','F_rad':'ik_input_actual_rotation_rad'}
            episode['split']={}
            for label,key in columns.items():
                x=np.array([row[key] for row in r]);metrics[label].extend(x)
                peak=int(np.argmax(abs(x)))
                episode['split'][label]=dict(**stats(x),peak_time_s=float(t[peak]),episode_step=r[peak]['episode_step'])
            onset=episode['contact_s']
            episode['split_regions']={region:{label:stats([row[key] for row in r if predicate(row)])
                for label,key in columns.items()} for region,predicate in
                [('initial_0_1s',lambda row:row['time_s']<=.1),
                 ('work_after_0_1s',lambda row:row['time_s']>.1),
                 ('pre_contact',lambda row:onset is None or row['time_s']<onset),
                 ('post_contact',lambda row:onset is not None and row['time_s']>=onset)]}
            limit=np.asarray(meta['limits']['velocity'])[ids]
            episode['arm_velocity_limit_steps']=int(np.any(abs(v)>limit+1e-4,axis=1).sum())
            episode['large_command_steps']=(np.flatnonzero(np.max(abs(speed*dt),axis=1)>.1)+1).tolist()
            episode['command_acceleration_max']=np.max(abs(path_acc),axis=0).tolist()
            episode['actual_acceleration_max_including_reset']=np.max(abs(acc),axis=0).tolist()
            episode['velocity_target_equals_path_max_error']=float(np.max(abs(speed-submitted)))
            qi=np.array([x['raw_solve']['q'] for x in r]);jumps=np.diff(qi,axis=0)
            peak=int(np.argmax(np.max(abs(jumps),axis=1)))+1
            row=r[peak];previous=r[peak-1]
            _,jac=PoseJacobian(kin).evaluate(qi[peak],np.asarray(row['ik_input_pos']),
                Rotation.from_quat(row['ik_input_quat']).as_matrix(),10.)
            sv=np.linalg.svd(jac,compute_uv=False)
            episode['largest_ik_step']=dict(time_s=float(t[peak]),joint_step_rad=jumps[peak-1].tolist(),
                wrist2_rad=float(qi[peak,4]),position_input_step_m=float(np.linalg.norm(
                    np.asarray(row['ik_input_pos'])-previous['ik_input_pos'])),
                rotation_input_step_rad=float((Rotation.from_quat(row['ik_input_quat'])*
                    Rotation.from_quat(previous['ik_input_quat']).inv()).magnitude()),
                weighted_pose_jacobian_singular_values=sv.tolist(),
                jacobian_note='Existing IK residual Jacobian, translation weight 10, rotation weight 1; scale-dependent, not a manufacturer singularity criterion.')
            near=abs(v)>=.95*limit
            episode['near_actual_speed_fraction']=near.mean(0).tolist()
            durations=[]
            for j in range(6):
                edges=np.diff(np.r_[False,near[:,j],False].astype(int))
                lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
                durations.append(float(lengths.max()*dt) if len(lengths) else 0.)
            episode['longest_near_actual_speed_s']=durations
            if ep==plot_episode:
                for label,key in columns.items():
                    if label.endswith('_m'):axes[0,0].plot(t,np.array([x[key] for x in r])*1000,label=label[0])
                    else:axes[0,1].plot(t,np.rad2deg([x[key] for x in r]),label=label[0])
                for j in (3,5):
                    label=meta['joint_names'][ids[j]]
                    axes[1,0].plot(t,error[:,j],label=label)
                    axes[1,1].plot(t,v[:,j],label=label+' actual')
                    axes[1,1].plot(t,submitted[:,j],'--',label=label+' native target')
                    axes[2,0].plot(t,acc[:,j],label=label)
                axes[2,1].plot(t,[x['state']['object_state'][2]-init['object_state'][2] for x in r],label='can lift')
                if onset is not None:
                    for ax in axes.flat:ax.axvline(onset,color='gray',ls=':',label='contact onset')
        base['aggregate_split']={k:stats(v) for k,v in metrics.items()}
        base['per_joint']={key:[stats(np.asarray(value)[:,j]) for j in range(6)] for key,value in
            [('tracking_error_rad',joint_errors),('path_speed_rad_s',path_speeds),('path_acceleration_rad_s2',path_accels),
             ('actual_speed_rad_s',actual_speeds),('actual_acceleration_rad_s2',actual_accels),
             ('native_velocity_target_rad_s',submitted_speeds)]}
        base['runtime_arm']={k:np.asarray(meta['recovery_runtime'][k])[ids].tolist() for k in
            ('kp','kd','drive_type','effort_limit','velocity_limit')}
        if base['runtime_arm']['drive_type']!=[1]*6:
            raise ValueError('Approximate N·m PD terms require the verified force-drive benchmark')
        base['policy_frozen']=meta['frozen_policy_verified']
        base['raw_solver_failure_steps']=sum(not r['raw_solve']['success'] for r in rows)
        base['strict_pose_not_met_steps']=sum(r['raw_solve']['position_error_m']>1e-4 or
                                            r['raw_solve']['orientation_error_rad']>1e-3 for r in rows)
        base['optimizer_failure_steps']=sum(not r['raw_solve']['optimizer_success'] for r in rows)
        base['command_rejected_steps']=sum(r.get('ik_command_accepted') is False for r in rows)
        base['velocity_bounded_ik']=meta.get('velocity_bounded_ik',False)
        base['max_fast_fallback_counter']=max(r['ik_fallback_count'] for r in rows)
        base['approximate_force_PD_pre']={}
        for label,values in [('P',[(np.asarray(r['controller']['position'])[ids]-np.asarray(r['controller']['pre_state']['all_q'])[ids])*
                                   np.asarray(base['runtime_arm']['kp']) for r in rows]),
                             ('D',[(np.asarray(r['controller']['velocity'])[ids]-np.asarray(r['controller']['pre_state']['all_v'])[ids])*
                                   np.asarray(base['runtime_arm']['kd']) for r in rows])]:
            base['approximate_force_PD_pre'][label]=[stats(np.asarray(values)[:,j]) for j in range(6)]
        base['explicit_feedforward_max_Nm']=float(np.max(abs(np.asarray([np.asarray(r['controller']['feedforward'])[ids] for r in rows]))))
        base['timing_note']='IK is nested inside root_apply; scene_update includes trace/FK/logging. Do not sum nested columns or claim uninstrumented speed.'
        timing=json.loads((path/'realtime.json').read_text())
        base['timing']=dict(simulation_s=timing['simulation_s'],wall_s=timing['wall_s'],
            sim_per_wall=timing['simulation_s']/timing['wall_s'],
            ik_ms=stats(np.asarray(timing['ik_s'])*1000),
            inference_ms=stats([s['inference_s']*1000 for s in timing['steps']]),
            components_ms={k:stats(np.asarray(v)*1000) for k,v in timing['component_s'].items()})
        for ax,ylabel in zip(axes.flat,('Pose errors [mm]','Rotation errors [deg]','Joint error [rad]',
                                      'Velocity [rad/s]','Actual acceleration [rad/s²]','Can lift [m]')):
            ax.set(xlabel='Episode time [s]',ylabel=ylabel);ax.grid(alpha=.3);ax.legend(fontsize=7)
        fig.suptitle(name+f' / placement {plot_episode}, aligned post-step measurements')
        fig.tight_layout();fig.savefig(path/f'ik120_ep{plot_episode}.png',dpi=140);plt.close(fig)
        (path/'ik120_summary.json').write_text(json.dumps(base,indent=2)+'\n')
        print(name,base['successes'],'/',len(base['episodes']),'proxy',base['lift_proxy_count'],
              'C mm',base['aggregate_split']['C_m'],'w3',base['per_joint']['tracking_error_rad'][5],flush=True)
    pairs=[]
    if paired:
        if len(names)%2:raise ValueError('Pair adjacent baseline/candidate runs')
        for a,b in zip(names[::2],names[1::2]):
            paired_initials(metas[a],metas[b])
            initial_policy=[]
            for name in (a,b):
                records=json.loads((root/name/'policy.json').read_text())
                initial_policy.append({r['episode']:r for r in records if r['time_s']==0})
            for ep in range(len(metas[a]['ends'])):
                for key in ('observation','action','phase'):
                    np.testing.assert_array_equal(initial_policy[0][ep][key],initial_policy[1][ep][key],
                                                  err_msg='Initial policy/history '+key)
            old=result[a]['episodes'];new=result[b]['episodes']
            pairs.append(dict(baseline=a,candidate=b,actual_initial_states_equal=True,
                initial_policy_observations_actions_equal=True,
                success_to_failure=[x['episode'] for x,y in zip(old,new) if x['success'] and not y['success']],
                failure_to_success=[x['episode'] for x,y in zip(old,new) if not x['success'] and y['success']],
                proxy_success_to_failure=[x['episode'] for x,y in zip(old,new) if x['lift_proxy'] and not y['lift_proxy']]))
            # Zoom on the requested episode: small wrist motion versus large
            # joint-coordinate changes. Each run is its own live-policy target.
            fig,axes=plt.subplots(2,3,figsize=(14,8))
            for name,color in ((a,'#708090'),(b,'#db8a38')):
                with (root/name/'physics.jsonl').open() as stream:
                    trace=[r for line in stream if (r:=json.loads(line))['episode']==plot_episode]
                t=np.array([r['time_s'] for r in trace]);mask=t>=.9
                if not mask.any():mask[:]=True
                q=np.array([r['raw_solve']['q'] for r in trace]);v=np.array([r['state']['all_v'] for r in trace])[:,metas[name]['arm_ids']]
                position=np.array([r['ik_input_pos'] for r in trace]);quat=np.array([r['ik_input_quat'] for r in trace])
                rotation_step=(Rotation.from_quat(quat[1:])*Rotation.from_quat(quat[:-1]).inv()).magnitude()
                for j,style in enumerate(('-',':','--')):
                    axes[0,0].plot(t[mask],((position-position[mask][0])*1000)[mask,j],style,color=color,label=name+' '+'xyz'[j])
                axes[0,1].plot(t[1:][mask[1:]],np.rad2deg(rotation_step[mask[1:]]),color=color,label=name)
                axes[0,2].plot(t[1:][mask[1:]],np.rad2deg(np.max(abs(np.diff(q,axis=0)),axis=1)[mask[1:]]),color=color,label=name)
                axes[1,0].plot(t[mask],np.rad2deg(q[mask,4]),color=color,label=name)
                pe,_=pose_errors(position,quat,np.array([r['state']['wrist_pos'] for r in trace]),np.array([r['state']['wrist_quat'] for r in trace]))
                axes[1,1].plot(t[mask],pe[mask]*1000,color=color,label=name)
                axes[1,2].plot(t[1:][mask[1:]],np.max(abs(np.diff(v,axis=0)/metas[name]['physics_dt']),axis=1)[mask[1:]],color=color,label=name)
            titles=('IK input displacement [mm]','IK input rotation step [deg]','Largest IK joint step [deg]',
                    'Wrist2 coordinate [deg]','IK input to actual wrist error [mm]','Actual arm acceleration peak [rad/s²]')
            for ax,title in zip(axes.flat,titles):ax.set(xlabel='Episode time [s]',title=title);ax.grid(alpha=.3)
            axes[0,0].legend(fontsize=6);axes[0,2].legend(fontsize=7)
            fig.suptitle(f'Placement {plot_episode}: live policy, same initial state, no time shift')
            fig.tight_layout();fig.savefig(root/f'singularity_{a}_{b}_ep{plot_episode}.png',dpi=140);plt.close(fig)
    (root/'ik120_comparison.json').write_text(json.dumps(dict(runs=result,pairs=pairs),indent=2)+'\n')
    if pairs:
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        labels=[n.replace('old20_','old20\n').replace('held20_','held20\n') for n in names]
        values=[
            [result[n]['aggregate_split']['C_m']['p95']*1000 for n in names],
            [np.rad2deg(result[n]['per_joint']['tracking_error_rad'][5]['p95']) for n in names],
            [max(x['max'] for x in result[n]['per_joint']['actual_acceleration_rad_s2']) for n in names],
            [result[n]['successes'] for n in names]]
        for ax,y,title in zip(axes.flat,values,('Wrist C P95 [mm]','Wrist3 joint error P95 [deg]',
                                              'Peak actual arm acceleration [rad/s²]','Task successes / 20')):
            ax.bar(labels,y,color=['#708090','#db8a38']*len(pairs));ax.set_title(title)
            for i,value in enumerate(y):ax.text(i,value,f'{value:.3g}',ha='center',va='bottom')
            ax.set_ylim(0,max(y)*1.2);ax.grid(axis='y',alpha=.25)
        proxy=[result[n]['lift_proxy_count'] for n in names]
        axes[1,1].plot(range(len(names)),proxy,'kx',markersize=9,label='Lift/contact proxy')
        axes[1,1].legend(loc='lower left')
        for i,value in enumerate(proxy):
            if value!=values[-1][i]:axes[1,1].annotate(f'proxy {value}',(i,value),xytext=(-45,-20),textcoords='offset points')
        fig.suptitle('120 Hz IK: wrist3-only candidate; identical paired initial states')
        fig.tight_layout();fig.savefig(root/'paired_comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('runs',nargs='+');p.add_argument('--paired',action='store_true')
    p.add_argument('--plot-episode',type=int,default=0)
    args=p.parse_args();analyze(args.root,args.runs,args.paired,args.plot_episode)
