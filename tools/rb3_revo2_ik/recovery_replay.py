"""Policy-free command/actual-state controls and isolated arm diagnostics."""
import json
import numpy as np


def validate_unique_placements(initials):
    xy=np.asarray([s['object_state'][:2] for s in initials])
    if not np.isfinite(xy).all():raise ValueError('Nonfinite placement bank')
    if len(xy)>1:
        distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
        np.fill_diagonal(distance,np.inf)
        if distance.min()<1e-5:raise ValueError('Repeated placements: seed alone is not a new state bank')
    return xy


def schedule(sequence,speed=1,holds=False,dt=1/120):
    """Known sampled path: ZOH stretching only; no hidden phase/time shift."""
    if holds:
        selected=np.linspace(0,len(sequence)-1,5,dtype=int)
        return [(sequence[i],hold) for hold,i in enumerate(selected) for _ in range(round(4/dt))]
    return [(row,None) for row in sequence for _ in range(speed)]


def run_recovery(env,root,hand,command,robot,sm,sequences,counter,ends,snapshot,initials,app,out,args):
    import torch
    from tools.rb3_revo2_ik.trace_arm_execution import serial
    device=env.device;tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=device)
    floating=args.mode=='floating'
    source_names=sm['joint_names']
    source_ids=[source_names.index(robot.joint_names[i]) for i in command.hand_joint_ids]
    hand_ids=[robot.joint_names.index(n) for n in source_names]
    old_base=hand.get_base_joint_pos
    checks=[]
    try:
        for ep,seq in enumerate(sequences):
            if ep:
                counter['active']=False;env._reset_idx(torch.tensor([0],device=device))
            keys=['wrist_pos','wrist_quat','wrist_velocity','hand_q','hand_v','follower_q','follower_v','phase']
            if not args.recovery_no_can:keys.append('object_state')
            for key in keys:np.testing.assert_allclose(initials[-1][key],sm['initial_states'][ep][key],atol=2e-6,rtol=0)
            first=None
            for i,(row,hold) in enumerate(schedule(seq,args.recovery_speed,args.recovery_holds,env.physics_dt)):
                if not app.is_running():raise RuntimeError('Replay interrupted')
                c=row['controller'];s=row['state']
                # F1 replays already decoded equilibrium targets; other modes
                # use measured floating wrist, WITHOUT residual or response filter.
                pos=c['wrist_equilibrium_pos'] if args.recovery_kind=='F1' else s['wrist_pos']
                quat=c['wrist_equilibrium_quat'] if args.recovery_kind=='F1' else s['wrist_quat']
                root.target_pos=tensor([pos]);root.target_quat=tensor([quat])
                if not floating:
                    root.ik_target_pos=root.target_pos.clone();root.ik_target_quat=root.target_quat.clone()
                    r=root.solve(root.target_pos[0],root.target_quat[0],root.goal[0]);root.solve_result=r
                    if r.success and r.finite and not r.joint_limit_violation:root.goal[0]=tensor(r.q)
                root.apply_actions()
                if args.recovery_kind=='F2':
                    measured=tensor([s['hand_q']]);hand.get_base_joint_pos=lambda:measured
                    hand.processed_actions.zero_();hand.apply_actions()
                else:
                    # These include the existing deterministic mimic commands;
                    # no additional independent actions or second coupling pass.
                    for key,method in [('position',robot.set_joint_position_target_index),
                                       ('velocity',robot.set_joint_velocity_target_index),
                                       ('feedforward',robot.set_joint_effort_target_index)]:
                        method(target=tensor([c[key]]),joint_ids=hand_ids)
                    hand._last_joint_target.copy_(tensor([np.asarray(c['position'])[source_ids]]))
                command.time_steps[:]=row['command_frame']
                counter.update(active=True,command_frame=row['command_frame'],command_time=i*env.physics_dt,
                    action=None,source_actual=row,hold_index=hold)
                env._sim_step_counter+=1;env.scene.write_data_to_sim()
                env.sim.step(render=env.sim.has_gui);env.scene.update(dt=env.physics_dt)
                if (i+1)%env.cfg.decimation==0:
                    env.episode_length_buf[:]=(i+1)//env.cfg.decimation
                    flags={}
                    for name in env.termination_manager.active_terms:
                        cfg=env.termination_manager.get_term_cfg(name);flags[name]=bool(cfg.func(env,**cfg.params)[0])
                    checks.append(dict(episode=ep,time_s=(i+1)*env.physics_dt,terms=flags))
                    if first is None and any(flags.values()):first=checks[-1]
            if first is None:raise RuntimeError('No termination at recorded end')
            ends.append(dict(episode=ep,state=snapshot(),termination=first['terms'],
                first_termination_time_s=first['time_s'],full_recording_replayed=True))
            print('[recovery replay]',args.recovery_kind,ep,first,flush=True)
        (out/'termination_checks.json').write_text(json.dumps(checks,default=serial)+'\n')
    finally:
        counter['active']=False;hand.get_base_joint_pos=old_base
