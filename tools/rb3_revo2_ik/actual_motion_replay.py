"""Recorded physical floating motion -> mounted drives (not a policy evaluation)."""
import json
from pathlib import Path
import numpy as np


def load_actual_motion(directory, episodes, physics_dt):
    directory=Path(directory)
    meta=json.loads((directory/'metadata.json').read_text())
    if meta['mode']!='floating' or meta['stage']!='grasp':
        raise ValueError('Need an actual successful floating policy rollout')
    if not np.isclose(meta['physics_dt'],physics_dt,atol=1e-12,rtol=0):
        raise ValueError('Physics timestep mismatch; no implicit resampling')
    rows=[json.loads(line) for line in (directory/'physics.jsonl').open()]
    sequences=[]
    for ep in range(episodes):
        if not meta['ends'][ep]['termination']['success']:
            raise ValueError(f'Source episode {ep} was not successful')
        seq=[r for r in rows if r['episode']==ep]
        if not seq:raise ValueError(f'Missing episode {ep}')
        np.testing.assert_allclose([r['time_s'] for r in seq],
            np.arange(1,len(seq)+1)*physics_dt,atol=1e-10,rtol=0)
        for r in seq:
            for key,shape in [('wrist_pos',(3,)),('wrist_quat',(4,)),('hand_q',(6,)),('follower_q',(5,))]:
                value=np.asarray(r['state'][key])
                if value.shape!=shape or not np.isfinite(value).all():raise ValueError(f'Invalid measured {key}')
            if not np.isclose(np.linalg.norm(r['state']['wrist_quat']),1,atol=1e-5):
                raise ValueError('Invalid XYZW quaternion')
        sequences.append(seq)
    return meta,sequences


def run_actual_motion(env, root_action, hand_action, command, robot, source_meta,
                      sequences, counter, ends, snapshot, initial_records, app, output):
    import torch
    from tools.rb3_revo2_ik.trace_arm_execution import serial
    device=env.device
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    source_names=[source_meta['joint_names'][i] for i in source_meta['hand_ids']]
    if source_names!=[robot.joint_names[i] for i in command.hand_joint_ids]:
        raise ValueError('Recorded leader joint names/order mismatch')
    # Match the recorded follower measurements by names too, not array position.
    follower_source_ids=[source_meta['joint_names'].index(robot.joint_names[i])
                         for i in command.follower_ids]
    for seq in sequences:
        for row in seq:
            np.testing.assert_allclose(row['state']['follower_q'],
                np.asarray(row['state']['all_q'])[follower_source_ids],atol=0,rtol=0)
    limits=robot.data.soft_joint_pos_limits.torch[:,command.hand_joint_ids]
    old_base=hand_action.get_base_joint_pos
    termination_log=[]
    exported={k:[] for k in ('episode','time_s','wrist_pos','wrist_quat_xyzw','revo2_joints')}
    try:
        for ep,seq in enumerate(sequences):
            if ep:
                counter['active']=False
                env._reset_idx(torch.tensor([0],device=device))
            initial=initial_records[-1];expected=source_meta['initial_states'][ep]
            for key in ('wrist_pos','wrist_quat','wrist_velocity','hand_q','hand_v',
                        'follower_q','follower_v','object_state','phase'):
                np.testing.assert_allclose(initial[key],expected[key],atol=2e-6,rtol=0,
                                           err_msg=f'Initial floating/replay mismatch: {ep} {key}')
            first_termination=None
            for i,row in enumerate(seq):
                if not app.is_running():raise RuntimeError('Simulation closed before motion completed')
                state=row['state']
                # Offline right-endpoint ZOH: sample t_k commanded BEFORE interval
                # (t_{k-1},t_k], measured after it. No post-hoc lag removal.
                root_action.target_pos=tensor([state['wrist_pos']])
                root_action.target_quat=tensor([state['wrist_quat']])
                root_action.ik_target_pos=root_action.target_pos.clone()
                root_action.ik_target_quat=root_action.target_quat.clone()
                r=root_action.solve(root_action.target_pos[0],root_action.target_quat[0],root_action.goal[0])
                root_action.solve_result=r
                if r.success and r.finite and not r.joint_limit_violation:
                    root_action.goal[0]=tensor(r.q)
                # Existing six-leader position clamp and deterministic follower
                # mapping are retained, without residual/reference decoding.
                measured=tensor([state['hand_q']])
                hand_action.get_base_joint_pos=lambda:measured
                hand_action.processed_actions.zero_()
                command.time_steps[:]=row['command_frame']
                counter.update(active=True,command_frame=row['command_frame'],
                    command_time=i*env.physics_dt,action=None,source_actual=row,
                    recorded_hand_clipped=bool(((measured<limits[...,0])|(measured>limits[...,1])).any()))
                root_action.apply_actions();hand_action.apply_actions()
                env._sim_step_counter+=1
                env.scene.write_data_to_sim()
                env.sim.step(render=env.sim.has_gui)
                env.scene.update(dt=env.physics_dt)
                for key,value in [('episode',ep),('time_s',row['time_s']),('wrist_pos',state['wrist_pos']),
                                  ('wrist_quat_xyzw',state['wrist_quat']),('revo2_joints',state['hand_q'])]:
                    exported[key].append(value)
                if (i+1)%env.cfg.decimation==0:
                    env.episode_length_buf[:]=(i+1)//env.cfg.decimation
                    flags={}
                    for name in env.termination_manager.active_terms:
                        cfg=env.termination_manager.get_term_cfg(name)
                        flags[name]=bool(cfg.func(env,**cfg.params)[0])
                    termination_log.append(dict(episode=ep,time_s=row['time_s'],terms=flags))
                    if first_termination is None and any(flags.values()):
                        first_termination=dict(time_s=row['time_s'],terms=flags)
            if first_termination is None:raise RuntimeError('Recorded motion ended without existing termination')
            ends.append(dict(episode=ep,state=snapshot(),termination=first_termination['terms'],
                first_termination_time_s=first_termination['time_s'],full_recording_replayed=True))
            print('[actual replay]',ep,'first termination',first_termination,flush=True)
        np.savez_compressed(output/'actual_motion_targets.npz',**{k:np.asarray(v) for k,v in exported.items()},
                            quaternion_order='xyzw',physics_dt=env.physics_dt)
        (output/'termination_checks.json').write_text(json.dumps(termination_log,default=serial,indent=2)+'\n')
    finally:
        counter['active']=False
        hand_action.get_base_joint_pos=old_base
