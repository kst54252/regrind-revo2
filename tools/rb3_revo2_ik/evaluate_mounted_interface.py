"""Independent frozen-policy evaluator: floating, legacy arm, or minimal bridge."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=('floating','legacy','simple'),required=True)
p.add_argument('--stage',choices=('ab','grasp','actual','recovery'),default='grasp')
p.add_argument('--recovery-capture',action='store_true')
p.add_argument('--realtime-view',action='store_true',help='Live policy with reduced diagnostics and wall-clock pacing; does not guarantee real-time throughput')
p.add_argument('--record-video',action='store_true',help='Record native simulation frames at control FPS, not wall-clock speed')
p.add_argument('--fast-ik',action='store_true',help='Opt-in warm-first existing IK with bounded-step validation and full fallback')
p.add_argument('--ik-policy-rate',action='store_true',help='Live simple mode: solve IK on the first physics substep of each policy action, holding accepted q between solves')
p.add_argument('--velocity-bounded-ik',action='store_true',help='Opt-in IK inside q_cmd +/- velocity_limit*physics_dt; explicit 5 mm/.05 rad pose budget')
p.add_argument('--ik-acceleration-limit',type=float,default=0.,help='Optional command-only rad/s² bound inside IK; reports best feasible pose even outside the pose budget')
p.add_argument('--zero-actions',action='store_true',help='Live reference-only comparison: all 12 residuals zero, no policy inference')
p.add_argument('--recovery-kind',choices=('F1','F1_pipeline','F2','R1'),default='R1')
p.add_argument('--recovery-no-can',action='store_true')
p.add_argument('--recovery-speed',type=int,choices=(1,4),default=1)
p.add_argument('--recovery-holds',action='store_true')
p.add_argument('--arm-gains-key',choices=('baseline','c1','c2','c3'),default='baseline')
p.add_argument('--arm-gains-file',default=str(ROOT/'config/experiments/rb3_precision_candidates.json'),
    help='Opt-in diagnostic gain table; does not change the task defaults')
p.add_argument('--arm-velocity-path',action='store_true')
p.add_argument('--arm-response-physics',action='store_true')
p.add_argument('--new-state-seed',type=int)
p.add_argument('--save-state-bank')
p.add_argument('--transfer-config',help='Opt-in screened live controller JSON; never applied to other modes')
p.add_argument('--actual-source',default='outputs/diagnostics/minimal_interface_20260907/floating20',
    help='Successful measured floating rollout directory, used only by --stage actual')
p.add_argument('--episodes',type=int,default=1)
p.add_argument('--checkpoint',required=True)
p.add_argument('--states',default='outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl')
p.add_argument('--output',required=True)
p.add_argument('--response-tau',type=float,default=0.,
    help='Experimental simple-mode wrist command response time in seconds; 0 disables')
p.add_argument('--headless',dest='legacy_headless',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.transfer_config:
    if args.mode!='simple' or args.stage!='grasp':raise ValueError('Transfer config requires live simple mode')
    selected=json.loads(Path(args.transfer_config).read_text())
    if selected['arm_gains_key'] not in ('baseline','c1','c2','c3'):raise ValueError('Unknown gain profile')
    for key in ('arm_gains_key','arm_velocity_path','arm_response_physics','response_tau'):
        setattr(args,key,selected[key])
    if 'arm_gains_file' in selected:args.arm_gains_file=selected['arm_gains_file']
    if 'velocity_bounded_ik' in selected:args.velocity_bounded_ik=selected['velocity_bounded_ik']
    if 'ik_acceleration_limit' in selected:args.ik_acceleration_limit=selected['ik_acceleration_limit']
args.headless=args.legacy_headless;del args.legacy_headless
if args.record_video:args.enable_cameras=True
out=Path(args.output)
if out.exists():raise FileExistsError(out)
app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.sensors import ContactSensorCfg
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata
import regrind.tasks
from regrind.tasks.manager_based.dexterous.mdp.simple_mounted_interface import SimpleMountedWristCfg
from tools.rb3_revo2_ik.frozen_policy_adapter import FrozenPolicyAdapter
from tools.rb3_revo2_ik.trace_arm_execution import array,serial
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.analyze_arm_execution import pose_errors


def main():
    if args.zero_actions and args.stage!='grasp':raise ValueError('Zero agent supports live grasp only')
    import carb.settings
    import omni.physx
    from omni.physx.bindings import _physx
    native_settings={key:carb.settings.get_settings().get(getattr(_physx,key)) for key in
                     ('SETTING_NUM_THREADS','SETTING_UPDATE_TO_USD','SETTING_UPDATE_VELOCITIES_TO_USD','SETTING_PHYSX_DISPATCHER')}
    if args.fast_ik and args.mode!='simple':raise ValueError('Fast IK requires simple mode')
    if not np.isfinite(args.ik_acceleration_limit) or args.ik_acceleration_limit<0 or (args.ik_acceleration_limit and not args.velocity_bounded_ik):
        raise ValueError('Command acceleration bound requires velocity-bounded IK and a finite nonnegative value')
    if args.velocity_bounded_ik and (args.mode!='simple' or args.stage not in ('grasp','ab') or
                                   not args.arm_response_physics or args.ik_policy_rate):
        raise ValueError('Velocity-bounded IK requires live/AB physics-rate simple mode')
    if args.ik_policy_rate and (args.mode!='simple' or args.stage!='grasp' or not args.arm_response_physics):
        raise ValueError('Policy-rate IK requires live simple mode with physics-rate response')
    if args.realtime_view and (args.stage!='grasp' or args.recovery_capture or args.save_state_bank):
        raise ValueError('Realtime view is live grasp only, without recovery capture/state collection')
    if args.record_video and args.stage!='grasp':raise ValueError('Video capture supports live grasp only')
    if not np.isfinite(args.response_tau) or args.response_tau<0:
        raise ValueError('--response-tau must be finite and nonnegative')
    if args.response_tau and args.mode!='simple':
        raise ValueError('--response-tau requires --mode simple')
    if args.stage=='actual' and (args.mode!='simple' or args.response_tau!=0):
        raise ValueError('Actual motion replay requires simple mode with response filter disabled')
    if args.stage=='recovery' and (args.response_tau or args.mode!=('floating' if args.recovery_kind.startswith('F') else 'simple')):
        raise ValueError('Recovery kind and embodiment mismatch, or unwanted response filter')
    if args.new_state_seed is not None and (args.mode!='legacy' or args.stage!='grasp' or not args.save_state_bank):
        raise ValueError('New states must be captured with baseline live legacy policy and --save-state-bank')
    if args.save_state_bank and Path(args.save_state_bank).exists():raise FileExistsError(args.save_state_bank)
    if args.arm_velocity_path and args.mode!='simple':raise ValueError('Velocity option is simple-mode only')
    if args.arm_response_physics and (args.mode!='simple' or args.response_tau<=0 or args.stage not in ('grasp','ab')):
        raise ValueError('Physics response is a live simple-mode positive-tau experiment')
    bank_records=[json.loads(line) for line in Path(args.states).open()]
    bank_meta=bank_records[0]
    bank={r['episode']:r for r in bank_records if r['event']=='initialization'}
    if args.episodes<1 or args.episodes>20 or len(bank)<args.episodes+1:raise ValueError('Need complete state bank plus final autoreset state')
    floating=args.mode=='floating'
    task='Regrind-Floating-Revo2-TunaCan-Play-v0' if floating else 'Regrind-RB3-Revo2-TunaCan-Online-Play-v0'
    cfg=parse_env_cfg(task,device=args.device,num_envs=1,use_fabric=True)
    cfg.seed=bank_meta['seed'] if args.new_state_seed is None else args.new_state_seed
    cfg.commands.reference.trajectory_path=bank_meta['reference']
    if args.new_state_seed is not None:
        # PLAY disables placement sampling; enable ONLY for the explicit bank
        # capture. Existing training-defined XY ranges and all dynamics remain.
        cfg.commands.reference.randomize_object_xy=True
    gains=None
    if args.arm_gains_key!='baseline' or Path(args.arm_gains_file).resolve()!=ROOT/'config/experiments/rb3_precision_candidates.json':
        if floating:raise ValueError('Arm gains cannot be used for floating')
        from regrind.data.rb3_revo2_reference import RB3_JOINT_NAMES
        gains=json.loads(Path(args.arm_gains_file).read_text())[args.arm_gains_key]
        for name in ('kp','kd'):
            values=np.asarray(gains[name],dtype=float)
            if values.shape!=(6,) or not np.isfinite(values).all() or np.any(values<=0):
                raise ValueError('Arm gain table requires six positive finite '+name+' values')
        cfg.scene.robot.actuators['rb3_arm'].stiffness=dict(zip(RB3_JOINT_NAMES,gains['kp']))
        cfg.scene.robot.actuators['rb3_arm'].damping=dict(zip(RB3_JOINT_NAMES,gains['kd']))
    if args.mode=='simple':
        old=cfg.actions.root_pose
        cfg.actions.root_pose=SimpleMountedWristCfg(asset_name='robot',scale_pos=old.scale_pos,
            scale_rot=old.scale_rot,raw_clip=old.raw_clip,response_tau=args.response_tau,
            velocity_path=args.arm_velocity_path,response_at_physics=args.arm_response_physics,fast_ik=args.fast_ik,
            ik_policy_rate=args.ik_policy_rate,velocity_bounded_ik=args.velocity_bounded_ik,
            ik_acceleration_limit=args.ik_acceleration_limit)
    cfg.scene.object.spawn.activate_contact_sensors=True
    cfg.scene.can_robot_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Object',update_period=0.)
    if args.record_video:
        cfg.viewer.eye=(1.4,1.2,.85);cfg.viewer.lookat=(.2,0.,.15)
        cfg.viewer.resolution=(1280,720)
    env=gym.make(task,cfg=cfg,render_mode='rgb_array' if args.record_video else None).unwrapped
    if env.event_manager.active_terms:raise ValueError('Evaluation must be deterministic')
    command=env.command_manager.get_term('reference');robot=command.robot
    root_action=env.action_manager.get_term('root_pose');hand_action=env.action_manager.get_term('joint_pos')
    sensor=env.scene.sensors['can_robot_contact']
    arm_ids=[] if floating else [robot.joint_names.index(n) for n in bank_meta['joint_names']]
    hand_ids=list(command.hand_joint_ids);followers=list(command.follower_ids)
    kin=None if floating else RB3730Kinematics(base_position=bank_meta['base_position'],base_quaternion_xyzw=bank_meta['base_quaternion_xyzw'])
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=env.device)
    counter={'placement':0,'episode':0,'step':0,'active':False,'command_frame':0,'action':None}
    observations_checked=0
    initial_records=[];ends=[];samples=[];policy_records=[]
    original_sample=command._sample_placement
    def sample(ids):
        original_sample(ids)
        i=counter['placement'];counter['placement']+=1
        if args.new_state_seed is None:command.placement_offset[ids]=tensor(bank[i]['placement_offset'])
    command._sample_placement=sample
    original_resample=command._resample_command
    def resample(ids):
        original_resample(ids)
        expected=bank[counter['placement']-1]
        if floating:
            # Reset only: align common physical wrist pose and zero wrist twist
            # with mounted reset (arm starts at zero joint velocity).
            pose=np.r_[expected['actual_base_pos'],expected['actual_base_quat_xyzw']]
            robot.write_root_link_pose_to_sim_index(root_pose=tensor(pose[None]),env_ids=ids)
            robot.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1,6))),env_ids=ids)
        if args.stage=='ab' or args.recovery_no_can:
            state=array(command.object.data.root_state_w).copy();state[0,0]+=10.
            command.object.write_root_link_pose_to_sim_index(root_pose=tensor(state[:,:7]),env_ids=ids)
    command._resample_command=resample
    def snapshot():
        return dict(wrist_pos=array(command.current_hand_wrist_pos)[0],wrist_quat=array(command.current_hand_wrist_quat)[0],
            fingertips=array(command.current_fingertips_pos)[0],
            wrist_velocity=array(robot.data.body_link_vel_w)[0,command.wrist_body_id],
            hand_q=array(command.current_hand_joint_pos)[0],hand_v=array(command.current_hand_joint_vel)[0],
            follower_q=array(robot.data.joint_pos)[0,followers],follower_v=array(robot.data.joint_vel)[0,followers],
            object_state=array(command.object.data.root_state_w)[0],
            all_q=array(robot.data.joint_pos)[0],all_v=array(robot.data.joint_vel)[0],
            robot_root=array(robot.data.root_state_w)[0],phase=int(command.time_steps[0]))
    def verify_initial():
        s=snapshot();expected=bank[counter['episode']]
        if args.new_state_seed is not None:
            s.update(episode=counter['episode'],placement_offset=array(command.placement_offset)[0],
                     applied_arm_target=array(root_action.applied_joint_target)[0])
            initial_records.append(s);return
        np.testing.assert_allclose(s['wrist_pos'],expected['actual_base_pos'],atol=2e-6,rtol=0)
        np.testing.assert_allclose(s['wrist_quat'],expected['actual_base_quat_xyzw'],atol=2e-6,rtol=0)
        names=bank_meta['user_joint_names']
        for key,ids in [('hand_q',hand_ids),('follower_q',followers),('hand_v',hand_ids),('follower_v',followers)]:
            source='all_joint_vel' if key.endswith('_v') else 'all_joint_pos'
            np.testing.assert_allclose(s[key],np.array(expected[source])[[names.index(robot.joint_names[i]) for i in ids]],atol=1e-6,rtol=0)
        if not floating:
            np.testing.assert_allclose(s['all_q'],expected['all_joint_pos'],atol=1e-6,rtol=0)
            np.testing.assert_allclose(s['all_v'],expected['all_joint_vel'],atol=1e-6,rtol=0)
        if args.stage!='ab' and not args.recovery_no_can:np.testing.assert_allclose(s['object_state'],expected['object_root_state'],atol=1e-6,rtol=0)
        if s['phase']!=expected['reference_frame']:raise ValueError('Phase mismatch')
        s['episode']=counter['episode']
        if args.recovery_capture:
            # Read reset-owned buffers only; never advance observation history.
            s['reset_buffers']={key:array(getattr(root_action,key)) for key in
                ('raw_actions','processed_actions','target_pos','target_quat','goal','applied',
                 'ik_target_pos','ik_target_quat','bounded_velocity') if isinstance(getattr(root_action,key,None),torch.Tensor)}
            s['reset_buffers']['previous_action']=array(env.action_manager.prev_action)
            s['reset_buffers']['action']=array(env.action_manager.action)
        initial_records.append(s)
    view_env=env
    if args.record_video:
        # Original record-video integration, FPS derived from unchanged control dt.
        view_env=gym.wrappers.RecordVideo(env,video_folder=str(out/'video'),
            episode_trigger=lambda episode:episode==1,video_length=10000,
            fps=round(1/env.step_dt),disable_logger=True,name_prefix='mounted_policy_1x_recorded')
    wrapper=RslRlVecEnvWrapper(view_env)
    agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    adapter=None;source_meta=None;sequences=None
    if args.stage in ('actual','recovery'):
        from tools.rb3_revo2_ik.actual_motion_replay import load_actual_motion,run_actual_motion
        source_meta,sequences=load_actual_motion(args.actual_source,args.episodes,env.physics_dt)
        if source_meta['checkpoint_sha256']!=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest():
            raise ValueError('Source checkpoint mismatch')
        if Path(source_meta['reference']).resolve()!=Path(bank_meta['reference']).resolve():
            raise ValueError('Source reference mismatch')
    elif not args.zero_actions:
        runner=OnPolicyRunner(wrapper,agent.to_dict(),log_dir=None,device=env.device)
        runner.load(args.checkpoint);policy=runner.get_inference_policy(device=env.device)
        adapter=FrozenPolicyAdapter(policy)
    counter['placement']=0  # Wrapper construction resets once; evaluation starts anew.
    obs,_=wrapper.reset();verify_initial()
    from isaaclab_physx.physics import PhysxManager as SimulationManager
    contact_paths=robot.root_view.link_paths[0]
    contact_view=SimulationManager.get_physics_sim_view().create_rigid_contact_view(
        ['/World/envs/env_0/Object'],filter_patterns=[contact_paths])
    out.mkdir(parents=True,exist_ok=args.record_video)
    f=(out/'physics.jsonl').open('x')
    probe=None
    if args.recovery_capture or args.stage=='recovery':
        from tools.rb3_revo2_ik.recovery_probe import RecoveryProbe
        probe=RecoveryProbe(env,counter,snapshot,root_action)
        if gains is not None:
            np.testing.assert_allclose(np.asarray(probe.runtime['kp'])[arm_ids],gains['kp'])
            np.testing.assert_allclose(np.asarray(probe.runtime['kd'])[arm_ids],gains['kd'])
    original_update=env.scene.update
    def update(dt):
        original_update(dt)
        if not counter['active']:return
        counter['step']+=1
        if args.realtime_view:return  # Physics/action/observation updates remain unchanged.
        s=snapshot()
        target_pos=array(root_action.target_pos);target_quat=array(root_action.target_quat)
        pe,re=pose_errors(target_pos,target_quat,s['wrist_pos'][None],s['wrist_quat'][None])
        forces=array(contact_view.get_contact_force_matrix(dt=env.physics_dt))
        if forces.size==0:raise RuntimeError('Missing can-robot contact filter data')
        contact=float(np.max(np.linalg.norm(forces,axis=-1)))
        qtarget=array(hand_action.last_joint_target)[0]
        row=dict(episode=counter['episode'],physics_step=int(env._sim_step_counter),episode_step=counter['step'],
            time_s=counter['step']*env.physics_dt,command_frame=counter['command_frame'],
            command_time_s=counter['command_time'],state=s,action=counter['action'],
            desired_wrist_pos=target_pos[0],desired_wrist_quat=target_quat[0],
            wrist_position_error_m=float(pe[0]),wrist_rotation_error_rad=float(re[0]),
            hand_target=qtarget,hand_error_rad=qtarget-s['hand_q'],
            object_keypoint_error_m=float(torch.linalg.vector_norm(command.current_object_keypoints_pos-command.target_object_keypoints_pos,dim=-1).mean()),
            can_robot_contact_N=contact,effort_saturation='UNKNOWN')
        row['can_robot_contact_by_body_N']=np.linalg.norm(forces.reshape(-1,3),axis=-1)
        if probe is not None:row['controller']=probe.last
        if args.stage=='recovery':
            source=sequences[counter['episode']][counter['step']-1] if args.recovery_kind=='F1_pipeline' else counter['source_actual']
            row.update(source_time_s=source['time_s'],
                source_state=source['state'],recovery_kind=args.recovery_kind,
                diagnostic_hold=counter.get('hold_index'),policy_called=False)
        if args.stage=='actual':
            source=counter['source_actual']
            row.update(source_time_s=source['time_s'],source_object_state=source['state']['object_state'],
                recorded_hand_q=source['state']['hand_q'],recorded_hand_clipped=counter['recorded_hand_clipped'],
                recorded_hand_error_rad=np.asarray(source['state']['hand_q'])-s['hand_q'],
                recorded_follower_error_rad=np.asarray(source['state']['follower_q'])-s['follower_q'],
                policy_called=False)
        limits=array(robot.data.soft_joint_pos_limits)[0];vlim=array(robot.data.joint_vel_limits)[0]
        row['position_limit_violation']=bool(np.any(s['all_q']<limits[:,0]-1e-6)|np.any(s['all_q']>limits[:,1]+1e-6))
        row['velocity_limit_violation']=bool(np.any(abs(s['all_v'])>vlim+1e-4))
        row['speed_near_limit']=abs(s['all_v'])>=.95*vlim
        if not floating:
            qcmd=array(root_action.applied_joint_target)[0]
            qik=array(root_action.goal)[0] if args.mode=='simple' else array(root_action.last_joint_target)[0]
            ikpos=array(root_action.ik_target_pos) if args.mode=='simple' else target_pos
            ikquat=array(root_action.ik_target_quat) if args.mode=='simple' else target_quat
            if args.ik_policy_rate:
                ikpos=array(root_action.last_ik_input_pos);ikquat=array(root_action.last_ik_input_quat)
                row['ik_updated_this_step']=root_action.ik_updated_this_step
                row['response_target_pos']=array(root_action.ik_target_pos)[0]
                row['response_target_quat']=array(root_action.ik_target_quat)[0]
            fkpos,fkquat=kin.forward(qik);ap,ar=pose_errors(ikpos,ikquat,fkpos[None],fkquat[None])
            cpos,cquat=kin.forward(qcmd);cp,cr=pose_errors(cpos[None],cquat[None],s['wrist_pos'][None],s['wrist_quat'][None])
            actual_fk=kin.forward(s['all_q'][arm_ids]);fp,fr=pose_errors(actual_fk[0][None],actual_fk[1][None],s['wrist_pos'][None],s['wrist_quat'][None])
            if args.recovery_capture:
                bp,br=pose_errors(fkpos[None],fkquat[None],cpos[None],cquat[None])
                cp_fk,cr_fk=pose_errors(cpos[None],cquat[None],actual_fk[0][None],actual_fk[1][None])
                row.update(postprocess_position_m=float(bp[0]),postprocess_rotation_rad=float(br[0]),
                           tracking_fk_position_m=float(cp_fk[0]),tracking_fk_rotation_rad=float(cr_fk[0]))
                if args.mode=='simple':
                    from dataclasses import asdict
                    row['raw_solve']=asdict(root_action.solve_result)
                    row['ik_strict_pose_success']=bool(root_action.solve_result.finite and
                        root_action.solve_result.position_error_m<=1e-4 and
                        root_action.solve_result.orientation_error_rad<=1e-3)
                    row['velocity_bounded_ik']=args.velocity_bounded_ik
                    row['ik_command_accepted']=root_action.ik_command_accepted
                    row['ik_pose_budget_met']=root_action.solve_result.success
                    row['held_previous_accepted_ik']=not root_action.ik_command_accepted if args.arm_response_physics else not root_action.solve_result.success
                    row['ik_fallback_count']=root_action.fast_kin.fallbacks if root_action.fast_kin else None
            row.update(q_ik=qik,q_cmd=qcmd,ik_position_error_m=float(ap[0]),ik_rotation_error_rad=float(ar[0]),
                       ik_input_pos=ikpos[0],ik_input_quat=ikquat[0],
                       tracking_position_m=float(cp[0]),tracking_rotation_rad=float(cr[0]),
                       runtime_fk_position_m=float(fp[0]),runtime_fk_rotation_rad=float(fr[0]),
                       ik_failed=not (root_action.solve_result.success if args.mode=='simple' else bool(root_action.ik_success[0])),
                       command_rate_limited=root_action.rate_limited if args.mode=='simple' else None)
        if not all(np.isfinite(row[k]).all() for k in ('wrist_position_error_m','wrist_rotation_error_rad','hand_error_rad','object_keypoint_error_m')):raise ValueError('Nonfinite runtime measurement')
        f.write(json.dumps(row,default=serial)+'\n');f.flush();samples.append(row)
    env.scene.update=update
    original_reset=env._reset_idx
    def reset(ids):
        if counter['active']:
            ends.append(dict(episode=counter['episode'],state=snapshot(),termination={n:bool(env.termination_manager.get_term(n)[0]) for n in env.termination_manager.active_terms}))
        counter['active']=False
        original_reset(ids);counter['episode']+=1;counter['step']=0;verify_initial()
    env._reset_idx=reset
    timing=[]
    solve_timing=[]
    component_timing={}
    profiled=[]
    if args.realtime_view or args.recovery_capture:
        for label,obj,name in [('root_apply',root_action,'apply_actions'),('hand_apply',hand_action,'apply_actions'),
                               ('write',env.scene,'write_data_to_sim'),('physics_render',env.sim,'step'),
                               ('scene_update',env.scene,'update')]:
            original=getattr(obj,name)
            def timed(*a,_fn=original,_label=label,**kw):
                begin=time.perf_counter();result=_fn(*a,**kw)
                component_timing.setdefault(_label,[]).append(time.perf_counter()-begin)
                return result
            setattr(obj,name,timed);profiled.append((obj,name,original))
    if (args.realtime_view or args.recovery_capture) and args.mode=='simple':
        original_solve=root_action.solve
        def timed_solve(*a,**kw):
            started=time.perf_counter()
            result=original_solve(*a,**kw)
            solve_timing.append(time.perf_counter()-started)
            return result
        root_action.solve=timed_solve
    try:
        if args.stage=='recovery' and args.recovery_kind=='F1_pipeline':
            # Replay captured RAW actions through the original manager exactly
            # once; verify decoded/native commands against F0 offline. No policy.
            while len(ends)<args.episodes and app.is_running():
                source=sequences[counter['episode']][counter['step']]
                counter.update(active=True,command_frame=source['command_frame'],
                    command_time=counter['step']*env.physics_dt,action=source['action'])
                obs,_,_,_=wrapper.step(tensor([source['action']]))
        elif args.stage=='recovery':
            from tools.rb3_revo2_ik.recovery_replay import run_recovery
            run_recovery(env,root_action,hand_action,command,robot,source_meta,sequences,
                         counter,ends,snapshot,initial_records,app,out,args)
        elif args.stage=='actual':
            run_actual_motion(env,root_action,hand_action,command,robot,source_meta,sequences,
                              counter,ends,snapshot,initial_records,app,out)
        elif args.stage=='ab':
            if floating:raise ValueError('AB requires an arm mode')
            for frame in range(command.reference.frames):
                command.time_steps[:]=frame
                counter.update(active=True,command_frame=frame,command_time=frame*env.step_dt,action=np.zeros(12))
                env.action_manager.process_action(torch.zeros((1,12),device=env.device))
                for _ in range(env.cfg.decimation):
                    env._sim_step_counter+=1;env.action_manager.apply_action();env.scene.write_data_to_sim()
                    env.sim.step(render=False);env.scene.update(dt=env.physics_dt)
        else:
            wall_start=time.perf_counter()
            while len(ends)<args.episodes and app.is_running():
                # Existing history owner is called with update_history=False.
                started=time.perf_counter()
                if not args.realtime_view and not args.zero_actions:
                    same=env.observation_manager.compute_group('policy',update_history=False)
                    torch.testing.assert_close(same,obs['policy'],rtol=0,atol=0)
                inference_started=time.perf_counter()
                actions=torch.zeros((1,12),device=env.device) if args.zero_actions else adapter(obs)
                inference_elapsed=time.perf_counter()-inference_started
                if not args.realtime_view:
                    if not args.zero_actions:
                        with torch.inference_mode():torch.testing.assert_close(actions,policy(obs),rtol=0,atol=0)
                        observations_checked+=1
                    policy_records.append(dict(episode=counter['episode'],phase=int(command.time_steps[0]),
                        time_s=counter['step']*env.physics_dt,observation=array(obs['policy'])[0],action=array(actions)[0]))
                counter.update(active=True,command_frame=int(command.time_steps[0]),
                    command_time=counter['step']*env.physics_dt,action=array(actions)[0])
                obs,_,_,_=wrapper.step(actions)
                if args.realtime_view or args.recovery_capture:
                    work=time.perf_counter()-started
                    # Only wait when ahead: never drop steps or change physics dt.
                    remaining=env.step_dt-work
                    if args.realtime_view and remaining>0:time.sleep(remaining)
                    timing.append(dict(work_s=work,wall_s=time.perf_counter()-started,simulation_s=env.step_dt,
                                       inference_s=inference_elapsed))
                    if len(timing)%30==0:
                        recent=timing[-30:];wall=sum(r['wall_s'] for r in recent)
                        print(f'[live speed] sim/wall={sum(r["simulation_s"] for r in recent)/wall:.3f}x; IK mean={np.mean(solve_timing[-120:])*1000 if solve_timing else 0:.2f} ms',flush=True)
                        if args.ik_policy_rate:
                            print(f'[IK schedule] {root_action.tracking_ik_count} solves / {root_action.physics_apply_count} physics applies; policy={1/env.step_dt:g} Hz, physics={1/env.physics_dt:g} Hz (reset IK excluded)',flush=True)
        if adapter is not None:adapter.assert_frozen()
        meta=dict(mode=args.mode,stage=args.stage,checkpoint=str(Path(args.checkpoint).resolve()),
            realtime_view=args.realtime_view,
            zero_actions=args.zero_actions,
            fast_ik=args.fast_ik,
            ik_policy_rate=args.ik_policy_rate,
            velocity_bounded_ik=args.velocity_bounded_ik,
            ik_acceleration_limit=args.ik_acceleration_limit,
            ik_acceptance_mode='converged feasible best pose, budget violations reported' if args.ik_acceleration_limit else 'pose budget or hold',
            ik_pose_acceptance=dict(position_m=.005,orientation_rad=.05) if args.velocity_bounded_ik else dict(position_m=1e-4,orientation_rad=1e-3),
            ik_solver='velocity-box existing least-squares; fast IK reset only' if args.velocity_bounded_ik else 'unchanged',
            tracking_ik_count=root_action.tracking_ik_count if args.mode=='simple' and args.arm_response_physics else None,
            tracking_physics_applies=root_action.physics_apply_count if args.mode=='simple' else None,
            initial_native_settings=native_settings,
            record_video=args.record_video,video_fps=round(1/env.step_dt) if args.record_video else None,
            response_tau_s=args.response_tau,
            recovery_kind=args.recovery_kind if args.stage=='recovery' else None,
            recovery_no_can=args.recovery_no_can,recovery_speed=args.recovery_speed,
            recovery_holds=args.recovery_holds,arm_gains_key=args.arm_gains_key,
            arm_gains_file=args.arm_gains_file,
            arm_velocity_path=args.arm_velocity_path,
            arm_response_physics=args.arm_response_physics,
            transfer_config=args.transfer_config,
            recovery_runtime=probe.runtime if probe is not None else None,
            seed=cfg.seed,
            input_sha256={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in
                (args.states,bank_meta['reference'],str(ROOT/'tools/rb3_revo2_ik/rb3_model.json'),
                 args.arm_gains_file)},
            checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            reference=bank_meta['reference'],state_bank=str(Path(args.states).resolve()),
            initial_states=initial_records,ends=ends,observation_action_parity_checks=observations_checked,
            frozen_policy_verified=adapter is not None,policy_loaded=adapter is not None,
            actual_motion_source=str(Path(args.actual_source).resolve()) if args.stage in ('actual','recovery') else None,
            actual_motion_timing='recorded t_k sample commanded during (t_k-dt,t_k]; actual measured at t_k; offline endpoint ZOH, no time shift' if args.stage=='actual' else None,
            target_update_dt=env.physics_dt if args.stage=='actual' else env.step_dt,
            policy_target_update_dt=env.step_dt,
            ik_update_dt=(env.physics_dt if args.stage in ('actual','recovery') or
                (args.arm_response_physics and not args.ik_policy_rate) else env.step_dt) if args.mode=='simple' else None,
            joint_target_update_dt=env.physics_dt,
            env_origins=array(env.scene.env_origins),
            physics_dt=env.physics_dt,control_dt=env.step_dt,
            joint_names=robot.joint_names,arm_ids=arm_ids,hand_ids=hand_ids,
            gains=dict(kp=array(robot.data.joint_stiffness)[0],kd=array(robot.data.joint_damping)[0]),
            limits=dict(position=array(robot.data.soft_joint_pos_limits)[0],velocity=array(robot.data.joint_vel_limits)[0],effort=array(robot.data.joint_effort_limits)[0]),
            contact_filter=contact_paths,contact_bodies=sensor.body_names,
            gravity=env.cfg.sim.gravity,robot_spawn=robot.cfg.spawn.to_dict(),
            observation_terms=env.observation_manager.active_terms['policy'],
            observation_dimensions=env.observation_manager.group_obs_term_dim['policy'],
            arm_target_mode=('causal wrist response -> IK -> per-physics bounded position target; '
                +('velocity = final position backward difference / physics_dt' if args.arm_velocity_path else 'zero velocity target')) if args.mode=='simple' else 'unchanged',
            effort_provenance='No verified drive-only torque; saturation UNKNOWN',
            timestamp='command before first physics substep; actual state after scene.update; no post-hoc time shift')
        def encode(x):
            if callable(x):return x.__module__+'.'+x.__qualname__
            return serial(x)
        (out/'metadata.json').write_text(json.dumps(meta,default=encode,indent=2)+'\n')
        (out/'policy.json').write_text(json.dumps(policy_records,default=serial)+'\n')
        if args.realtime_view or (args.recovery_capture and args.stage=='grasp'):
            measured=dict(steps=timing,ik_s=solve_timing,simulation_s=len(timing)*env.step_dt,
                          component_s=component_timing,
                          wall_s=time.perf_counter()-wall_start,
                          ik_fallbacks=root_action.fast_kin.fallbacks if args.fast_ik else None,
                          physics_dt_unchanged=env.physics_dt,no_skipped_physics_steps=True)
            (out/'realtime.json').write_text(json.dumps(measured,indent=2)+'\n')
        if args.save_state_bank:
            from tools.rb3_revo2_ik.recovery_replay import validate_unique_placements
            validate_unique_placements(initial_records[:args.episodes])
            header=dict(bank_meta,seed=cfg.seed,source_capture=str(out.resolve()),
                        bank_kind='explicit_initial_states_only',full_state=False)
            records=[header]
            for s in initial_records:
                records.append(dict(event='initialization',episode=s['episode'],reference_frame=s['phase'],
                    placement_offset=s['placement_offset'],actual_base_pos=s['wrist_pos'],
                    actual_base_quat_xyzw=s['wrist_quat'],all_joint_pos=s['all_q'],all_joint_vel=s['all_v'],
                    robot_root_state=s['robot_root'],object_root_state=s['object_state'],
                    applied_arm_target=s['applied_arm_target']))
            with Path(args.save_state_bank).open('x') as bank_out:
                for record in records:bank_out.write(json.dumps(record,default=encode)+'\n')
        print('[interface complete]',args.mode,args.stage,'episodes',len(ends),'samples',len(samples),flush=True)
    finally:
        for obj,name,original in reversed(profiled):setattr(obj,name,original)
        counter['active']=False;env.scene.update=original_update;env._reset_idx=original_reset
        if probe is not None:probe.close()
        f.close();wrapper.close()


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
