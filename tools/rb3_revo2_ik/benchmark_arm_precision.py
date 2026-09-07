"""Isolated precision test of the existing physical assembled robot. No policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark',default=str(ROOT/'config/experiments/rb3_precision_benchmark.json'))
parser.add_argument('--candidates',default=str(ROOT/'config/experiments/rb3_precision_candidates.json'))
parser.add_argument('--candidate',default='baseline')
parser.add_argument('--held-out',action='store_true')
parser.add_argument('--single-joints',action='store_true',help='Six fixed 0.05 rad isolated quintics, 1 s each way')
parser.add_argument('--move-can-away',action='store_true',help='Move only the can +10 m at initialization; preserve collisions')
parser.add_argument('--command-mode',choices=('analytical','linear_zero','smooth_pv'),default='analytical')
parser.add_argument('--output',required=True)
parser.add_argument('--headless',dest='legacy_headless',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.headless=args.legacy_headless
del args.legacy_headless
output=Path(args.output)
if output.exists():raise FileExistsError(output)
config=json.loads(Path(args.benchmark).read_text());gains=json.loads(Path(args.candidates).read_text())[args.candidate]
app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
import regrind.tasks
from tools.rb3_revo2_ik.precision_trajectory import build,command_rows
from tools.rb3_revo2_ik.trace_arm_execution import array,serial
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.arm_diagnostics.analyze_arm_execution import pose_errors


def main():
    global config
    if args.single_joints:
        from tools.rb3_revo2_ik.precision_trajectory import single_joint_config
        config=single_joint_config(config)
    q0,trajectory,phases=build(config,args.held_out)
    cfg=parse_env_cfg('Regrind-RB3-Revo2-TunaCan-Online-Play-v0',device=args.device,num_envs=1,use_fabric=True)
    cfg.commands.reference.trajectory_path=str(ROOT/config['reference'])
    # Contact reporting is measurement only; no collision filtering changes.
    cfg.scene.robot.spawn.activate_contact_sensors=True
    cfg.scene.precision_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.0)
    arm=cfg.scene.robot.actuators['rb3_arm']
    arm.stiffness=dict(zip(config['joint_names'],gains['kp']))
    arm.damping=dict(zip(config['joint_names'],gains['kd']))
    env=gym.make('Regrind-RB3-Revo2-TunaCan-Online-Play-v0',cfg=cfg).unwrapped
    env.reset()
    command=env.command_manager.get_term('reference');robot=command.robot
    if args.move_can_away:
        pose=command.object.data.root_link_pose_w.torch.clone();pose[:,0]+=10.
        command.object.write_root_link_pose_to_sim_index(root_pose=pose)
    if env.event_manager.active_terms or env.physics_dt!=config['physics_dt']:raise ValueError('Unexpected events/dt')
    original_trajectory=trajectory
    trajectory=command_rows(trajectory,q0,args.command_mode,cfg.decimation)
    ids=[robot.joint_names.index(n) for n in config['joint_names']]
    bids=[robot.backend_joint_names.index(n) for n in config['joint_names']]
    sensor=env.scene.sensors['precision_contacts']
    kin=RB3730Kinematics(base_position=config['base_position'],base_quaternion_xyzw=config['base_quaternion_xyzw'])
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=env.device)
    targets=torch.full((1,robot.num_joints),config['hand_position_target_rad'],device=env.device)
    targets[0,ids]=tensor(q0)
    # Exactly one explicit benchmark initialization, before timed physics.
    robot.write_joint_state_to_sim(targets,torch.zeros_like(targets))
    robot.set_joint_position_target_index(target=targets)
    robot.set_joint_velocity_target_index(target=torch.zeros_like(targets))
    robot.set_joint_effort_target_index(target=torch.zeros_like(targets))
    env.scene.write_data_to_sim();env.sim.forward()
    native=robot.root_view
    contact_paths=sensor.body_physx_view.prim_paths
    link_paths=native.link_paths[0]
    if set(contact_paths)!=set(link_paths):
        raise ValueError('Contact reporter does not cover every actual articulation link')
    runtime={key:array(getattr(native,method)())[0] for key,method in {
        'kp':'get_dof_stiffnesses','kd':'get_dof_dampings','effort':'get_dof_max_forces',
        'velocity':'get_dof_max_velocities','position_limits':'get_dof_limits','drive_type':'get_drive_types',
        'mass':'get_masses','inertia':'get_inertias','gravity_disabled':'get_disable_gravities'}.items()}
    np.testing.assert_allclose(runtime['kp'][bids],gains['kp']);np.testing.assert_allclose(runtime['kd'][bids],gains['kd'])
    np.testing.assert_array_equal(runtime['effort'][bids],[10,100,100,100,100,10])
    np.testing.assert_array_equal(runtime['velocity'][bids],[10]*6)
    np.testing.assert_array_equal(runtime['drive_type'][bids],[1]*6)
    qref=np.array([r[1] for r in trajectory]);vref=np.array([r[2] for r in trajectory])
    bounds=runtime['position_limits'][bids]
    if np.any(qref<bounds[:,0]) or np.any(qref>bounds[:,1]) or np.any(abs(vref)>runtime['velocity'][bids]):raise ValueError('Benchmark exceeds runtime limits')
    refpos,refquat=kin.forward_batch(qref)
    original_pos,original_quat=kin.forward_batch(np.array([r[1] for r in original_trajectory]))
    metadata=dict(candidate=args.candidate,held_out=args.held_out,gains=gains,benchmark=config,
        benchmark_sha256=hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest(),phases=phases,
        user_joint_names=robot.joint_names,backend_joint_names=robot.backend_joint_names,arm_ids=ids,backend_arm_ids=bids,
        body_names=robot.body_names,runtime=runtime,physics=cfg.sim.to_dict(),robot_spawn=cfg.scene.robot.spawn.to_dict(),
        initial_q=array(robot.data.joint_pos)[0],initial_v=array(robot.data.joint_vel)[0],
        initial_root=array(robot.data.root_state_w)[0],initial_object=array(command.object.data.root_state_w)[0],
        contact_bodies=sensor.body_names,contact_body_paths=contact_paths,articulation_link_paths=link_paths,
        compensation='none',drive_saturation='UNKNOWN',
        single_joints=args.single_joints,can_moved_away=args.move_can_away,
        command_mode=args.command_mode,command_substeps=cfg.decimation,
        a_target_definition=('Unused zero placeholder; linear corners have undefined instantaneous acceleration. See comparison backward differences.'
                             if args.command_mode=='linear_zero' else 'Analytical path acceleration, not submitted to simulator.'),
        known_command_delay_s=0. if args.command_mode=='analytical' else env.step_dt,
        original_target_definition='Unshifted frozen analytical trajectory; separately measured, never fitted to actual.',
        timestamp='Command is analytical endpoint q(t_k),dq(t_k), submitted at t_k-dt; measured after physics at t_k. No time shift.',
        state_writes='One benchmark initialization only; no per-step state writes or manager/policy/IK calls.')
    output.mkdir(parents=True,exist_ok=False)
    def config_serial(value):
        if callable(value):return value.__module__+'.'+value.__qualname__
        return serial(value)
    (output/'metadata.json').write_text(json.dumps(metadata,default=config_serial,indent=2)+'\n')
    saved=[]
    sent={};restores=[]
    for key,name in [('q','set_dof_position_targets'),('v','set_dof_velocity_targets'),('ff','set_dof_actuation_forces')]:
        original=getattr(native,name)
        def observe(data,*a,_key=key,_fn=original,**kw):
            sent[_key]=array(data)[0].copy()
            return _fn(data,*a,**kw)
        setattr(native,name,observe);restores.append((name,original))
    try:
        with (output/'trace.jsonl').open('x') as stream:
            for i,(t,q,v,acc,phase) in enumerate(trajectory):
                qpre=array(robot.data.joint_pos)[0,ids];vpre=array(robot.data.joint_vel)[0,ids]
                targets[0,ids]=tensor(q)
                velocities=torch.zeros_like(targets);velocities[0,ids]=tensor(v)
                robot.set_joint_position_target_index(target=targets)
                robot.set_joint_velocity_target_index(target=velocities)
                robot.set_joint_effort_target_index(target=torch.zeros_like(targets))
                env._sim_step_counter+=1;env.scene.write_data_to_sim()
                np.testing.assert_array_equal(sent['q'][bids],array(targets)[0,ids])
                np.testing.assert_array_equal(sent['v'][bids],array(velocities)[0,ids])
                np.testing.assert_array_equal(sent['ff'],np.zeros(robot.num_joints))
                env.sim.step(render=False);env.scene.update(dt=env.physics_dt)
                pos=array(command.current_hand_wrist_pos)[0]+array(env.scene.env_origins)[0]
                quat=array(command.current_hand_wrist_quat)[0]
                pe,re=pose_errors(refpos[i:i+1],refquat[i:i+1],pos[None],quat[None])
                ope,ore=pose_errors(original_pos[i:i+1],original_quat[i:i+1],pos[None],quat[None])
                dpe,dre=pose_errors(original_pos[i:i+1],original_quat[i:i+1],refpos[i:i+1],refquat[i:i+1])
                row=dict(step=i+1,time_s=t,command_time_s=t-env.physics_dt,phase=phase,
                    q_target=q,v_target=v,a_target=acc,q_pre=qpre,v_pre=vpre,
                    q_actual=array(robot.data.joint_pos)[0,ids],v_actual=array(robot.data.joint_vel)[0,ids],
                    all_q_actual=array(robot.data.joint_pos)[0],target_pos=refpos[i],target_quat_xyzw=refquat[i],
                    actual_pos=pos,actual_quat_xyzw=quat,position_error_m=float(pe[0]),rotation_error_rad=float(re[0]),
                    original_q=original_trajectory[i][1],original_pos=original_pos[i],original_quat_xyzw=original_quat[i],
                    original_position_error_m=float(ope[0]),original_rotation_error_rad=float(ore[0]),
                    command_deviation_position_m=float(dpe[0]),command_deviation_rotation_rad=float(dre[0]),
                    contact_force_N=array(sensor.data.net_forces_w)[0],
                    submitted_velocity=sent['v'][bids],drive_saturation='UNKNOWN')
                stream.write(json.dumps(row,default=serial)+'\n');saved.append(row)
                if i%600==0:print(f'[precision] {args.candidate} {t:.2f}s error={pe[0]*1000:.3f}mm',flush=True)
        from tools.arm_diagnostics.analyze_arm_precision import summarize
        summary=summarize(metadata,saved,output)
        print('[precision complete]',json.dumps(summary['headline']),flush=True)
    finally:
        for name,fn in restores:setattr(native,name,fn)
        env.close()

try:main()
finally:app.close()
