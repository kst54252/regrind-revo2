"""Opt-in pre-command/native-setter instrumentation. No control changes."""
import importlib.metadata
import numpy as np
from tools.rb3_revo2_ik.trace_arm_execution import array


class RecoveryProbe:
    def __init__(self,env,counter,snapshot,root_action):
        self.env,self.counter,self.snapshot,self.root=env,counter,snapshot,root_action
        self.robot=env.scene['robot'];self.view=self.robot.root_view
        self.bids=[self.robot.backend_joint_names.index(n) for n in self.robot.joint_names]
        self.restores=[];self.calls={};self.last=None
        self.runtime={k:array(getattr(self.view,method)())[0,self.bids] for k,method in
            [('drive_type','get_drive_types'),('kp','get_dof_stiffnesses'),('kd','get_dof_dampings'),
             ('effort_limit','get_dof_max_forces'),('velocity_limit','get_dof_max_velocities'),
             ('position_limit','get_dof_limits')]}
        self.runtime['disable_gravity']=array(self.view.get_disable_gravities())[0]
        self.runtime['actuators']={k:dict(class_name=type(v).__module__+'.'+type(v).__name__,
            implicit=bool(v.is_implicit_model)) for k,v in self.robot.actuators.items()}
        self.runtime['versions']={k:importlib.metadata.version(k) for k in ('isaacsim','isaaclab','isaaclab_physx')}
        self.runtime['wrist_controller']={k:getattr(root_action,k,None) for k in ('kp_pos','kd_pos','kp_rot','kd_rot')}
        for key,method in [('position','set_dof_position_targets'),('velocity','set_dof_velocity_targets'),
                           ('feedforward','set_dof_actuation_forces')]:
            original=getattr(self.view,method)
            def wrapped(data,*a,_key=key,_fn=original,**kw):
                if self.counter['active']:self.calls.setdefault(_key,[]).append(array(data)[0,self.bids])
                return _fn(data,*a,**kw)
            setattr(self.view,method,wrapped);self.restores.append((self.view,method,original))
        original=env.scene.write_data_to_sim
        def send():
            if not counter['active']:return original()
            self.calls={};pre=snapshot()
            staged={k:array(getattr(self.robot.data,attr))[0] for k,attr in
                [('position','joint_pos_target'),('velocity','joint_vel_target'),('feedforward','joint_effort_target')]}
            result=original()
            for key in staged:
                if len(self.calls.get(key,[]))!=1:raise RuntimeError(f'Unexpected native {key} overwrite/count')
                np.testing.assert_allclose(staged[key],self.calls[key][0],rtol=0,atol=0)
            self.last=dict(pre_state=pre,position=self.calls['position'][0],velocity=self.calls['velocity'][0],
                feedforward=self.calls['feedforward'][0],wrist_equilibrium_pos=array(root_action.target_pos)[0],
                wrist_equilibrium_quat=array(root_action.target_quat)[0],
                command_time_s=counter['step']*env.physics_dt,state_time_s=(counter['step']+1)*env.physics_dt,
                overwrite_detected=False,feedforward_provenance='explicit native actuation effort, NOT implicit drive torque')
            return result
        env.scene.write_data_to_sim=send;self.restores.append((env.scene,'write_data_to_sim',original))

    def close(self):
        for obj,name,method in reversed(self.restores):setattr(obj,name,method)
