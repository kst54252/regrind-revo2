"""Opt-in pose IK inside the next physical command's reachable joint box.

Reuses RB3730Kinematics.inverse and the verified analytic Jacobian. No new
model, actuator limits, time scaling, angle wrapping, or dynamics controller.
The near-singularity pose approximation is explicit, not strict IK success.
"""
import copy
import numpy as np
from tools.rb3_revo2_ik.warm_start_ik import PoseJacobian


class VelocityBoundedIK:
    def __init__(self,reference):
        self.reference=reference
        self.calls=0

    def inverse(self,position,quaternion,*,command_q,velocity_limit,dt,
                position_lower,position_upper,position_tolerance_m=.005,
                orientation_tolerance_rad=.05,max_nfev=300,previous_velocity=None,acceleration_limit=0.):
        center=np.asarray(command_q,dtype=float)
        speed=np.asarray(velocity_limit,dtype=float)
        if center.shape!=(6,) or speed.shape!=(6,) or not np.isfinite(center).all() or not np.isfinite(speed).all():
            raise ValueError('Expected finite six-joint command and speed limit')
        if not np.isfinite(dt) or dt<=0 or np.any(speed<=0):raise ValueError('Invalid timestep/speed')
        model=self.reference
        lower=np.maximum(model.joint_lower,np.asarray(position_lower))
        upper=np.minimum(model.joint_upper,np.asarray(position_upper))
        if np.any(center<lower-1e-6) or np.any(center>upper+1e-6):
            raise ValueError('Previous command outside position limits')
        center=np.clip(center,lower,upper)
        local=copy.copy(model)
        local.joint_lower=np.maximum(lower,center-speed*dt)
        local.joint_upper=np.minimum(upper,center+speed*dt)
        if not np.isfinite(acceleration_limit) or acceleration_limit<0:raise ValueError('Invalid command acceleration bound')
        if acceleration_limit:
            velocity=np.asarray(previous_velocity,dtype=float)
            if velocity.shape!=(6,) or not np.isfinite(velocity).all():raise ValueError('Invalid previous command velocity')
            local.joint_lower=np.maximum(local.joint_lower,center+(velocity-acceleration_limit*dt)*dt)
            local.joint_upper=np.minimum(local.joint_upper,center+(velocity+acceleration_limit*dt)*dt)
        if np.any(local.joint_lower>=local.joint_upper):raise ValueError('Empty velocity-bounded IK box')
        seed=np.clip(center,local.joint_lower,local.joint_upper)
        local._candidate_seeds=lambda warm,neutral:[seed.copy()]
        differential=PoseJacobian(model)
        local._residual=differential.residual
        self.calls+=1
        return local.inverse(position,quaternion,initial_q=center,neutral_q=center,
            jacobian=differential.jacobian,max_nfev=max_nfev,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad)
