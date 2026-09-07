"""Opt-in mounted bridge. Reuse floating pose decoding, not its force application."""
import math
import torch
from isaaclab.utils.configclass import configclass
from regrind.data.rb3_revo2_reference import RB3_JOINT_NAMES
from regrind.tasks.manager_based.dexterous.mdp.actions import SE3ImpedanceActionTerm, SE3ImpedanceActionCfg
from regrind.tasks.manager_based.dexterous.mdp.rb3_revo2_actions import _load_rb3_kinematics_class
from regrind.workcell import ROBOT_MOUNT_POSITION, ROBOT_MOUNT_QUATERNION_XYZW
from regrind.tasks.manager_based.dexterous.mdp.actions import (
    quat_mul, quat_conjugate, axis_angle_from_quat, _quat_positive_real, _rotvec_to_quat,
)


def response_step(position, quaternion, target_position, target_quaternion, dt, tau):
    """Causal first-order command shaping, not a measured/virtual robot state.

    XYZW; rotations follow the shortest world-axis rotation. One ZOH update at
    the caller's configured boundary (policy or physics, never both).
    tau=0 preserves the unmodified decoder target exactly.
    This overdamped approximation does NOT reproduce contact/inertia dynamics.
    """
    if not math.isfinite(tau) or tau < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid response time or control timestep')
    if tau == 0:
        return target_position.clone(), target_quaternion.clone()
    alpha = -math.expm1(-dt / tau)
    error = _quat_positive_real(quat_mul(target_quaternion, quat_conjugate(quaternion)))
    rotation = _rotvec_to_quat(alpha * axis_angle_from_quat(error))
    return position + alpha * (target_position - position), quat_mul(rotation, quaternion)


class SimpleMountedWrist(SE3ImpedanceActionTerm):
    """Same floating decoder and bounded pose IK; zero dq by default.

    IK forward/inverse already includes link6 -> mount -> right_hand_base_link.
    Positions are env-local (subtract env origin); rotations remain world XYZW.
    Never apply the inherited floating wrench to the mounted articulation.
    """
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if self.num_envs!=1:raise ValueError('Minimal interface is validated for num_envs=1 only')
        self.ids=[self._asset.joint_names.index(n) for n in RB3_JOINT_NAMES]
        self.kin=_load_rb3_kinematics_class()(base_position=ROBOT_MOUNT_POSITION,
            base_quaternion_xyzw=ROBOT_MOUNT_QUATERNION_XYZW)
        if self.kin.mounted_wrist_frame.rsplit('/',1)[-1]!=cfg.body_name:
            raise ValueError('IK endpoint must be the policy wrist; do not apply mount offset twice')
        if tuple(self.kin.joint_names)!=tuple(RB3_JOINT_NAMES):raise ValueError('Arm order mismatch')
        self.fast_kin=None
        if cfg.fast_ik:
            from tools.rb3_revo2_ik.warm_start_ik import WarmStartIK
            self.fast_kin=WarmStartIK(self.kin)
        self.goal=torch.zeros((1,6),device=self.device)
        self.applied=self.goal.clone();self.solve_result=None;self.rate_limited=False
        if not math.isfinite(cfg.response_tau) or cfg.response_tau < 0:
            raise ValueError('response_tau must be finite and nonnegative')
        self.ik_target_pos=None;self.ik_target_quat=None
        self.velocity_path=cfg.velocity_path
        self.response_at_physics=cfg.response_at_physics
        if self.response_at_physics and cfg.response_tau<=0:
            raise ValueError('Physics-rate response requires positive response_tau')
        self.limits=self._asset.data.soft_joint_pos_limits.torch[:,self.ids]
        self.speed_limit=self._asset.data.joint_vel_limits.torch[:,self.ids]
        if not torch.isfinite(self.speed_limit).all() or (self.speed_limit<=0).any():raise ValueError('Invalid speed limits')

    @property
    def applied_joint_target(self):return self.applied

    def solve(self,pos,quat,previous):
        actual=self._asset.data.joint_pos.torch[0,self.ids].detach().cpu().numpy()
        solver=self.fast_kin or self.kin
        return solver.inverse(pos.detach().cpu().numpy(),quat.detach().cpu().numpy(),
            initial_q=previous.detach().cpu().numpy(),neutral_q=actual,max_nfev=300)

    def process_actions(self,actions):
        if actions.shape!=(1,6) or not torch.isfinite(actions).all():raise ValueError('Nonfinite/invalid wrist action')
        # Single source of truth for clip -> scale -> reference + residual.
        super().process_actions(actions)
        if self.response_at_physics:
            return  # The raw equilibrium is held; shaping/IK runs at physics rate.
        self.ik_target_pos,self.ik_target_quat=response_step(
            self.ik_target_pos,self.ik_target_quat,self.target_pos,self.target_quat,
            self._env.step_dt,self.cfg.response_tau)
        self.solve_result=self.solve(self.ik_target_pos[0],self.ik_target_quat[0],self.goal[0])
        r=self.solve_result
        if r.success and r.finite and not r.joint_limit_violation:
            self.goal[0]=torch.as_tensor(r.q,device=self.device,dtype=self.goal.dtype)
        # Otherwise keep the previous accepted target, explicitly logged by runner.

    def apply_actions(self):
        if getattr(self,'response_at_physics',False):
            self.ik_target_pos,self.ik_target_quat=response_step(
                self.ik_target_pos,self.ik_target_quat,self.target_pos,self.target_quat,
                self._env.physics_dt,self.cfg.response_tau)
            r=self.solve(self.ik_target_pos[0],self.ik_target_quat[0],self.goal[0])
            self.solve_result=r
            if r.success and r.finite and not r.joint_limit_violation:
                self.goal[0]=torch.as_tensor(r.q,device=self.device,dtype=self.goal.dtype)
        # Bound raw coordinate changes, never wrap angles. Velocity is opt-in.
        limit=self.speed_limit*self._env.physics_dt
        delta=self.goal-self.applied
        bounded=torch.clamp(delta,min=-limit,max=limit)
        self.rate_limited=bool((bounded!=delta).any())
        previous=self.applied.clone()
        self.applied.copy_(torch.clamp(self.applied+bounded,min=self.limits[...,0],max=self.limits[...,1]))
        self._asset.set_joint_position_target_index(target=self.applied,joint_ids=self.ids)
        velocity=(self.applied-previous)/self._env.physics_dt if getattr(self,'velocity_path',False) else torch.zeros_like(self.applied)
        self._asset.set_joint_velocity_target_index(target=torch.clamp(velocity,min=-self.speed_limit,max=self.speed_limit),joint_ids=self.ids)

    def reset_from_reference(self,env_ids):
        # Called by the existing command reset AFTER placement/phase selection.
        command=self._env.command_manager.get_term(self.cfg.command_name)
        actual=self._asset.data.joint_pos.torch[0,self.ids]
        r=self.solve(command.target_hand_wrist_pos[0],command.target_hand_wrist_quat[0],actual)
        if not (r.success and r.finite and not r.joint_limit_violation):raise RuntimeError('Reset IK failed')
        self.goal[0]=torch.as_tensor(r.q,device=self.device,dtype=self.goal.dtype)
        self.applied.copy_(self.goal);self.solve_result=None;self.rate_limited=False
        self._asset.write_joint_state_to_sim(self.goal,torch.zeros_like(self.goal),joint_ids=self.ids,env_ids=env_ids)
        self._asset.set_joint_position_target_index(target=self.goal,joint_ids=self.ids,env_ids=env_ids)
        self._asset.set_joint_velocity_target_index(target=torch.zeros_like(self.goal),joint_ids=self.ids,env_ids=env_ids)
        self.target_pos=command.target_hand_wrist_pos.clone();self.target_quat=command.target_hand_wrist_quat.clone()
        # No previous episode's command may leak into this response history.
        self.ik_target_pos=self.target_pos.clone();self.ik_target_quat=self.target_quat.clone()


@configclass
class SimpleMountedWristCfg(SE3ImpedanceActionCfg):
    class_type:type=SimpleMountedWrist
    body_name:str='right_hand_base_link'
    command_name:str='reference'
    base_action_source:str='motion_target'
    response_tau:float=0.0  # Experimental only; normal behavior is unchanged.
    velocity_path:bool=False  # Diagnostic feedforward; never enabled implicitly.
    response_at_physics:bool=False
    fast_ik:bool=False  # Existing multi-seed selection remains the default.
