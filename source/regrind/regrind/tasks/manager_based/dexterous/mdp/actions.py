from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_conjugate, quat_from_angle_axis, quat_mul
from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg

from regrind.utils.math import quat_from_euler_xyz_intrinsic

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ClippedRelativeJointPositionAction(joint_actions.RelativeJointPositionAction):
    """Relative joint position action with raw-action clipping.

    Intended for **finger** DOFs only; floating-hand root pose is controlled by ``SE3ImpedanceActionTerm``.
    ``hand_joint_pos`` observation and ``delayed_hand_joint_pos`` match this action dimension.
    """

    cfg: "ClippedRelativeJointPositionActionCfg"

    def __init__(self, cfg: "ClippedRelativeJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # set joint limits
        joint_pos_limits = self._asset.root_physx_view.get_dof_limits().to(self.device)  # (num_envs, num_joints, 2)
        self._dof_lower_limits = joint_pos_limits[:, self._joint_ids, 0]
        self._dof_upper_limits = joint_pos_limits[:, self._joint_ids, 1]

        self._offset = torch.zeros_like(self._asset.data.joint_pos[:, self._joint_ids], device=self.device)

    def process_actions(self, actions: torch.Tensor):
        if self.cfg.raw_clip is not None:
            actions = torch.clamp(actions, min=self.cfg.raw_clip[0], max=self.cfg.raw_clip[1])
        super().process_actions(actions)

    def get_base_joint_pos(self) -> torch.Tensor:
        """Finger joint positions used as the base before adding policy deltas."""
        if self.cfg.base_action_source == "motion_target":
            command = self._env.command_manager.get_term(self.cfg.command_name)
            base_pos = command.target_hand_joint_pos
            if base_pos.shape[-1] != self.processed_actions.shape[-1]:
                raise RuntimeError(
                    f"Motion target_hand_joint_pos last dim {base_pos.shape[-1]} != action dim "
                    f"{self.processed_actions.shape[-1]}; use joint_names aligned with motion "
                    "actuated_joint_names (e.g. ACTUATED_JOINT_NAMES)."
                )
            return base_pos
        if self.cfg.base_action_source == "current_obs":
            # Finger joints only; matches :func:`hand_joint_pos` / ``delayed_hand_joint_pos``.
            if hasattr(self._env, "delayed_hand_joint_pos"):
                return self._env.delayed_hand_joint_pos[:, :]
            return self._asset.data.joint_pos[:, self._joint_ids]  # TODO: consistency with observations
        if self.cfg.base_action_source == "zero":
            return torch.zeros_like(self.processed_actions)
        raise ValueError(f"Invalid base_action_source: {self.cfg.base_action_source}")

    def apply_actions(self):
        current_actions = self.processed_actions + self.get_base_joint_pos()
        # clip the actions to the joint limits
        current_actions = torch.clamp(current_actions, min=self._dof_lower_limits, max=self._dof_upper_limits)
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


@configclass
class ClippedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Config for raw-clipped relative joint position action."""

    class_type: type[ActionTerm] = ClippedRelativeJointPositionAction
    raw_clip: tuple[float, float] | None = (-1.0, 1.0)
    # wrist_pos_lower_bound: tuple[float, float, float] | None = None  # (-0.25, -0.25, 0.0)
    # wrist_pos_upper_bound: tuple[float, float, float] | None = None  # (0.25, 0.25, 1.5)
    base_action_source: Literal["current_obs", "motion_target", "zero"] = "motion_target"
    #: Command term name when ``base_action_source`` is ``motion_target`` (e.g. ``"motion"``).
    command_name: str = "motion"


def _quat_positive_real(q: torch.Tensor) -> torch.Tensor:
    """Map quaternion to the hemisphere with nonnegative real part (w)."""
    return torch.where(q[..., 0:1] < 0, -q, q)


def _rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors (axis-angle, radians) to quaternions (w, x, y, z)."""
    angle = torch.norm(rotvec, dim=-1)
    axis = rotvec / angle.unsqueeze(-1).clamp(min=1e-8)
    return quat_from_angle_axis(angle, axis)


class SE3ImpedanceActionTerm(ActionTerm):
    """SE(3) Cartesian impedance: PD wrench on the asset root from pose targets."""

    cfg: SE3ImpedanceActionCfg

    def __init__(self, cfg: SE3ImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.kp_pos = cfg.kp_pos
        self.kd_pos = cfg.kd_pos
        self.kp_rot = cfg.kp_rot
        self.kd_rot = cfg.kd_rot
        if cfg.body_name is None:
            self._wrench_body_ids: list[int] = [0]
        else:
            body_ids, body_names = self._asset.find_bodies(cfg.body_name)
            if len(body_ids) != 1:
                raise ValueError(
                    f"body_name {cfg.body_name!r} must match exactly one body; "
                    f"got {len(body_ids)}: {body_names}"
                )
            self._wrench_body_ids = [body_ids[0]]
        self.target_pos: torch.Tensor | None = None
        self.target_quat: torch.Tensor | None = None
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Cache masses once (randomized at startup, not per-reset).
        # TODO: get domain randomized masses
        self._body_mass = self._asset.data.default_mass.to(self.device)  # (num_envs, num_bodies)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def get_base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Base pose: env-relative position (matches motion command), world quaternion wxyz."""
        return self._get_base_pose()

    def _get_base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Base pose: env-relative position (matches motion command), world quaternion wxyz."""
        if self.cfg.base_action_source == "motion_target":
            command = self._env.command_manager.get_term(self.cfg.command_name)
            base_pos = command.target_hand_wrist_pos
            wrist_rot = command.target_hand_wrist_rot
            base_quat = quat_from_euler_xyz_intrinsic(wrist_rot[:, 0], wrist_rot[:, 1], wrist_rot[:, 2])
            return base_pos, base_quat

        pos = self._asset.data.root_pos_w - self._env.scene.env_origins
        quat = self._asset.data.root_quat_w
        if self.cfg.base_action_source == "current_obs":
            if hasattr(self._env, "delayed_hand_wrist_pos"):
                base_pos = self._env.delayed_hand_wrist_pos
            else:
                base_pos = pos
            if hasattr(self._env, "delayed_hand_wrist_quat"):
                base_quat = self._env.delayed_hand_wrist_quat
            else:
                base_quat = quat
            return base_pos, base_quat  # (num_envs, 3) and (num_envs, 4)
        if self.cfg.base_action_source == "zero":
            n, device, dtype = pos.shape[0], pos.device, pos.dtype
            base_pos = torch.zeros(n, 3, device=device, dtype=dtype)
            base_quat = torch.zeros(n, 4, device=device, dtype=dtype)
            base_quat[:, 0] = 1.0
            return base_pos, base_quat
        raise ValueError(f"Invalid base_action_source: {self.cfg.base_action_source}")
    
    def _com_pos_w(self) -> torch.Tensor:
        body_com_w = self._asset.data.body_com_pos_w          # (num_envs, num_bodies, 3)
        total_mass = self._body_mass.sum(dim=-1, keepdim=True) # (num_envs, 1)
        com_w = (self._body_mass.unsqueeze(-1) * body_com_w).sum(dim=1) / total_mass  # (num_envs, 3)
        return com_w

    def _com_pos_b(self) -> torch.Tensor:
        """Mass-weighted CoM of the full articulation in the root link frame. Shape (num_envs, 3)."""
        com_w = self._com_pos_w()

        root_pos_w = self._asset.data.root_link_pos_w
        root_quat_w = self._asset.data.root_link_quat_w
        return quat_apply_inverse(root_quat_w, com_w - root_pos_w)
    
    def _root_com_pos_b(self) -> torch.Tensor:
        """Root CoM position in the root link frame. Shape (num_envs, 3)."""
        root_pos_w = self._asset.data.root_link_pos_w
        root_quat_w = self._asset.data.root_link_quat_w
        return quat_apply_inverse(root_quat_w, self._asset.data.root_com_pos_w - root_pos_w)

    def process_actions(self, actions: torch.Tensor):
        """Called once per policy step. Compute targets."""

        if self.cfg.raw_clip is not None:
            actions = torch.clamp(actions, min=self.cfg.raw_clip[0], max=self.cfg.raw_clip[1])
        self._raw_actions[:] = actions
        delta_pos = actions[:, :3] * self.cfg.scale_pos
        delta_rot = actions[:, 3:6] * self.cfg.scale_rot
        self._processed_actions[:, :3] = delta_pos
        self._processed_actions[:, 3:6] = delta_rot

        base_pos, base_quat = self._get_base_pose()

        self.target_pos = base_pos + delta_pos
        delta_quat = _rotvec_to_quat(delta_rot)
        self.target_quat = quat_mul(delta_quat, base_quat)

    def apply_actions(self):
        """Called every sim step. PD runs here."""
        root_state = self._asset.data.root_state_w
        pos = self._asset.data.root_pos_w - self._env.scene.env_origins
        quat = self._asset.data.root_quat_w
        vel = root_state[:, 7:10]
        omega = root_state[:, 10:13]

        assert self.target_pos is not None and self.target_quat is not None

        f = self.kp_pos * (self.target_pos - pos) - self.kd_pos * vel

        q_err = quat_mul(self.target_quat, quat_conjugate(quat))
        q_err = _quat_positive_real(q_err)
        rot_error = axis_angle_from_quat(q_err)

        tau = self.kp_rot * rot_error - self.kd_rot * omega

        # World-frame wrench (same as PD above). Applied on write_data_to_sim.
        self._asset.set_external_force_and_torque(
            forces=f.unsqueeze(1),
            torques=tau.unsqueeze(1),
            positions=None,
            body_ids=self._wrench_body_ids,
            is_global=True,
        )


@configclass
class SE3ImpedanceActionCfg(ActionTermCfg):
    """Config for :class:`SE3ImpedanceActionTerm` (6D delta: position + axis-angle)."""

    class_type: type[ActionTerm] = SE3ImpedanceActionTerm

    raw_clip: tuple[float, float] | None = (-1.0, 1.0)
    scale_pos: float = 1.0
    scale_rot: float = 1.0

    #: World-frame force PD on position error (N/m) and damping (N·s/m). Leap tasks override in ``LeapHandBaseEnvCfg``.
    kp_pos: float = 300.0
    kd_pos: float = 30.0
    #: World-frame torque PD on axis-angle error (N·m/rad) and damping (N·m·s/rad).
    kp_rot: float = 300.0
    kd_rot: float = 30.0
    #: Base pose before adding the policy delta: sim root pose, motion demo wrist pose, or world origin + identity.
    base_action_source: Literal["current_obs", "motion_target", "zero"] = "motion_target"
    #: Command term name when ``base_action_source`` is ``motion_target`` (e.g. ``"motion"``).
    command_name: str = "motion"
    #: Link to apply the external wrench to (regex, must match a single body name). ``None`` = articulation root (index 0).
    body_name: str | None = None

    # position_lower_bound: tuple[float, float, float] | None = None  # (-0.25, -0.25, 0.0)
    # position_upper_bound: tuple[float, float, float] | None = None  # (0.25, 0.25, 1.5)
