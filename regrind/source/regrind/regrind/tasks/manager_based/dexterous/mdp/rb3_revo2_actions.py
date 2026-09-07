"""Residual action term for the 12 controllable RB3+Revo2 joints."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_mul

from regrind.assets import REGRIND_PROJECT_ROOT
from regrind.data.rb3_revo2_reference import REFERENCE_JOINT_NAMES
from regrind.data.rb3_revo2_reference import RB3_JOINT_NAMES
from regrind.robots.rb3_revo2 import REVO2_FOLLOWER_JOINTS
from regrind.tasks.manager_based.dexterous.mdp.actions import (
    ClippedRelativeJointPositionAction,
    ClippedRelativeJointPositionActionCfg,
    _rotvec_to_quat,
)
from regrind.utils.math import quat_from_euler_xyz_intrinsic


class RB3Revo2ResidualJointPositionAction(ClippedRelativeJointPositionAction):
    """Apply ``q_target = q_ref + scale * residual`` with Revo2 coupling.

    Only the six RB3 joints and six Revo2 leader joints consume policy actions.
    The five distal joints receive deterministic mimic targets and therefore do
    not increase the 12-dimensional action space.
    """

    cfg: "RB3Revo2ResidualJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if isinstance(self._joint_ids, slice):
            controlled_ids = list(range(self._asset.num_joints))[self._joint_ids]
        else:
            controlled_ids = [int(index) for index in self._joint_ids]
        controlled_names = tuple(self._asset.joint_names[index] for index in controlled_ids)
        expected_joint_names = tuple(cfg.expected_joint_names or REFERENCE_JOINT_NAMES)
        if controlled_names != expected_joint_names:
            raise RuntimeError(
                "Revo2 residual action joint order mismatch: "
                f"expected={expected_joint_names}, got={controlled_names}"
            )

        self._leader_columns = {name: index for index, name in enumerate(controlled_names)}
        self._follower_names = tuple(REVO2_FOLLOWER_JOINTS)
        missing = [name for name in self._follower_names if name not in self._asset.joint_names]
        if missing:
            raise RuntimeError(f"assembled robot is missing Revo2 follower joints: {missing}")
        self._follower_ids = [self._asset.joint_names.index(name) for name in self._follower_names]
        limits = self._asset.data.soft_joint_pos_limits.torch
        self._follower_lower = limits[:, self._follower_ids, 0]
        self._follower_upper = limits[:, self._follower_ids, 1]
        self._last_joint_target = torch.zeros_like(self.processed_actions)

    @property
    def last_joint_target(self) -> torch.Tensor:
        """Last clipped target for the 12 controllable joints."""

        return self._last_joint_target

    def apply_actions(self):
        joint_target = self.processed_actions + self.get_base_joint_pos()
        joint_target = torch.clamp(
            joint_target,
            min=self._dof_lower_limits,
            max=self._dof_upper_limits,
        )
        self._last_joint_target.copy_(joint_target)
        self._asset.set_joint_position_target_index(
            target=joint_target,
            joint_ids=self._joint_ids,
        )

        follower_targets = []
        for follower_name in self._follower_names:
            leader_name, multiplier, offset = REVO2_FOLLOWER_JOINTS[follower_name]
            leader = joint_target[:, self._leader_columns[leader_name]]
            follower_targets.append(offset + multiplier * leader)
        follower_target = torch.stack(follower_targets, dim=-1)
        follower_target = torch.clamp(
            follower_target,
            min=self._follower_lower,
            max=self._follower_upper,
        )
        self._asset.set_joint_position_target_index(
            target=follower_target,
            joint_ids=self._follower_ids,
        )


@configclass
class RB3Revo2ResidualJointPositionActionCfg(ClippedRelativeJointPositionActionCfg):
    class_type: type[ActionTerm] = RB3Revo2ResidualJointPositionAction
    # Defaults to the legacy assembled 12-DoF order. Floating Revo2 supplies
    # only the six leader names without changing the action implementation.
    expected_joint_names: tuple[str, ...] | None = None


def _load_rb3_kinematics_class():
    """Load the single source-of-truth strict IK module from the project tools.

    Isaac launchers add the installed ``regrind`` package to ``PYTHONPATH`` but
    do not consistently add the repository root. Loading by absolute path
    keeps the online controller and offline trajectory builder on the exact
    same FK/IK implementation without copying its equations.
    """

    module_name = "regrind_shared_rb3_kinematics"
    if module_name in sys.modules:
        return sys.modules[module_name].RB3730Kinematics
    module_path = REGRIND_PROJECT_ROOT / "tools" / "rb3_revo2_ik" / "rb3_kinematics.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"strict RB3 kinematics module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strict RB3 kinematics module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.RB3730Kinematics


class RB3WristIKAction(ClippedRelativeJointPositionAction):
    """Map the floating policy's Cartesian wrist residual to six RB3 joints.

    The policy-visible action is intentionally identical to
    :class:`SE3ImpedanceActionTerm`: three translation deltas followed by a
    rotation vector.  A bounded strict IK solve runs once per policy step and
    the resulting arm target is applied by the existing RB3 actuator, optionally
    interpolated across physics substeps.
    """

    cfg: "RB3WristIKActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.velocity_target_mode not in ("zero", "v_path"):
            raise ValueError("velocity_target_mode must be zero or v_path")
        if isinstance(self._joint_ids, slice):
            joint_ids = list(range(self._asset.num_joints))[self._joint_ids]
        else:
            joint_ids = [int(index) for index in self._joint_ids]
        joint_names = tuple(self._asset.joint_names[index] for index in joint_ids)
        if joint_names != tuple(RB3_JOINT_NAMES):
            raise RuntimeError(
                "RB3 wrist IK action joint order mismatch: "
                f"expected={tuple(RB3_JOINT_NAMES)}, got={joint_names}"
            )

        kinematics_class = _load_rb3_kinematics_class()
        model_path = Path(cfg.model_config_path).expanduser()
        if not model_path.is_absolute():
            model_path = REGRIND_PROJECT_ROOT / model_path
        self._kinematics = kinematics_class(
            model_config=model_path,
            base_position=cfg.base_position,
            base_quaternion_xyzw=cfg.base_quaternion_xyzw,
            verify_model_hash=cfg.verify_model_hash,
        )
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 3] = 1.0
        self._last_joint_target = torch.zeros(
            (self.num_envs, len(joint_ids)), device=self.device
        )
        self._applied_joint_target = torch.zeros_like(self._last_joint_target)
        self._interpolation_start_target = torch.zeros_like(self._last_joint_target)
        self._interpolation_step = 0
        self._warm_start_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._ik_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._ik_position_error = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self._ik_orientation_error = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self._solve_count = 0

    @property
    def last_joint_target(self) -> torch.Tensor:
        return self._last_joint_target

    @property
    def applied_joint_target(self) -> torch.Tensor:
        """Joint target sent on the most recent physics substep."""

        return self._applied_joint_target

    @property
    def ik_success(self) -> torch.Tensor:
        return self._ik_success

    @property
    def ik_position_error(self) -> torch.Tensor:
        return self._ik_position_error

    @property
    def ik_orientation_error(self) -> torch.Tensor:
        return self._ik_orientation_error

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )[env_ids]
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
        self._warm_start_valid[env_ids_tensor] = False
        self._ik_success[env_ids_tensor] = False
        self._ik_position_error[env_ids_tensor] = float("inf")
        self._ik_orientation_error[env_ids_tensor] = float("inf")
        if self.cfg.velocity_target_mode == "v_path":
            self._applied_joint_target[env_ids_tensor] = self._asset.data.joint_pos.torch[env_ids_tensor][:, self._joint_ids]
            self._asset.set_joint_velocity_target_index(
                target=torch.zeros_like(self._applied_joint_target[env_ids_tensor]),
                joint_ids=self._joint_ids, env_ids=env_ids_tensor,
            )

    def reset_from_reference(self, env_ids) -> None:
        """Synchronize arm state after the command selects its new RSI/placement.

        ActionManager.reset precedes CommandManager.reset in Isaac Lab; solving
        here from action.reset would use the previous episode's reference.
        """
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # The online task may sample a new rigid XY placement on every reset.
        # Put the arm directly at the corresponding first/reference wrist IK
        # solution before the first policy observation, instead of asking the
        # actuator to traverse the placement offset in one control step.
        command = self._env.command_manager.get_term(self.cfg.command_name)
        base_pos = command.target_hand_wrist_pos
        base_quat = command.target_hand_wrist_quat
        current_q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        for env_index in env_ids_tensor.detach().cpu().tolist():
            result = self._kinematics.inverse(
                base_pos[env_index].detach().cpu().numpy(),
                base_quat[env_index].detach().cpu().numpy(),
                initial_q=current_q[env_index].detach().cpu().numpy(),
                neutral_q=current_q[env_index].detach().cpu().numpy(),
                position_tolerance_m=self.cfg.position_tolerance_m,
                orientation_tolerance_rad=self.cfg.orientation_tolerance_rad,
                position_weight=self.cfg.position_weight,
                max_nfev=self.cfg.max_nfev,
            )
            self._ik_success[env_index] = result.success
            self._ik_position_error[env_index] = result.position_error_m
            self._ik_orientation_error[env_index] = result.orientation_error_rad
            if not (result.success and result.finite and not result.joint_limit_violation):
                raise RuntimeError(
                    "random-placement reset wrist is outside strict IK limits: "
                    f"env={env_index}, position={base_pos[env_index].detach().cpu().tolist()}, "
                    f"position_error={result.position_error_m:.6g} m, "
                    f"orientation_error={result.orientation_error_rad:.6g} rad"
                )
            self._last_joint_target[env_index] = torch.as_tensor(
                result.q,
                dtype=self._last_joint_target.dtype,
                device=self.device,
            )
            self._warm_start_valid[env_index] = True

        reset_target = self._last_joint_target[env_ids_tensor]
        self._applied_joint_target[env_ids_tensor] = reset_target
        self._interpolation_start_target[env_ids_tensor] = reset_target
        # This reset target is the first finite-difference predecessor. Do not
        # carry a velocity target (or position history) across episode resets.
        if self.cfg.velocity_target_mode == "v_path":
            self._asset.set_joint_velocity_target_index(
                target=torch.zeros_like(reset_target), joint_ids=self._joint_ids,
                env_ids=env_ids_tensor,
            )
        self.target_pos[env_ids_tensor] = base_pos[env_ids_tensor]
        self.target_quat[env_ids_tensor] = base_quat[env_ids_tensor]
        self._asset.write_joint_state_to_sim(
            reset_target,
            torch.zeros_like(reset_target),
            joint_ids=self._joint_ids,
            env_ids=env_ids_tensor,
        )
        self._asset.set_joint_position_target_index(
            target=reset_target,
            joint_ids=self._joint_ids,
            env_ids=env_ids_tensor,
        )

    def get_base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.base_action_source == "motion_target":
            command = self._env.command_manager.get_term(self.cfg.command_name)
            base_pos = command.target_hand_wrist_pos
            wrist_rot = command.target_hand_wrist_rot
            base_quat = quat_from_euler_xyz_intrinsic(
                wrist_rot[:, 0], wrist_rot[:, 1], wrist_rot[:, 2]
            )
            return base_pos, base_quat
        if self.cfg.base_action_source == "current_obs":
            command = self._env.command_manager.get_term(self.cfg.command_name)
            return command.current_hand_wrist_pos, command.current_hand_wrist_quat
        if self.cfg.base_action_source == "zero":
            pos = torch.zeros((self.num_envs, 3), device=self.device)
            quat = torch.zeros((self.num_envs, 4), device=self.device)
            quat[:, 3] = 1.0
            return pos, quat
        raise ValueError(f"invalid base_action_source: {self.cfg.base_action_source}")

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, 6):
            raise ValueError(
                f"RB3 wrist action must have shape ({self.num_envs},6), got {tuple(actions.shape)}"
            )
        if self.cfg.raw_clip is not None:
            actions = torch.clamp(actions, self.cfg.raw_clip[0], self.cfg.raw_clip[1])
        self._raw_actions.copy_(actions)
        self._processed_actions[:, :3] = actions[:, :3] * self.cfg.scale_pos
        self._processed_actions[:, 3:] = actions[:, 3:] * self.cfg.scale_rot

        # ``process_actions`` runs once per policy step while ``apply_actions``
        # runs once per physics substep.  Preserve the target that was actually
        # sent at the end of the previous step so the new 30 Hz IK command can
        # be ramped at the 120 Hz physics rate instead of arriving as a joint-
        # space step at contact.
        self._interpolation_start_target.copy_(self._applied_joint_target)
        self._interpolation_step = 0

        base_pos, base_quat = self.get_base_pose()
        self.target_pos.copy_(base_pos + self._processed_actions[:, :3])
        self.target_quat.copy_(
            quat_mul(_rotvec_to_quat(self._processed_actions[:, 3:]), base_quat)
        )

        current_q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        target_pos_np = self.target_pos.detach().cpu().numpy()
        target_quat_np = self.target_quat.detach().cpu().numpy()
        current_q_np = current_q.detach().cpu().numpy()
        previous_q_np = self._last_joint_target.detach().cpu().numpy()

        for env_index in range(self.num_envs):
            warm_q = (
                previous_q_np[env_index]
                if bool(self._warm_start_valid[env_index].item())
                else current_q_np[env_index]
            )
            result = self._kinematics.inverse(
                target_pos_np[env_index],
                target_quat_np[env_index],
                initial_q=warm_q,
                neutral_q=current_q_np[env_index],
                position_tolerance_m=self.cfg.position_tolerance_m,
                orientation_tolerance_rad=self.cfg.orientation_tolerance_rad,
                position_weight=self.cfg.position_weight,
                max_nfev=self.cfg.max_nfev,
            )
            self._ik_success[env_index] = result.success
            self._ik_position_error[env_index] = result.position_error_m
            self._ik_orientation_error[env_index] = result.orientation_error_rad
            if result.success and result.finite and not result.joint_limit_violation:
                self._last_joint_target[env_index] = torch.as_tensor(
                    result.q,
                    dtype=self._last_joint_target.dtype,
                    device=self.device,
                )
                self._warm_start_valid[env_index] = True
            elif not bool(self._warm_start_valid[env_index].item()):
                # On the first failed solve, hold the measured arm state. On a
                # later failure, retain the previous valid IK target. This is
                # explicit and prevents an optimizer branch jump.
                self._last_joint_target[env_index].copy_(current_q[env_index])

        self._solve_count += 1
        if self.cfg.debug_output and (
            self._solve_count == 1
            or self._solve_count % self.cfg.debug_interval == 0
            or not bool(torch.all(self._ik_success).item())
        ):
            failed = torch.nonzero(~self._ik_success, as_tuple=False).squeeze(-1)
            print(
                "[online wrist IK] "
                f"step={self._solve_count} success={self.num_envs - failed.numel()}/{self.num_envs} "
                f"max_pos={float(self._ik_position_error.max().item()):.6g} m "
                f"max_ori={float(self._ik_orientation_error.max().item()):.6g} rad "
                f"failed_envs={failed.detach().cpu().tolist()}"
            )

    def apply_actions(self):
        if self.cfg.interpolation_substeps > 1:
            self._interpolation_step = min(
                self._interpolation_step + 1,
                self.cfg.interpolation_substeps,
            )
            alpha = self._interpolation_step / self.cfg.interpolation_substeps
            target = torch.lerp(
                self._interpolation_start_target,
                self._last_joint_target,
                alpha,
            )
        else:
            target = self._last_joint_target
        velocity_target = None
        if self.cfg.velocity_target_mode == "v_path":
            # Final interpolated positions at physics rate, not policy/IK rate.
            # Preserve raw joint winding and genuine command discontinuities.
            velocity_target = (target - self._applied_joint_target) / self._env.physics_dt
        self._applied_joint_target.copy_(target)
        self._asset.set_joint_position_target_index(
            target=target,
            joint_ids=self._joint_ids,
        )
        if velocity_target is not None:
            self._asset.set_joint_velocity_target_index(
                target=velocity_target, joint_ids=self._joint_ids,
            )


@configclass
class RB3WristIKActionCfg(ClippedRelativeJointPositionActionCfg):
    """Configuration for online floating-policy wrist residual to RB3 IK."""

    class_type: type[ActionTerm] = RB3WristIKAction
    base_action_source: str = "motion_target"
    command_name: str = "reference"
    model_config_path: str = "tools/rb3_revo2_ik/rb3_model.json"
    base_position: tuple[float, float, float] = (0.0, 0.0, -0.02)
    base_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    verify_model_hash: bool = True
    scale_pos: float = 1.0
    scale_rot: float = 1.0
    position_tolerance_m: float = 1.0e-4
    orientation_tolerance_rad: float = 1.0e-3
    position_weight: float = 10.0
    max_nfev: int = 300
    # Number of physics substeps used to ramp each policy-rate IK target.
    # Leave at one outside the online assembled-arm bridge.
    interpolation_substeps: int = 1
    # Opt-in online evaluation; zero preserves the original command path.
    velocity_target_mode: str = "zero"
    debug_output: bool = False
    debug_interval: int = 10
