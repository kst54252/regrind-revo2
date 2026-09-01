"""REGRIND-compatible reference command for assembled RB3-730 + Revo2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch

from isaaclab.assets import BaseArticulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    normalize,
    quat_apply,
    quat_from_angle_axis,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    sample_uniform,
)

from regrind.data.rb3_revo2_reference import load_rb3_revo2_reference
from regrind.robots.rb3_revo2 import REVO2_FOLLOWER_JOINTS


class RB3Revo2ReferenceCommand(CommandTerm):
    """Expose a strict 12-DoF reference through the existing REGRIND MDP API.

    The properties intentionally mirror ``MotionCommand`` so the public
    REGRIND observation, reward, and termination functions can be reused
    without copying their equations.
    """

    cfg: "RB3Revo2ReferenceCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reference = load_rb3_revo2_reference(
            cfg.trajectory_path,
            require_success=cfg.require_success,
        )
        if abs(self.reference.dt - env.step_dt) > cfg.dt_tolerance:
            raise ValueError(
                f"reference dt {self.reference.dt:.9g}s does not match env step_dt "
                f"{env.step_dt:.9g}s; set sim.dt/decimation or prepare matching FPS"
            )

        keypoint_path = Path(cfg.object_keypoints_path).expanduser().resolve()
        if not keypoint_path.is_file():
            raise FileNotFoundError(f"tuna object keypoints not found: {keypoint_path}")
        object_keypoints = np.asarray(np.load(keypoint_path, allow_pickle=False), dtype=np.float32)
        if object_keypoints.shape != (50, 3) or not np.isfinite(object_keypoints).all():
            raise ValueError(
                f"tuna object keypoints must be finite with shape (50,3), got {object_keypoints.shape}"
            )

        self.robot: BaseArticulation = env.scene[cfg.robot_asset_name]
        self.object: RigidObject = env.scene[cfg.object_asset_name]
        missing = [name for name in self.reference.joint_names if name not in self.robot.joint_names]
        if missing:
            raise RuntimeError(f"assembled articulation is missing controlled joints: {missing}")
        self.joint_ids = [self.robot.joint_names.index(name) for name in self.reference.joint_names]
        self.actuated_dof_indices = self.joint_ids

        self.follower_names = tuple(REVO2_FOLLOWER_JOINTS)
        missing_followers = [name for name in self.follower_names if name not in self.robot.joint_names]
        if missing_followers:
            raise RuntimeError(f"assembled articulation is missing follower joints: {missing_followers}")
        self.follower_ids = [self.robot.joint_names.index(name) for name in self.follower_names]
        self.leader_columns = {name: index for index, name in enumerate(self.reference.joint_names)}

        if cfg.wrist_body_name not in self.robot.body_names:
            raise RuntimeError(
                f"wrist body {cfg.wrist_body_name!r} is absent; available={self.robot.body_names}"
            )
        self.wrist_body_id = self.robot.body_names.index(cfg.wrist_body_name)
        missing_tips = [name for name in cfg.fingertip_body_names if name not in self.robot.body_names]
        if missing_tips:
            raise RuntimeError(
                f"Revo2 fingertip bodies are absent: {missing_tips}; available={self.robot.body_names}"
            )
        if len(cfg.fingertip_body_names) != 5:
            raise ValueError("Revo2 critic observation requires exactly five fingertip links")
        self.fingertip_body_ids = [self.robot.body_names.index(name) for name in cfg.fingertip_body_names]

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        self.reference_joint_pos = tensor(self.reference.reference_joints)
        self.reference_object_pos = tensor(self.reference.object_pos)
        self.reference_object_quat = tensor(self.reference.object_quat_xyzw)
        self.reference_wrist_pos = tensor(self.reference.wrist_pos)
        self.reference_wrist_quat = tensor(self.reference.wrist_quat_xyzw)
        self.object_keypoints_local = tensor(object_keypoints)
        self.reference_joint_vel = self._finite_difference_vector(self.reference_joint_pos)
        self.reference_object_lin_vel = self._finite_difference_vector(self.reference_object_pos)
        self.reference_object_ang_vel = self._finite_difference_quat(self.reference_object_quat)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wrap_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_rsi_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for name in (
            "joint_tracking_error",
            "error_object_pos",
            "error_object_quat",
            "error_object_keypoints_pos",
            "error_object_lin_vel",
            "error_object_ang_vel",
            "error_hand_wrist_pos",
            "error_hand_wrist_rot",
            "error_hand_joint_pos",
            "error_hand_joint_vel",
            "gravity_z",
            "curriculum_level",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _finite_difference_vector(self, values: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(values)
        if values.shape[0] > 2:
            velocity[1:-1] = (values[2:] - values[:-2]) / (2.0 * self.reference.dt)
        velocity[-1] = (values[-1] - values[-2]) / self.reference.dt
        return velocity

    def _finite_difference_quat(self, values: torch.Tensor) -> torch.Tensor:
        count = values.shape[0]
        indices = torch.arange(count, device=self.device)
        previous = torch.clamp(indices - 1, min=0)
        following = torch.clamp(indices + 1, max=count - 1)
        delta = quat_mul(values[following], quat_inv(values[previous]))
        delta = torch.where(delta[:, 3:4] < 0.0, -delta, delta)
        duration = (following - previous).clamp_min(1).to(torch.float32) * self.reference.dt
        velocity = axis_angle_from_quat(delta) / duration.unsqueeze(-1)
        velocity[0] = 0.0
        return velocity

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Trajectory phase, matching REGRIND's generated command observation."""

        return self.phi.unsqueeze(-1)

    @property
    def phi(self) -> torch.Tensor:
        return self.time_steps.to(torch.float32) / float(max(self.reference.frames - 1, 1))

    @property
    def timesteps_in_current_demo(self) -> torch.Tensor:
        return self.time_steps

    @property
    def current_demo_length(self) -> torch.Tensor:
        return torch.full_like(self.time_steps, self.reference.frames)

    @property
    def target_joint_pos(self) -> torch.Tensor:
        return self.reference_joint_pos[self.time_steps]

    @property
    def target_hand_joint_pos(self) -> torch.Tensor:
        return self.target_joint_pos

    @property
    def target_hand_joint_vel(self) -> torch.Tensor:
        return self.reference_joint_vel[self.time_steps]

    @property
    def target_object_pos(self) -> torch.Tensor:
        return self.reference_object_pos[self.time_steps]

    @property
    def target_object_quat(self) -> torch.Tensor:
        return self.reference_object_quat[self.time_steps]

    @property
    def target_object_lin_vel(self) -> torch.Tensor:
        return self.reference_object_lin_vel[self.time_steps]

    @property
    def target_object_ang_vel(self) -> torch.Tensor:
        return self.reference_object_ang_vel[self.time_steps]

    @property
    def target_hand_wrist_pos(self) -> torch.Tensor:
        return self.reference_wrist_pos[self.time_steps]

    @property
    def target_hand_wrist_quat(self) -> torch.Tensor:
        return self.reference_wrist_quat[self.time_steps]

    @property
    def current_object_pos(self) -> torch.Tensor:
        return self.object.data.root_pos_w.torch - self._env.scene.env_origins

    @property
    def current_object_quat(self) -> torch.Tensor:
        return self.object.data.root_quat_w.torch

    @property
    def current_object_lin_vel(self) -> torch.Tensor:
        return self.object.data.root_lin_vel_w.torch

    @property
    def current_object_ang_vel(self) -> torch.Tensor:
        return self.object.data.root_ang_vel_w.torch

    def _world_object_keypoints(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        points = self.object_keypoints_local.unsqueeze(0).expand(pos.shape[0], -1, -1)
        rotations = quat.unsqueeze(1).expand(-1, points.shape[1], -1)
        return quat_apply(rotations, points) + pos.unsqueeze(1)

    @property
    def target_object_keypoints_pos(self) -> torch.Tensor:
        return self._world_object_keypoints(self.target_object_pos, self.target_object_quat)

    @property
    def current_object_keypoints_pos(self) -> torch.Tensor:
        return self._world_object_keypoints(self.current_object_pos, self.current_object_quat)

    @property
    def current_hand_wrist_pos(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.wrist_body_id] - self._env.scene.env_origins

    @property
    def current_hand_wrist_quat(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.wrist_body_id]

    @property
    def current_hand_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch[:, self.joint_ids]

    @property
    def current_hand_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch[:, self.joint_ids]

    @property
    def default_hand_joint_pos(self) -> torch.Tensor:
        return self.robot.data.default_joint_pos.torch[:, self.joint_ids]

    @property
    def current_fingertips_pos(self) -> torch.Tensor:
        positions = self.robot.data.body_pos_w.torch[:, self.fingertip_body_ids]
        return positions - self._env.scene.env_origins.unsqueeze(1)

    def _follower_target(self, joint_target: torch.Tensor) -> torch.Tensor:
        values = []
        for follower_name in self.follower_names:
            leader_name, multiplier, offset = REVO2_FOLLOWER_JOINTS[follower_name]
            values.append(offset + multiplier * joint_target[:, self.leader_columns[leader_name]])
        return torch.stack(values, dim=-1)

    def _follower_velocity(self, joint_velocity: torch.Tensor) -> torch.Tensor:
        values = []
        for follower_name in self.follower_names:
            leader_name, multiplier, _ = REVO2_FOLLOWER_JOINTS[follower_name]
            values.append(multiplier * joint_velocity[:, self.leader_columns[leader_name]])
        return torch.stack(values, dim=-1)

    def _update_metrics(self):
        self.metrics["error_object_pos"] = torch.linalg.vector_norm(
            self.target_object_pos - self.current_object_pos,
            dim=-1,
        )
        self.metrics["error_object_quat"] = quat_error_magnitude(
            self.target_object_quat,
            self.current_object_quat,
        )
        self.metrics["joint_tracking_error"].copy_(
            torch.linalg.vector_norm(
                self.target_joint_pos - self.current_hand_joint_pos,
                dim=-1,
            )
        )
        self.metrics["error_hand_joint_pos"].copy_(self.metrics["joint_tracking_error"])
        self.metrics["error_hand_joint_vel"].copy_(
            torch.linalg.vector_norm(
                self.target_hand_joint_vel - self.current_hand_joint_vel,
                dim=-1,
            )
        )
        gravity_z = float(self._env.sim.physics_sim_view.get_gravity()[2])
        self.metrics["gravity_z"].fill_(gravity_z)
        self.metrics["curriculum_level"].fill_(min(abs(gravity_z) / 9.81, 1.0))

    def _sample_rsi_frames(self, count: int) -> torch.Tensor:
        if not self.cfg.rsi_enabled:
            return torch.full((count,), int(self.cfg.start_frame), dtype=torch.long, device=self.device)
        # Same uniform-reference-state policy as MotionCommand: exclude the
        # terminal frame so every reset has at least one transition remaining.
        return torch.randint(
            low=0,
            high=max(self.reference.frames - 1, 1),
            size=(count,),
            device=self.device,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        selected = self._sample_rsi_frames(env_ids.numel())
        self.time_steps[env_ids] = selected
        self.last_rsi_frame[env_ids] = selected
        self.wrap_count[env_ids] = 0

        joint_pos = self.target_joint_pos[env_ids].clone()
        joint_vel = self.target_hand_joint_vel[env_ids].clone()
        object_pos = self.target_object_pos[env_ids].clone()
        object_quat = self.target_object_quat[env_ids].clone()
        if self.cfg.enable_reset_perturbation:
            joint_pos += sample_uniform(
                -self.cfg.joint_reset_noise,
                self.cfg.joint_reset_noise,
                joint_pos.shape,
                device=self.device,
            )
            object_pos += sample_uniform(
                -self.cfg.object_pos_reset_noise,
                self.cfg.object_pos_reset_noise,
                object_pos.shape,
                device=self.device,
            )
            axes = normalize(torch.randn((env_ids.numel(), 3), device=self.device))
            angles = sample_uniform(
                -self.cfg.object_rot_reset_noise,
                self.cfg.object_rot_reset_noise,
                (env_ids.numel(),),
                device=self.device,
            )
            object_quat = quat_mul(quat_from_angle_axis(angles, axes), object_quat)
            object_quat = normalize(object_quat)

        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids][:, self.joint_ids]
        joint_pos = torch.clamp(joint_pos, min=limits[..., 0], max=limits[..., 1])
        follower_pos = self._follower_target(joint_pos)
        follower_vel = self._follower_velocity(joint_vel)
        self.robot.write_joint_state_to_sim(
            joint_pos, joint_vel, joint_ids=self.joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            follower_pos, follower_vel, joint_ids=self.follower_ids, env_ids=env_ids
        )
        self.robot.set_joint_position_target_index(
            target=joint_pos, joint_ids=self.joint_ids, env_ids=env_ids
        )
        self.robot.set_joint_position_target_index(
            target=follower_pos, joint_ids=self.follower_ids, env_ids=env_ids
        )

        root_pose = torch.cat(
            (object_pos + self._env.scene.env_origins[env_ids], object_quat),
            dim=-1,
        )
        root_velocity = torch.cat(
            (self.target_object_lin_vel[env_ids], self.target_object_ang_vel[env_ids]),
            dim=-1,
        )
        self.object.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim_index(
            root_velocity=root_velocity,
            env_ids=env_ids,
        )
        if self.cfg.debug_output:
            print(f"[RSI] selected frame(s): {selected.detach().cpu().tolist()}")

    def _update_command(self):
        next_steps = self.time_steps + 1
        reached_end = next_steps >= self.reference.frames
        if self.cfg.loop:
            self.time_steps = torch.where(reached_end, torch.zeros_like(next_steps), next_steps)
            self.wrap_count += reached_end.long()
            if self.cfg.reset_object_on_loop and bool(torch.any(reached_end)):
                wrapped_env_ids = torch.nonzero(reached_end, as_tuple=False).squeeze(-1)
                object_pos = self.target_object_pos[wrapped_env_ids]
                object_quat = self.target_object_quat[wrapped_env_ids]
                root_pose = torch.cat(
                    (
                        object_pos + self._env.scene.env_origins[wrapped_env_ids],
                        object_quat,
                    ),
                    dim=-1,
                )
                root_velocity = torch.cat(
                    (
                        self.target_object_lin_vel[wrapped_env_ids],
                        self.target_object_ang_vel[wrapped_env_ids],
                    ),
                    dim=-1,
                )
                self.object.write_root_link_pose_to_sim_index(
                    root_pose=root_pose,
                    env_ids=wrapped_env_ids,
                )
                self.object.write_root_link_velocity_to_sim_index(
                    root_velocity=root_velocity,
                    env_ids=wrapped_env_ids,
                )
        else:
            self.time_steps = torch.clamp(next_steps, max=self.reference.frames - 1)


@configclass
class RB3Revo2ReferenceCommandCfg(CommandTermCfg):
    class_type: type = RB3Revo2ReferenceCommand
    trajectory_path: str = MISSING
    object_keypoints_path: str = MISSING
    robot_asset_name: str = "robot"
    object_asset_name: str = "object"
    wrist_body_name: str = "right_hand_base_link"
    fingertip_body_names: tuple[str, ...] = (
        "right_thumb_touch_link",
        "right_index_touch_link",
        "right_middle_touch_link",
        "right_ring_touch_link",
        "right_pinky_touch_link",
    )
    start_frame: int = 0
    loop: bool = False
    # Teleport the dynamic object back to the reference start pose whenever a
    # looping replay wraps. Disabled by default so training behavior is unchanged.
    reset_object_on_loop: bool = False
    require_success: bool = True
    rsi_enabled: bool = True
    enable_reset_perturbation: bool = False
    joint_reset_noise: float = 0.02
    object_pos_reset_noise: float = 0.002
    object_rot_reset_noise: float = 0.02
    debug_output: bool = False
    dt_tolerance: float = 1.0e-6
