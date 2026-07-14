from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
from regrind.tasks.manager_based.dexterous.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


SHAPING_FUNCTIONS = {
    "exp": lambda x, std: torch.exp(-x / std),
    "hyperbolic": lambda x, std: 1.0 / (1.0 + x / std),
}


def is_early_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize early terminated episodes that don't correspond to episodic timeouts."""
    return (env.termination_manager.terminated * (~env.termination_manager.get_term("demo_end_reached"))).float()


def object_keypoints_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, shaping_func: str="exp") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.current_object_keypoints_pos - command.target_object_keypoints_pos, p=2, dim=-1).mean(-1)
    reward = SHAPING_FUNCTIONS[shaping_func](error, std)
    command.metrics["error_object_keypoints_pos"] = error
    command.metrics["reward_object_keypoints_pos"] = reward
    return reward


def object_lin_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, shaping_func: str="exp") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.current_object_lin_vel - command.target_object_lin_vel), dim=-1)
    reward = torch.exp(-error / std**2)
    command.metrics["error_object_lin_vel"] = error
    command.metrics["reward_object_lin_vel"] = reward
    return reward


def object_ang_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, shaping_func: str="exp") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.current_object_ang_vel - command.target_object_ang_vel), dim=-1)
    reward = torch.exp(-error / std**2)
    command.metrics["error_object_ang_vel"] = error
    command.metrics["reward_object_ang_vel"] = reward
    return reward


def fingertips_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, shaping_func: str="exp") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.current_fingertips_pos - command.target_fingertips_pos, p=2, dim=-1).mean(-1)
    reward = SHAPING_FUNCTIONS[shaping_func](error, std)
    command.metrics["error_fingertips_pos"] = error
    command.metrics["reward_fingertips_pos"] = reward
    return reward


def hand_wrist_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, shaping_func: str="exp") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.current_hand_wrist_pos - command.target_hand_wrist_pos, p=2, dim=-1)
    reward = SHAPING_FUNCTIONS[shaping_func](error, std)
    command.metrics["error_hand_wrist_pos"] = error
    command.metrics["reward_hand_wrist_pos"] = reward
    return reward


def hand_wrist_rot_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, shaping_func: str="exp") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.current_hand_wrist_quat, command.target_hand_wrist_quat)
    reward = SHAPING_FUNCTIONS[shaping_func](error, std)
    command.metrics["error_hand_wrist_rot"] = error
    command.metrics["reward_hand_wrist_rot"] = reward
    return reward


def action_l2_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    action_l2 = torch.square(env.action_manager.action).mean(dim=-1)
    reward = torch.exp(-action_l2 / std**2)
    command.metrics["action_l2"] = action_l2
    command.metrics["action_rms"] = torch.sqrt(action_l2)
    command.metrics["reward_action_l2_exp"] = reward
    return reward


def action_rate_l2_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    action_rate_l2 = torch.square(env.action_manager.action - env.action_manager.prev_action).mean(dim=-1)
    reward = torch.exp(-action_rate_l2 / std**2)
    command.metrics["action_rate_l2"] = action_rate_l2
    command.metrics["action_rate_rms"] = torch.sqrt(action_rate_l2)
    command.metrics["reward_action_rate_l2_exp"] = reward
    return reward


def action_out_of_bounds(env: ManagerBasedRLEnv) -> torch.Tensor:
    actions = env.action_manager.action
    out_of_limits = -(actions - (-1.0)).clip(max=0.0) + (actions - 1.0).clip(min=0.0)
    return out_of_limits.sum(dim=-1)


def action_out_of_bounds_exp(env: ManagerBasedRLEnv, std: float = 1.0) -> torch.Tensor:
    actions = env.action_manager.action
    out_of_limits = -(actions - (-1.0)).clip(max=0.0) + (actions - 1.0).clip(min=0.0)
    return torch.exp(-out_of_limits.sum(dim=-1) / std)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def action_rate_l1_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    action_rate = torch.abs(env.action_manager.action - env.action_manager.prev_action).mean(dim=-1)
    return torch.exp(-action_rate / std)


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def fingertip_table_normal_force_exp(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    command_name: str = "motion",
    std: float = 2.0,
) -> torch.Tensor:
    """Shaped reward ``exp(-cost / std**2)`` from net **normal** fingertip–table forces (world frame, N).

    Same Gaussian kernel as ``action_l2_exp`` / velocity error terms: ``cost`` is the mean over
    fingers of ``||F||^2`` (from filtered ``force_matrix_w``). Low contact → reward near 1; large
    forces → reward toward 0.

    Leap/other hands need their own filtered sensors and ``sensor_names``. Empty ``sensor_names``
    returns zeros so a global non-zero weight does not add a constant bias.

    Args:
        env: RL env with ``env.scene.sensors[name]`` for each name in ``sensor_names``.
        sensor_names: Scene keys for fingertip-vs-table sensors; if empty, returns zeros.
        command_name: Motion command for metrics.
        std: Scale on ``cost`` (same units as ``cost``: mean squared force in N²).

    Returns:
        Shape ``(num_envs,)``, values in ``(0, 1]`` when sensors are valid.
    """
    device = env.device
    num_envs = env.num_envs
    if not sensor_names:
        return torch.zeros(num_envs, device=device)

    sq_terms: list[torch.Tensor] = []
    for name in sensor_names:
        if name not in env.scene.sensors:
            return torch.zeros(num_envs, device=device)
        cs: ContactSensor = env.scene.sensors[name]
        fm = cs.data.force_matrix_w
        if fm is None:
            return torch.zeros(num_envs, device=device)
        f = fm[:, 0, 0, :]
        f = torch.nan_to_num(f, nan=0.0)
        sq_terms.append(torch.sum(torch.square(f), dim=-1))

    cost = torch.stack(sq_terms, dim=-1).mean(dim=-1)
    reward = torch.exp(-cost / (std**2))

    command: MotionCommand = env.command_manager.get_term(command_name)
    command.metrics["cost_fingertip_table_contact_mean_sq_norm"] = cost.detach()
    command.metrics["reward_fingertip_table_contact_exp"] = reward.detach()

    return reward
