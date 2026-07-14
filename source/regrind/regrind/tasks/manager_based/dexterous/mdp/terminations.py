from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from regrind.tasks.manager_based.dexterous.mdp.commands import MotionCommand


@torch.jit.script
def _is_out_of_bounds(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """
    Check if any element of x is out of bounds.
    Args:
        x: Tensor of shape (..., n)
        lower: Tensor of shape (n)
        upper: Tensor of shape (n)
    Returns:
        Boolean tensor of shape (...) — the last dimension is reduced.
    """
    return torch.logical_or(
        torch.any(x < lower, dim=-1),
        torch.any(x > upper, dim=-1),
    )


def object_deviation(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.current_object_keypoints_pos - command.target_object_keypoints_pos, p=2, dim=-1).mean(-1) > threshold


def object_out_of_bounds(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _is_out_of_bounds(command.current_object_pos, command.object_pos_lower_bound, command.object_pos_upper_bound)


def object_rotation_error(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return quat_error_magnitude(command.current_object_quat, command.target_object_quat) > threshold


def hand_far_from_object(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.current_hand_wrist_pos - command.current_object_pos, p=2, dim=-1) > threshold


def demo_end_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return (command.timesteps_in_current_demo + 1) >= command.current_demo_length


def hit_table(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """True when any fingertip–table normal contact force magnitude exceeds ``force_threshold`` (N).

    Expects filtered contact sensors (e.g. ``contact_fingertip{i}_table``) whose
    ``ContactSensorData.force_matrix_w`` reports normal forces in world frame. NaNs (no contact)
    are treated as zero force. If ``sensor_names`` is empty or sensors are missing, never terminates.
    """
    device = env.device
    num_envs = env.num_envs
    if not sensor_names:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    mags: list[torch.Tensor] = []
    for name in sensor_names:
        if name not in env.scene.sensors:
            return torch.zeros(num_envs, dtype=torch.bool, device=device)
        cs: ContactSensor = env.scene.sensors[name]
        fm = cs.data.force_matrix_w
        if fm is None:
            return torch.zeros(num_envs, dtype=torch.bool, device=device)
        f = fm[:, 0, 0, :]
        f = torch.nan_to_num(f, nan=0.0)
        mags.append(torch.linalg.norm(f, dim=-1))

    max_mag = torch.stack(mags, dim=-1).max(dim=-1).values
    return max_mag > force_threshold
