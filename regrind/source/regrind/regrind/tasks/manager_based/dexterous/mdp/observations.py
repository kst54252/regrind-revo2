from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import (
    matrix_from_quat,
    normalize,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
)

from regrind.tasks.manager_based.dexterous.mdp.actions import (
    ClippedRelativeJointPositionAction,
    SE3ImpedanceActionTerm,
)
from regrind.tasks.manager_based.dexterous.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _maybe_apply_delay(env: "ManagerBasedEnv", key: str | None, data: torch.Tensor) -> torch.Tensor:
    """Apply observation delay using env.obs_delay_buffers if available and key is provided."""
    if key is None:
        return data
    # ObservationManager probes observation shapes before startup events create buffers.
    if not hasattr(env, "obs_delay_buffers") or env.obs_delay_buffers is None:
        return data
    buffers = env.obs_delay_buffers
    if key not in buffers:
        raise KeyError(f"obs delay key {key!r} missing from obs_delay_buffers (have {tuple(buffers.keys())})")
    return buffers[key].compute(data)


def _remove_command_translation(
    data: torch.Tensor,
    command: object,
) -> torch.Tensor:
    """Express world positions in the command's canonical translated frame.

    Floating-hand placement augmentation moves the complete object+wrist
    reference by one rigid translation. Subtracting that translation from
    positional observations makes the policy invariant to the sampled table
    location while retaining all deviations from the reference motion. Other
    command types do not expose ``observation_translation_offset`` and are left
    unchanged.
    """

    offset = getattr(command, "observation_translation_offset", None)
    if offset is None:
        return data
    while offset.ndim < data.ndim:
        offset = offset.unsqueeze(-2)
    return data - offset


# -- Object observations --

def object_pos(
    env: ManagerBasedEnv,
    command_name: str,
    delay_key: str | None = None,
    apply_noise: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    data = command.current_object_pos
    data = _remove_command_translation(data, command)
    if apply_noise:
        data = data + sample_uniform(-0.002, 0.002, data.shape, device=data.device)
    return _maybe_apply_delay(env, delay_key, data)


def object_ori(
    env: ManagerBasedEnv,
    command_name: str,
    delay_key: str | None = None,
    apply_noise: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    quat = command.current_object_quat
    if apply_noise:
        num_envs = quat.shape[0]
        # isotropic small rotation noise: random axis + uniform angle
        rand_axis = torch.randn((num_envs, 3), device=quat.device)
        rand_axis = normalize(rand_axis)
        rand_angle = sample_uniform(-0.02, 0.02, (num_envs,), device=quat.device)
        delta_quat = quat_from_angle_axis(rand_angle, rand_axis)
        quat = quat_mul(delta_quat, quat)
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    quat = _maybe_apply_delay(env, delay_key, quat)
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(mat.shape[0], -1)


def object_joint(
    env: ManagerBasedEnv,
    command_name: str,
    delay_key: str | None = None,
    apply_noise: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    data = command.current_object_joint
    if apply_noise and data is not None:
        data = data + sample_uniform(-0.02, 0.02, data.shape, device=data.device)
    return _maybe_apply_delay(env, delay_key, data)


def object_keypoints_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.current_object_keypoints_pos


def object_lin_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.current_object_lin_vel


def object_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.current_object_ang_vel


def target_object_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.target_object_pos


def target_object_ori(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.target_object_quat)
    return mat[..., :2].reshape(mat.shape[0], -1)


def target_object_joint(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.target_object_joint


def target_object_lin_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.target_object_lin_vel


def target_object_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.target_object_ang_vel


# -- Robot observations --

def hand_wrist_pos(
    env: ManagerBasedEnv,
    command_name: str,
    delay_key: str | None = None,
    apply_noise: bool = False,
    relative_to_default: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    data = command.current_hand_wrist_pos
    data = _remove_command_translation(data, command)
    if relative_to_default:
        raise NotImplementedError("Relative to default is not implemented for hand_wrist_pos")
    if apply_noise:
        noise = sample_uniform(-0.002, 0.002, data.shape, device=data.device)
        data = data + noise
    if delay_key is not None:
        data = _maybe_apply_delay(env, delay_key, data)
        # Delta-action controllers need an actual env/world-frame base even
        # when the policy observation itself is canonicalized.
        offset = getattr(command, "observation_translation_offset", None)
        env.delayed_hand_wrist_pos = data if offset is None else data + offset
    return data


def hand_wrist_rot6d(
    env: ManagerBasedEnv,
    command_name: str,
    delay_key: str | None = None,
    apply_noise: bool = False,
    relative_to_default: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    quat = command.current_hand_wrist_quat
    if relative_to_default:
        raise NotImplementedError("Relative to default is not implemented for hand_wrist_rot6d")
    if apply_noise:
        num_envs = quat.shape[0]
        rand_axis = torch.randn((num_envs, 3), device=quat.device)
        rand_axis = normalize(rand_axis)
        rand_angle = sample_uniform(-0.02, 0.02, (num_envs,), device=quat.device)
        delta_quat = quat_from_angle_axis(rand_angle, rand_axis)
        quat = quat_mul(delta_quat, quat)
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    if delay_key is not None:
        quat = _maybe_apply_delay(env, delay_key, quat)
        # Store delayed wrist quat for use by delta-action controllers.
        env.delayed_hand_wrist_quat = quat
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(mat.shape[0], -1)


def hand_joint_pos(
    env: ManagerBasedEnv,
    command_name: str,
    delay_key: str | None = None,
    apply_noise: bool = False,
    relative_to_default: bool = False,
) -> torch.Tensor:
    """Finger joint positions."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    data = command.current_hand_joint_pos
    if relative_to_default:
        data = data - command.default_hand_joint_pos
    if apply_noise:
        noise = sample_uniform(-0.02, 0.02, data.shape, device=data.device)
        data = data + noise
    if delay_key is not None:
        data = _maybe_apply_delay(env, delay_key, data)
        # Store delayed joints for use by delta-action controllers.
        env.delayed_hand_joint_pos = data
    return data


def hand_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.current_hand_joint_vel


# -- Action base observations (match delta-action controller bases) --


def action_base_wrist_pos(
    env: ManagerBasedEnv,
    action_term_name: str = "root_pose",
    command_name: str | None = None,
) -> torch.Tensor:
    """Env-relative wrist position used as the SE3 action base."""
    term = env.action_manager.get_term(action_term_name)
    if not isinstance(term, SE3ImpedanceActionTerm):
        raise TypeError(f"action term {action_term_name!r} must be SE3ImpedanceActionTerm, got {type(term)}")
    base_pos, _ = term.get_base_pose()
    if command_name is not None:
        command = env.command_manager.get_term(command_name)
        base_pos = _remove_command_translation(base_pos, command)
    return base_pos


def action_base_wrist_rot6d(env: ManagerBasedEnv, action_term_name: str = "root_pose") -> torch.Tensor:
    """Wrist orientation (rot6d) used as the SE3 action base."""
    term = env.action_manager.get_term(action_term_name)
    if not isinstance(term, SE3ImpedanceActionTerm):
        raise TypeError(f"action term {action_term_name!r} must be SE3ImpedanceActionTerm, got {type(term)}")
    _, base_quat = term.get_base_pose()
    mat = matrix_from_quat(base_quat)
    return mat[..., :2].reshape(mat.shape[0], -1)


def action_base_wrist_pos_and_rot6d(
    env: ManagerBasedEnv,
    action_term_name: str = "root_pose",
    command_name: str | None = None,
) -> torch.Tensor:
    """Wrist position and orientation (rot6d) used as the SE3 action base."""
    term = env.action_manager.get_term(action_term_name)
    if not isinstance(term, SE3ImpedanceActionTerm):
        raise TypeError(f"action term {action_term_name!r} must be SE3ImpedanceActionTerm, got {type(term)}")
    base_pos, base_quat = term.get_base_pose()
    if command_name is not None:
        command = env.command_manager.get_term(command_name)
        base_pos = _remove_command_translation(base_pos, command)
    mat = matrix_from_quat(base_quat)
    return torch.cat([base_pos, mat[..., :2].reshape(mat.shape[0], -1)], dim=-1)


def action_base_hand_joint_pos(env: ManagerBasedEnv, action_term_name: str = "joint_pos") -> torch.Tensor:
    """Absolute finger joint positions used as the relative joint-position action base."""
    term = env.action_manager.get_term(action_term_name)
    if not isinstance(term, ClippedRelativeJointPositionAction):
        raise TypeError(
            f"action term {action_term_name!r} must be ClippedRelativeJointPositionAction, got {type(term)}"
        )
    return term.get_base_joint_pos()


def fingertips_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    positions = _remove_command_translation(command.current_fingertips_pos, command)
    return positions.view(env.num_envs, -1)


def contact_sensor_net_forces_w(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Net contact reaction forces in world frame for bodies selected on the contact sensor."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    return forces.reshape(env.num_envs, -1)


def contact_sensor_net_normal_forces_w_ordered(
    env: ManagerBasedEnv,
    sensor_name: str,
    body_names_ordered: tuple[str, ...],
) -> torch.Tensor:
    """Net *normal* contact forces in world frame, in a fixed body order.

    Reads :attr:`ContactSensorData.net_forces_w` (Isaac Lab): the normal component of contact
    summed over all patches on each tracked rigid body. Friction / tangential contributions
    are not included. The vector for each body aggregates every contact on that link (e.g. table,
    manipulated object, and self-contact).

    Args:
        env: Environment exposing ``env.scene.sensors[sensor_name]``.
        sensor_name: Scene sensor key (e.g. ``\"contact_forces\"``).
        body_names_ordered: Names in the desired output order; each must appear in
            :attr:`ContactSensor.body_names`.

    Returns:
        Tensor of shape ``(num_envs, len(body_names_ordered) * 3)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    names = contact_sensor.body_names
    missing = [n for n in body_names_ordered if n not in names]
    if missing:
        raise ValueError(
            f"body_names_ordered has names not present on contact sensor {sensor_name!r}: {missing}"
        )
    indices = [names.index(n) for n in body_names_ordered]
    device = contact_sensor.data.net_forces_w.torch.device
    idx = torch.tensor(indices, device=device, dtype=torch.long)
    forces = contact_sensor.data.net_forces_w.torch[:, idx, :]
    return forces.reshape(env.num_envs, -1)
