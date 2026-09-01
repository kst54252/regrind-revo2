# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
from isaaclab.sim.utils.extensions import enable_extension
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import BaseArticulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from regrind.utils.buffers import ContinuousDelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, DirectRLEnv, ManagerBasedRLEnv


def randomize_simple_articulation_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
    equal_scale_axes: str | None = None,
):
    """This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    NOTE(Yunhai): This is a custom event adapted from the IsaacLab ``randomize_rigid_body_scale`` function to
    randomize the scale of a simple articulation that contains only a rigid body and a fixed joint in the USD stage.
    Modifications include:
    - Remove the check for the articulation type.
    - Add support for tying equal scale across specific axes via ``equal_scale_axes``

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression "/World/envs/env_.*/Object" has a child
    with the path "/World/envs/env_.*/Object/mesh", then the relative child path should be "mesh" or
    "/mesh".

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # if isinstance(asset, BaseArticulation):
    #     raise ValueError(
    #         "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
    #         " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
    #         " each version of the articulation and using multi-asset spawning. For more details, refer to:"
    #         " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
    #     )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # sample scale values
    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    # enforce equal scaling across specified axes (e.g., "xz" -> x and z are equal)
    if equal_scale_axes:
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        # filter to valid axes while preserving order, and deduplicate
        axes = []
        for axis in equal_scale_axes.lower():
            if axis in axis_to_index and axis not in axes:
                axes.append(axis)
        if len(axes) >= 2:
            base_idx = axis_to_index[axes[0]]
            base_vals = rand_samples[:, base_idx : base_idx + 1]
            for axis in axes[1:]:
                idx = axis_to_index[axis]
                rand_samples[:, idx : idx + 1] = base_vals
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[int(env_id)] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


def init_obs_delay_buffers(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Initialize continuous observation delay buffers for manager-based environments.

    The maximum lag (history length) for each observation key is read from ``env.cfg.obs_max_lags``.
    Buffers are stored on the environment as ``env.obs_delay_buffers`` and use quaternion
    interpolation for keys containing the substring ``\"quat\"``.
    """
    # Avoid re-initialization if buffers already exist
    if hasattr(env, "obs_delay_buffers"):
        return
    cfg = getattr(env, "cfg", None)
    if cfg is None or not hasattr(cfg, "obs_max_lags"):
        return

    obs_max_lags: dict[str, int] = cfg.obs_max_lags
    print("Initializing observation delay buffers with max lags: ", obs_max_lags)
    env.obs_delay_buffers = {
        key: ContinuousDelayBuffer(
            history_length=obs_max_lags[key],
            batch_size=env.num_envs,
            device=env.device,
            use_quaternion_interpolation=("quat" in key),
        )
        for key in obs_max_lags.keys()
    }


def randomize_observation_time_lag(
    env: DirectRLEnv | ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    time_lag_ranges: dict[str, tuple[int, int]],
    consistent_groups: list[list[str]] | None = None,
    continuous_time_lag: bool = False,
) -> None:
    """
    Randomize the time lag of the observation delay buffer.

    Args:
        env: The environment instance
        env_ids: Environment IDs
        time_lag_ranges: Dictionary of (key, (min, max) time lags (inclusive)) for each observation type
        consistent_groups: Groups of observation keys that must share the same sampled lag.
            - Each inner list defines one group, e.g., [["proprio", "eef_pose"], ["rgb", "depth"]]
            - Keys not present in any group are sampled independently.
        continuous_time_lag: If True, sample continuous (float) lag values; otherwise sample discrete integers.
    """
    grouped_keys: set[str] = set()
    if consistent_groups:
        # Validate groups: keys exist and no overlaps
        for group in consistent_groups:
            for key in group:
                if key not in time_lag_ranges:
                    raise KeyError(f"Observation key '{key}' in consistent_groups not found in time_lag_ranges.")
                if key in grouped_keys:
                    raise ValueError(f"Observation key '{key}' appears in multiple consistent groups.")
            grouped_keys.update(group)

        # Assign lags per group using intersection of their ranges
        for group in consistent_groups:
            min_lag = max(time_lag_ranges[key][0] for key in group)
            max_lag = min(time_lag_ranges[key][1] for key in group)
            if min_lag > max_lag:
                raise ValueError(
                    f"Non-overlapping time lag ranges within group {group}. "
                    f"Computed empty intersection [{min_lag}, {max_lag}]."
                )
            if continuous_time_lag:
                # Uniform float in [min_lag, max_lag)
                lag_timesteps = min_lag + (max_lag - min_lag) * torch.rand(
                    (len(env_ids),), dtype=torch.float32, device=env.device
                )
            else:
                lag_timesteps = torch.randint(
                    min_lag, max_lag + 1, (len(env_ids),), dtype=torch.int, device=env.device
                ).float()
            for key in group:
                buf = env.obs_delay_buffers[key]
                if continuous_time_lag:
                    assert hasattr(buf, "continuous_time_lags")
                buf.set_time_lag(time_lag=lag_timesteps, batch_ids=env_ids)

    # For ungrouped keys, sample independently
    for key, time_lag_range in time_lag_ranges.items():
        if key in grouped_keys:
            continue
        if continuous_time_lag:
            lag_timesteps = time_lag_range[0] + (time_lag_range[1] - time_lag_range[0]) * torch.rand(
                (len(env_ids),), dtype=torch.float32, device=env.device
            )
        else:
            lag_timesteps = torch.randint(
                time_lag_range[0], time_lag_range[1] + 1, (len(env_ids),), dtype=torch.int, device=env.device
            )
        buf = env.obs_delay_buffers[key]
        if continuous_time_lag:
            assert hasattr(buf, "continuous_time_lags")
        buf.set_time_lag(time_lag=lag_timesteps, batch_ids=env_ids)


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: BaseArticulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device=asset.device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device=asset.device)

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device
    ).unsqueeze(1)

    # start from the current body-frame CoM pose and add random offsets
    coms = asset.data.body_com_pose_b.torch.clone()

    # Randomize the com in range
    coms[env_ids[:, None], body_ids, :3] += rand_samples

    # Set the new coms
    asset.set_coms_index(coms=coms[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids.to(asset.device))


def randomize_rigid_object_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of a RigidObject by adding random offsets."""
    # Get the rigid object asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(asset.device)

    # Isaac Lab 3.x ``set_coms_index`` expects the complete CoM transform
    # (position + quaternion), not only the position.  Preserve the nominal
    # orientation and randomize the translation exactly as before.
    com = asset.data.body_com_pose_b.torch.clone()  # (N, 1, 7) for a RigidObject
    ranges = torch.tensor(
        [[*com_range.get("x", (0.0, 0.0))],
         [*com_range.get("y", (0.0, 0.0))],
         [*com_range.get("z", (0.0, 0.0))]], device=asset.device
    )
    rand = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)
    com[env_ids, 0, :3] += rand
    asset.set_coms_index(coms=com[env_ids], env_ids=env_ids)


def init_zero_joint_default_pos(
    env: DirectRLEnv | ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
):
    """
    Initialize the joint default positions to zero.
    """
    # extract the used quantities (to enable type-hinting)
    asset: BaseArticulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if env_ids != slice(None) and joint_ids != slice(None):
        env_ids = env_ids[:, None]
    asset.data.default_joint_pos.torch[env_ids, joint_ids] = 0.0


def randomize_joint_default_pos(
    env: DirectRLEnv | ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    action_term_name: str | None = None,
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: BaseArticulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos.torch[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.torch.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos.torch[env_ids, joint_ids] = pos

        if hasattr(env, "action_manager") and action_term_name is not None:
            # update the offset in action since it is not updated automatically
            env.action_manager.get_term(action_term_name)._offset[env_ids, joint_ids] = pos


###
# Curriculum for direct workflow
###

class curriculum_gravity_with_randomization(ManagerTermBase):
    """Apply the staged gravity curriculum through Isaac Lab's stateful gravity randomizer."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from isaaclab.envs.mdp import randomize_physics_scene_gravity

        first_stage = sorted(cfg.params["gravity_stages"])[0]
        randomizer_cfg = EventTermCfg(
            func=randomize_physics_scene_gravity,
            mode=cfg.mode,
            params={
                "gravity_distribution_params": (first_stage[1], first_stage[2]),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        self._gravity_randomizer = randomize_physics_scene_gravity(randomizer_cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        gravity_stages: list[tuple[int, tuple[float, float, float], tuple[float, float, float]]],
    ) -> None:
        current_step = env.common_step_counter
        target_gravity_min = (0.0, 0.0, -9.81)
        target_gravity_max = (0.0, 0.0, -9.81)

        for step_threshold, gravity_value_min, gravity_value_max in sorted(gravity_stages):
            if current_step >= step_threshold:
                target_gravity_min = gravity_value_min
                target_gravity_max = gravity_value_max
            else:
                break

        self._gravity_randomizer(
            env=env,
            env_ids=env_ids,
            gravity_distribution_params=(target_gravity_min, target_gravity_max),
            operation="abs",
            distribution="uniform",
        )


def curriculum_random_push(
    env: DirectRLEnv | ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    push_velocity_stages: list[tuple[int, dict[str, tuple[float, float]]]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """
    Apply random push curriculum based on training progress.
    """
    from isaaclab.envs.mdp import push_by_setting_velocity
    current_step = env.common_step_counter

    # Find the appropriate push velocity based on training progress
    target_push_velocity_range = {}  # default

    for step_threshold, push_velocity_range in sorted(push_velocity_stages):
        if current_step >= step_threshold:
            target_push_velocity_range = push_velocity_range
        else:
            break

    push_by_setting_velocity(
        env=env,
        env_ids=env_ids,
        velocity_range=target_push_velocity_range,
        asset_cfg=asset_cfg,
    )
