from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import BaseArticulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
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
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_joint_friction_coefficients(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize joint friction coefficients with correct (num_envs, num_joints) tensor shapes.

    Isaac Lab's ``randomize_joint_parameters`` indexes friction as
    ``friction_coeff[env_ids[:, None], joint_ids]``, which yields shape (N, 1, 1) for a single
    joint and breaks ``write_joint_friction_coefficient_to_sim``.
    """
    asset: BaseArticulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    def _select(coeff: torch.Tensor) -> torch.Tensor:
        if joint_ids == slice(None):
            return coeff[env_ids]
        return coeff[env_ids][:, joint_ids]

    friction_coeff = _randomize_prop_by_op(
        asset.data.default_joint_friction_coeff.torch.clone(),
        friction_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )
    friction_coeff = torch.clamp(friction_coeff, min=0.0)
    static_friction_coeff = _select(friction_coeff)

    major_version = int(env.sim.get_version()[0])
    if major_version >= 5:
        dynamic_friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_dynamic_friction_coeff.torch.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        viscous_friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_viscous_friction_coeff.torch.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
        viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)
        dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)
        dynamic_friction_coeff = _select(dynamic_friction_coeff)
        viscous_friction_coeff = _select(viscous_friction_coeff)
    else:
        dynamic_friction_coeff = None
        viscous_friction_coeff = None

    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static_friction_coeff,
        joint_dynamic_friction_coeff=dynamic_friction_coeff,
        joint_viscous_friction_coeff=viscous_friction_coeff,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
