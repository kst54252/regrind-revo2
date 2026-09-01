"""Residual action term for the 12 controllable RB3+Revo2 joints."""

from __future__ import annotations

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

from regrind.data.rb3_revo2_reference import REFERENCE_JOINT_NAMES
from regrind.robots.rb3_revo2 import REVO2_FOLLOWER_JOINTS
from regrind.tasks.manager_based.dexterous.mdp.actions import (
    ClippedRelativeJointPositionAction,
    ClippedRelativeJointPositionActionCfg,
)


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
        if controlled_names != REFERENCE_JOINT_NAMES:
            raise RuntimeError(
                "RB3+Revo2 action joint order mismatch: "
                f"expected={REFERENCE_JOINT_NAMES}, got={controlled_names}"
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
