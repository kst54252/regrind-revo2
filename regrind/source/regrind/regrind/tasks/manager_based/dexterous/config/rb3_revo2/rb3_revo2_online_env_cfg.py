"""Online floating-policy deployment on the assembled RB3+Revo2 workcell.

This environment is deliberately an evaluation bridge, not a new RL task.
It preserves the floating checkpoint's 67-D actor observation and 12-D action
contract, then maps the Cartesian wrist action through strict bounded RB3 IK.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

import regrind.tasks.manager_based.dexterous.mdp as mdp
from regrind.data.rb3_revo2_reference import RB3_JOINT_NAMES, REVO2_JOINT_NAMES
from regrind.robots.free_revo2_right_hand import REVO2_RELATIVE_ACTION_SCALE
from regrind.workcell import ROBOT_MOUNT_POSITION, ROBOT_MOUNT_QUATERNION_XYZW
from regrind.tasks.manager_based.dexterous.config.revo2_floating.revo2_floating_tuna_env_cfg import (
    FloatingObservationsCfg,
)

from .rb3_revo2_tuna_env_cfg import (
    CommandsCfg,
    DeterministicEventsCfg,
    RB3Revo2TunaEnvCfg,
)


@configclass
class OnlineActionsCfg:
    """Checkpoint-compatible action order: wrist 6D, then Revo2 leaders."""

    root_pose = mdp.RB3WristIKActionCfg(
        asset_name="robot",
        joint_names=list(RB3_JOINT_NAMES),
        preserve_order=True,
        base_action_source="motion_target",
        command_name="reference",
        raw_clip=(-1.0, 1.0),
        scale=1.0,
        base_position=ROBOT_MOUNT_POSITION,
        base_quaternion_xyzw=ROBOT_MOUNT_QUATERNION_XYZW,
    )
    joint_pos = mdp.RB3Revo2ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(REVO2_JOINT_NAMES),
        preserve_order=True,
        expected_joint_names=REVO2_JOINT_NAMES,
        base_action_source="motion_target",
        command_name="reference",
        command_joint_target_name="target_hand_joint_pos",
        raw_clip=(-1.0, 1.0),
        scale=1.0,
    )


@configclass
class RB3Revo2TunaOnlineEnvCfg(RB3Revo2TunaEnvCfg):
    """Assembled physics scene driven online by a floating-hand checkpoint."""

    observations: FloatingObservationsCfg = FloatingObservationsCfg()
    actions: OnlineActionsCfg = OnlineActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        control_dt = self.sim.dt * self.decimation

        # Exactly match the action normalization used by floating training.
        self.actions.root_pose.scale_pos = 1.0 * control_dt
        self.actions.root_pose.scale_rot = 3.2 * control_dt
        self.actions.joint_pos.scale = {
            key: value * control_dt for key, value in REVO2_RELATIVE_ACTION_SCALE.items()
        }

        # Reset all twelve joints from the strict reference, while exposing
        # only Revo2's six leaders to the checkpoint observations/rewards.
        self.commands.reference.joint_reference = "combined"
        self.commands.reference.expose_revo2_as_hand = True
        self.commands.reference.reset_floating_root = False
        # At every episode reset, translate the can and complete wrist/object
        # reference together inside the prevalidated strict-IK table region.
        # Canonical observations keep the floating checkpoint invariant to
        # this rigid placement offset.
        self.commands.reference.randomize_object_xy = True
        self.commands.reference.object_start_x_range = (0.40, 0.50)
        self.commands.reference.object_start_y_range = (-0.20, 0.20)
        self.commands.reference.canonicalize_translation_observations = True


@configclass
class RB3Revo2TunaOnlineEnvCfg_PLAY(RB3Revo2TunaOnlineEnvCfg):
    """One-environment deterministic closed-loop policy evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0
        self.events = DeterministicEventsCfg()
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.commands.reference.rsi_enabled = False
        self.commands.reference.debug_output = True
        self.commands.reference.enable_reset_perturbation = False
        self.actions.root_pose.debug_output = True
        for term_name in (
            "object_pos",
            "object_ori",
            "hand_wrist_pos",
            "hand_wrist_rot6d",
            "hand_joint_pos",
        ):
            term = getattr(self.observations.policy, term_name)
            term.params["delay_key"] = None
            term.params["apply_noise"] = False
