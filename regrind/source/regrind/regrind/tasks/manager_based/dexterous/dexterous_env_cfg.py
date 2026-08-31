from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg

##
# Pre-defined configs
##
from isaaclab.utils.configclass import configclass

import regrind.tasks.manager_based.dexterous.mdp as mdp
from regrind.objects.table import TABLE_CFG
from regrind.utils.markers import FRAME_MARKER_CFG
from regrind.envs.events import (
    curriculum_gravity_with_randomization,
    curriculum_random_push,
    init_obs_delay_buffers,
    randomize_observation_time_lag,
    randomize_joint_default_pos,
    init_zero_joint_default_pos,
)


PUSH_VELOCITY_RANGE = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.2, 0.2),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
}

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Coefficient of restitution for rigid-body material DR (robot, object, table; PhysX: 0 = inelastic).
RESTITUTION_RANGE = (0.0, 0.0)

# Table surface (see ``MySceneCfg.table``): friction DR for object–table and hand–table contact.
TABLE_STATIC_FRICTION_RANGE = (0.5, 1.5)
TABLE_DYNAMIC_FRICTION_RANGE = (0.5, 1.5)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # table
    table: RigidObjectCfg = TABLE_CFG.replace(
        prim_path="/World/envs/env_.*/Table",
    )
    # objects
    object: ArticulationCfg = MISSING
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    # frame transforms (site tracking)
    # site_transforms will be computed dynamically based on dict `tf_target_frames`
    def _get_site_transforms(self, tf_source_prim_path: str, tf_target_frames: dict[str, list]) -> dict[str, FrameTransformerCfg]:
        """Get the site transforms configuration with dynamically combined target frames."""
        return {
            key: FrameTransformerCfg(
                prim_path=tf_source_prim_path,
                target_frames=frames,
                debug_vis=False,
                visualizer_cfg=FRAME_MARKER_CFG.replace(
                    prim_path=f"/Visuals/SiteFrameTransformer_{key}"
                ),
            ) for key, frames in tf_target_frames.items()
        }

    def setup_frame_transformers(
        self,
        tf_source_prim_path: str,
        tf_target_frames: dict[str, list],
    ):
        """Setup the frame transformers for the scene."""
        site_transforms_dict = self._get_site_transforms(tf_source_prim_path=tf_source_prim_path, tf_target_frames=tf_target_frames)
        for key, cfg in site_transforms_dict.items():
            setattr(self, f"tf_{key}", cfg)

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        object_asset_name="object",
        robot_asset_name="robot",
        tf_asset_names=["tf_fingertips"],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    root_pose = mdp.SE3ImpedanceActionCfg(
        asset_name="robot",
        body_name=None,
    )
    joint_pos = mdp.ClippedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        object_pos = ObsTerm(
            func=mdp.object_pos,
            params={"command_name": "motion", "delay_key": "object_pos", "apply_noise": True},
        )  # (num_envs, 3)
        object_ori = ObsTerm(
            func=mdp.object_ori,
            params={"command_name": "motion", "delay_key": "object_quat", "apply_noise": True},
        )  # (num_envs, 6)
        object_joint = ObsTerm(
            func=mdp.object_joint,
            params={"command_name": "motion", "delay_key": "object_joint", "apply_noise": True},
        )  # (num_envs, 1)
        hand_wrist_pos = ObsTerm(
            func=mdp.hand_wrist_pos,
            params={"command_name": "motion", "delay_key": "hand_wrist_pos", "apply_noise": True, "relative_to_default": False},  # NOTE: relative_to_default is not implemented for wrist obs
            history_length=2,
        )  # (num_envs, 3)
        hand_wrist_rot6d = ObsTerm(
            func=mdp.hand_wrist_rot6d,
            params={"command_name": "motion", "delay_key": "hand_wrist_quat", "apply_noise": True, "relative_to_default": False},  # NOTE: relative_to_default is not implemented for wrist obs
            history_length=2,
        )  # (num_envs, 6)
        hand_joint_pos = ObsTerm(
            func=mdp.hand_joint_pos,
            params={"command_name": "motion", "delay_key": "hand_joint_pos", "apply_noise": True, "relative_to_default": True},
            history_length=2,
        )  # (num_envs, 16)
        actions = ObsTerm(func=mdp.last_action)   # (num_envs, 22) = 6 root SE3 + 16 fingers
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        action_base_wrist_pos_and_rot6d = ObsTerm(func=mdp.action_base_wrist_pos_and_rot6d, params={"action_term_name": "root_pose"})  # (num_envs, 9)
        action_base_hand_joint_pos = ObsTerm(
            func=mdp.action_base_hand_joint_pos, params={"action_term_name": "joint_pos"}
        )  # (num_envs, 16)

        def __post_init__(self):
            self.enable_corruption = False  # NOTE: noise is applied in the observations function
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic (undelayed state + privileged terms)."""

        object_pos = ObsTerm(func=mdp.object_pos, params={"command_name": "motion"})
        object_ori = ObsTerm(func=mdp.object_ori, params={"command_name": "motion"})
        object_joint = ObsTerm(func=mdp.object_joint, params={"command_name": "motion"})
        hand_wrist_pos = ObsTerm(func=mdp.hand_wrist_pos, params={"command_name": "motion"}, history_length=2)
        hand_wrist_rot6d = ObsTerm(func=mdp.hand_wrist_rot6d, params={"command_name": "motion"}, history_length=2)
        hand_joint_pos = ObsTerm(func=mdp.hand_joint_pos, params={"command_name": "motion"}, history_length=2)
        actions = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        action_base_wrist_pos_and_rot6d = ObsTerm(func=mdp.action_base_wrist_pos_and_rot6d, params={"action_term_name": "root_pose"})
        action_base_hand_joint_pos = ObsTerm(
            func=mdp.action_base_hand_joint_pos, params={"action_term_name": "joint_pos"}
        )

        # Privileged observations
        fingertips_pos = ObsTerm(func=mdp.fingertips_pos, params={"command_name": "motion"})
        hand_joint_vel = ObsTerm(func=mdp.hand_joint_vel, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": RESTITUTION_RANGE,
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),  # (0.3, 3.0),  # default: 3.0
            "damping_distribution_params": (0.9, 1.1),  # (0.75, 1.5),  # default: 0.1
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    init_zero_joint_default_pos = EventTerm(
        func=init_zero_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    robot_fingers_joint_pos_offset = EventTerm(
        func=randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(if|mf|rf|th)_(mcp|rot|pip|dip|cmc|axl|ipl)"]),
            "pos_distribution_params": (-0.03, 0.03),
            "operation": "abs",  # ignore default joint positions
            "action_term_name": "joint_pos",
        },
    )
    robot_wrist_translation_joint_pos_offset = EventTerm(
        func=randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_wrist_(tx|ty|tz)_joint"]),
            "pos_distribution_params": (-0.002, 0.002),
            "operation": "abs",  # ignore default joint positions
            "action_term_name": "joint_pos",
        },
    )
    robot_wrist_rotation_joint_pos_offset = EventTerm(
        func=randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_wrist_(roll|pitch|yaw)_joint"]),
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "abs",  # ignore default joint positions
            "action_term_name": "joint_pos",
        },
    )

    init_obs_delay_buffers = EventTerm(
        func=init_obs_delay_buffers,
        mode="startup",
        params={},
    )

    # -- table (RigidObject ``table`` in ``MySceneCfg``)
    table_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table", body_names=".*"),
            "static_friction_range": TABLE_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": TABLE_DYNAMIC_FRICTION_RANGE,
            "restitution_range": RESTITUTION_RANGE,
            "num_buckets": 250,
            "make_consistent": True,
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": RESTITUTION_RANGE,
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    object_com: EventTerm = MISSING  # to be specified in child classes
    object_scale: EventTerm | None = None  # to be specified in child classes
    object_joint_friction: EventTerm | None = None  # articulated objects (e.g. scissors_joint)

    # --observation
    observation_time_lag = EventTerm(
        func=randomize_observation_time_lag,
        mode="reset",
        params={
            "time_lag_ranges": {  # ranges are inclusive
                "object_pos": (0, 2),
                "object_quat": (0, 2),
                "object_joint": (0, 2),
                "hand_wrist_pos": (0, 2),
                "hand_wrist_quat": (0, 2),
                "hand_joint_pos": (0, 2),
            },
            "consistent_groups": [["object_pos", "object_quat", "object_joint"], ["hand_wrist_pos", "hand_wrist_quat"], ["hand_joint_pos"]],
            "continuous_time_lag": True,
        },
    )

    # curriculum
    curriculum_gravity = EventTerm(
        func=curriculum_gravity_with_randomization,
        mode="reset",
        params={
            "gravity_stages": [(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))] + [
                (i * 10_000 + 20_000, (0.0, 0.0, -g_max), (0.0, 0.0, -g_min))
                for i, (g_min, g_max) in enumerate([
                    (0.0, 1.0),
                    (0.5, 2.0),
                    (1.0, 3.0),
                    (2.0, 4.0),
                    (3.0, 5.0),
                    (4.0, 6.0),
                    (5.0, 7.0),
                    (6.0, 8.0),
                    (7.0, 9.0),
                    (8.0, 9.81),
                    (9.0, 9.81),
                    (9.81, 9.81),
                ])
            ]
        },
    )

    curriculum_random_push_robot = EventTerm(
        func=curriculum_random_push,
        mode="interval",
        interval_range_s=(1.0, 5.0),
        params={
            "push_velocity_stages": [
                (0, {}),
                (130_000, {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1), "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}),
                (140_000, {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2), "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}),
                (150_000, {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3), "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.6, 0.6)}),
                (160_000, {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.4, 0.4), "roll": (-0.8, 0.8), "pitch": (-0.8, 0.8), "yaw": (-0.8, 0.8)}),
                (170_000, {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)}),
                # (180_000, {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "z": (-0.6, 0.6), "roll": (-1.2, 1.2), "pitch": (-1.2, 1.2), "yaw": (-1.2, 1.2)}),
                # (190_000, {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "z": (-0.7, 0.7), "roll": (-1.4, 1.4), "pitch": (-1.4, 1.4), "yaw": (-1.4, 1.4)}),
                # (200_000, {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "z": (-0.8, 0.8), "roll": (-1.6, 1.6), "pitch": (-1.6, 1.6), "yaw": (-1.6, 1.6)}),
                # (210_000, {"x": (-0.9, 0.9), "y": (-0.9, 0.9), "z": (-0.9, 0.9), "roll": (-1.8, 1.8), "pitch": (-1.8, 1.8), "yaw": (-1.8, 1.8)}),
                # (220_000, {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0), "roll": (-2.0, 2.0), "pitch": (-2.0, 2.0), "yaw": (-2.0, 2.0)}),
            ],
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )

    curriculum_random_push_object = EventTerm(
        func=curriculum_random_push,
        mode="interval",
        interval_range_s=(1.0, 5.0),
        params={
            "push_velocity_stages": [
                (0, {}),
                (130_000, {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1), "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}),
                (140_000, {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2), "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}),
                (150_000, {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3), "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.6, 0.6)}),
                (160_000, {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.4, 0.4), "roll": (-0.8, 0.8), "pitch": (-0.8, 0.8), "yaw": (-0.8, 0.8)}),
                (170_000, {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)}),
                # (180_000, {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "z": (-0.6, 0.6), "roll": (-1.2, 1.2), "pitch": (-1.2, 1.2), "yaw": (-1.2, 1.2)}),
                # (190_000, {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "z": (-0.7, 0.7), "roll": (-1.4, 1.4), "pitch": (-1.4, 1.4), "yaw": (-1.4, 1.4)}),
                # (200_000, {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "z": (-0.8, 0.8), "roll": (-1.6, 1.6), "pitch": (-1.6, 1.6), "yaw": (-1.6, 1.6)}),
                # (210_000, {"x": (-0.9, 0.9), "y": (-0.9, 0.9), "z": (-0.9, 0.9), "roll": (-1.8, 1.8), "pitch": (-1.8, 1.8), "yaw": (-1.8, 1.8)}),
                # (220_000, {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0), "roll": (-2.0, 2.0), "pitch": (-2.0, 2.0), "yaw": (-2.0, 2.0)}),
            ],
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    object_keypoints_pos_error_exp = RewTerm(
        func=mdp.object_keypoints_pos_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.02, "shaping_func": "exp"},
    )
    object_lin_vel_error_exp = RewTerm(
        func=mdp.object_lin_vel_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0, "shaping_func": "exp"},
    )
    object_ang_vel_error_exp = RewTerm(
        func=mdp.object_ang_vel_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14, "shaping_func": "exp"},
    )
    hand_wrist_pos_error_exp = RewTerm(
        func=mdp.hand_wrist_pos_error_exp,
        weight=0.05,
        params={"command_name": "motion", "std": 0.02, "shaping_func": "exp"},
    )
    hand_wrist_rot_error_exp = RewTerm(
        func=mdp.hand_wrist_rot_error_exp,
        weight=0.05,
        params={"command_name": "motion", "std": 0.2, "shaping_func": "exp"},
    )
    action_l2_exp = RewTerm(
        func=mdp.action_l2_exp,
        weight=0.0,  # 0.5 for wuji, 0.0 for leap
        params={"command_name": "motion", "std": 1.0},
    )
    action_rate_l2_exp = RewTerm(
        func=mdp.action_rate_l2_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.5},
    )
    action_out_of_bounds_exp = RewTerm(
        func=mdp.action_out_of_bounds_exp,
        weight=1.0,
        params={"std": 1.0},
    )
    early_terminated_penalty = RewTerm(
        func=mdp.is_early_terminated,
        weight=-10.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    demo_end_reached = DoneTerm(
        func=mdp.demo_end_reached,
        params={"command_name": "motion"},
        time_out=False,
    )
    object_deviation = DoneTerm(
        func=mdp.object_deviation,
        params={"command_name": "motion", "threshold": 0.15},
    )
    object_out_of_bounds = DoneTerm(
        func=mdp.object_out_of_bounds,
        params={"command_name": "motion"},
    )
    hand_far_from_object = DoneTerm(
        func=mdp.hand_far_from_object,
        params={"command_name": "motion", "threshold": 0.5},
    )
    hit_table = DoneTerm(
        func=mdp.hit_table,
        params={"sensor_names": [], "force_threshold": 100.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class DexterousEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dexterous manipulation environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=0.75)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Observation delay configuration (maximum lag in timesteps per key)
    obs_max_lags: dict[str, int] = {
        "object_pos": 0,
        "object_quat": 0,
        "object_joint": 0,
        "robot_wrist": 0,
        "robot_fingers": 0,
    }

    # frame transformer related settings
    tf_source_prim_path: str = MISSING
    tf_target_frames: dict[str, list] = MISSING

    # When True, adds scene ContactSensor with viewport debug visualization (arrows in Isaac Sim when not headless).
    enable_contact_sensor: bool = False
    # When True, adds privileged critic term fingertip_contact_forces_w (implies ContactSensor on scene; debug_vis only if enable_contact_sensor).
    include_contact_forces_in_critic_obs: bool = False

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim = SimulationCfg(
            dt=1 / 120,
            render_interval=self.decimation,
            physics=PhysxCfg(
                solver_type=1,
                max_position_iteration_count=192,
                max_velocity_iteration_count=1,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.00625,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**23,
                gpu_collision_stack_size=2**28,
                gpu_max_num_partitions=1,
            ),
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        # viewer settings
        self.viewer.eye = (2, 2, 3)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        # If observation time lag event is enabled, set obs_max_lags to match its maximum ranges,
        # ensuring the delay buffers have sufficient history length.
        if self.events is not None and getattr(self.events, "observation_time_lag", None) is not None:
            self.obs_max_lags = {
                key: time_lag_range[1]
                for key, time_lag_range in self.events.observation_time_lag.params["time_lag_ranges"].items()
            }

        if self.enable_contact_sensor or self.include_contact_forces_in_critic_obs:
            # Isaac Lab resolves only one path segment after the parent: "Robot/.*" matches direct children of
            # Robot (e.g. wrapper Xforms), not fingertip links. Leap/Wuji links live under Robot/root/.
            self.scene.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root/.*",
                history_length=3,
                track_air_time=True,
                force_threshold=0.5,
                # Red/green spheres only; use MotionCommand debug arrows for direction + magnitude.
                debug_vis=self.enable_contact_sensor and not self.commands.motion.debug_vis_contact_force_arrows,
                filter_prim_paths_expr=[],
            )
        if self.include_contact_forces_in_critic_obs:
            self.observations.critic = self.observations.critic.replace(
                fingertip_contact_forces_w=ObsTerm(
                    func=mdp.contact_sensor_net_forces_w,
                    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
                ),
            )

        # Training / default: no motion debug draw (MANO, keypoints, contact arrows). *-Play-* cfgs turn these on after super().
        self.commands.motion.debug_vis = False
        self.commands.motion.debug_vis_contact_force_arrows = False

    def sync_contact_sensor_sphere_debug_vis(self) -> None:
        """Re-apply ``ContactSensorCfg.debug_vis`` after changing ``commands.motion.debug_vis_contact_force_arrows``.

        ``ContactSensorCfg`` is created in :meth:`__post_init__` while PLAY subclasses still have the default
        ``debug_vis_contact_force_arrows=False``, which incorrectly enables Isaac sphere markers. Call at the
        end of each ``*_PLAY.__post_init__`` (after setting motion debug flags).
        """
        cf = getattr(self.scene, "contact_forces", None)
        if cf is None:
            return
        self.scene.contact_forces = cf.replace(
            debug_vis=self.enable_contact_sensor and not self.commands.motion.debug_vis_contact_force_arrows,
        )
