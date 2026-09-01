"""REGRIND MDP environment for RB3-730 + Revo2 + YCB tuna can.

Observation, reward, RSI, PPO, domain-randomization, and curriculum semantics
follow the public Leap/Wuji tasks.  Tuna rotational symmetry remains out of
scope.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

import regrind.tasks.manager_based.dexterous.mdp as mdp
from regrind.assets import REGRIND_PROJECT_ROOT
from regrind.data.rb3_revo2_reference import REFERENCE_JOINT_NAMES
from regrind.envs.events import (
    curriculum_gravity_with_randomization,
    curriculum_random_push,
    init_obs_delay_buffers,
    randomize_observation_time_lag,
    randomize_rigid_object_com,
)
from regrind.objects.tuna_can import TUNA_CAN_CFG
from regrind.robots.rb3_revo2 import RB3_REVO2_CFG


DEFAULT_REFERENCE_PATH = (
    REGRIND_PROJECT_ROOT
    / "outputs"
    / "isaac"
    / "dexycb"
    / "20200709_143626_right"
    / "rb3_revo2_reference.h5"
)
DEFAULT_OBJECT_KEYPOINTS_PATH = (
    REGRIND_PROJECT_ROOT / "007_tuna_fish_can" / "object_points_50.npy"
)

RB3_JOINT_PATTERN = r"^(base|shoulder|elbow|wrist1|wrist2|wrist3)$"
REVO2_JOINT_PATTERN = r"^right_.*_joint$"
RB3_BODY_PATTERN = r"^link[0-6]$"
REVO2_BODY_PATTERN = r"^right_.*"


@configclass
class ResidualScaleCfg:
    """Normalized residual action scales, in radians."""

    arm: float = 0.05
    hand: float = 0.15


@configclass
class ActuatorBaselineCfg:
    """Configurable multipliers around the USD/asset actuator baseline."""

    rb3_stiffness_scale: float = 1.0
    rb3_damping_scale: float = 1.0
    rb3_effort_scale: float = 1.0
    rb3_velocity_limit: float = 10.0
    revo2_stiffness_scale: float = 1.0
    revo2_damping_scale: float = 1.0
    revo2_effort_scale: float = 1.0
    revo2_velocity_limit: float = 100.0


@configclass
class RandomizationRangesCfg:
    """Conservative, embodiment-specific first-training ranges."""

    rb3_friction: tuple[float, float] = (0.6, 1.0)
    revo2_friction: tuple[float, float] = (0.7, 1.3)
    rb3_mass_scale: tuple[float, float] = (0.98, 1.02)
    revo2_mass_scale: tuple[float, float] = (0.9, 1.1)
    rb3_gain_scale: tuple[float, float] = (0.95, 1.05)
    revo2_gain_scale: tuple[float, float] = (0.8, 1.2)
    object_friction: tuple[float, float] = (0.5, 1.2)
    object_mass_scale: tuple[float, float] = (0.85, 1.15)
    table_friction: tuple[float, float] = (0.6, 1.2)
    object_com_xy: tuple[float, float] = (-0.002, 0.002)
    object_com_z: tuple[float, float] = (-0.001, 0.001)


@configclass
class RB3Revo2TunaSceneCfg(InteractiveSceneCfg):
    robot = RB3_REVO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = TUNA_CAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8)),
    )


@configclass
class CommandsCfg:
    reference = mdp.RB3Revo2ReferenceCommandCfg(
        trajectory_path=str(DEFAULT_REFERENCE_PATH),
        object_keypoints_path=str(DEFAULT_OBJECT_KEYPOINTS_PATH),
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        rsi_enabled=True,
        loop=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.RB3Revo2ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(REFERENCE_JOINT_NAMES),
        preserve_order=True,
        base_action_source="motion_target",
        command_name="reference",
        command_joint_target_name="target_joint_pos",
        raw_clip=(-1.0, 1.0),
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor state in the same order as the public REGRIND policy group."""

        object_pos = ObsTerm(
            func=mdp.object_pos,
            params={
                "command_name": "reference",
                "delay_key": "object_pos",
                "apply_noise": True,
            },
        )
        object_ori = ObsTerm(
            func=mdp.object_ori,
            params={
                "command_name": "reference",
                "delay_key": "object_quat",
                "apply_noise": True,
            },
        )
        hand_wrist_pos = ObsTerm(
            func=mdp.hand_wrist_pos,
            params={
                "command_name": "reference",
                "delay_key": "hand_wrist_pos",
                "apply_noise": True,
            },
            history_length=2,
        )
        hand_wrist_rot6d = ObsTerm(
            func=mdp.hand_wrist_rot6d,
            params={
                "command_name": "reference",
                "delay_key": "hand_wrist_quat",
                "apply_noise": True,
            },
            history_length=2,
        )
        hand_joint_pos = ObsTerm(
            func=mdp.hand_joint_pos,
            params={
                "command_name": "reference",
                "relative_to_default": True,
                "delay_key": "hand_joint_pos",
                "apply_noise": True,
            },
            history_length=2,
        )
        actions = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "reference"})
        action_base_hand_joint_pos = ObsTerm(
            func=mdp.action_base_hand_joint_pos,
            params={"action_term_name": "joint_pos"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Undelayed actor state plus REGRIND privileged physical state."""

        object_pos = ObsTerm(func=mdp.object_pos, params={"command_name": "reference"})
        object_ori = ObsTerm(func=mdp.object_ori, params={"command_name": "reference"})
        hand_wrist_pos = ObsTerm(
            func=mdp.hand_wrist_pos,
            params={"command_name": "reference"},
            history_length=2,
        )
        hand_wrist_rot6d = ObsTerm(
            func=mdp.hand_wrist_rot6d,
            params={"command_name": "reference"},
            history_length=2,
        )
        hand_joint_pos = ObsTerm(
            func=mdp.hand_joint_pos,
            params={"command_name": "reference", "relative_to_default": True},
            history_length=2,
        )
        actions = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "reference"})
        action_base_hand_joint_pos = ObsTerm(
            func=mdp.action_base_hand_joint_pos,
            params={"action_term_name": "joint_pos"},
        )
        object_lin_vel = ObsTerm(func=mdp.object_lin_vel, params={"command_name": "reference"})
        object_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"command_name": "reference"})
        fingertips_pos = ObsTerm(func=mdp.fingertips_pos, params={"command_name": "reference"})
        hand_joint_vel = ObsTerm(func=mdp.hand_joint_vel, params={"command_name": "reference"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Weights and equations reused from the public REGRIND baseline."""

    object_keypoints_pos_error_exp = RewTerm(
        func=mdp.object_keypoints_pos_error_exp,
        weight=1.5,
        params={"command_name": "reference", "std": 0.02, "shaping_func": "exp"},
    )
    object_lin_vel_error_exp = RewTerm(
        func=mdp.object_lin_vel_error_exp,
        weight=1.0,
        params={"command_name": "reference", "std": 1.0, "shaping_func": "exp"},
    )
    object_ang_vel_error_exp = RewTerm(
        func=mdp.object_ang_vel_error_exp,
        weight=1.0,
        params={"command_name": "reference", "std": 3.14, "shaping_func": "exp"},
    )
    hand_wrist_pos_error_exp = RewTerm(
        func=mdp.hand_wrist_pos_error_exp,
        weight=0.05,
        params={"command_name": "reference", "std": 0.02, "shaping_func": "exp"},
    )
    hand_wrist_rot_error_exp = RewTerm(
        func=mdp.hand_wrist_rot_error_exp,
        weight=0.05,
        params={"command_name": "reference", "std": 0.2, "shaping_func": "exp"},
    )
    action_l2_exp = RewTerm(
        func=mdp.action_l2_exp,
        weight=0.5,
        params={"command_name": "reference", "std": 1.0},
    )
    action_rate_l2_exp = RewTerm(
        func=mdp.action_rate_l2_exp,
        weight=1.0,
        params={"command_name": "reference", "std": 0.5},
    )
    action_out_of_bounds_exp = RewTerm(
        func=mdp.action_out_of_bounds_exp,
        weight=1.0,
        params={"std": 1.0},
    )
    early_terminated_penalty = RewTerm(func=mdp.is_early_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    demo_end_reached = DoneTerm(
        func=mdp.demo_end_reached,
        params={"command_name": "reference"},
    )
    # The public task defines successful completion as reaching the end of the
    # reference without an earlier failure.  The duplicate boolean term gives
    # RSL-RL an explicit Episode_Termination/success scalar to log.
    success = DoneTerm(
        func=mdp.demo_end_reached,
        params={"command_name": "reference"},
    )
    object_deviation = DoneTerm(
        func=mdp.object_deviation,
        params={"command_name": "reference", "threshold": 0.15},
    )
    hand_far_from_object = DoneTerm(
        func=mdp.hand_far_from_object,
        params={"command_name": "reference", "threshold": 0.5},
    )


@configclass
class EventsCfg:
    """Public REGRIND randomizers mapped to the assembled robot and rigid tuna."""

    rb3_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RB3_BODY_PATTERN),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    revo2_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=REVO2_BODY_PATTERN),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    rb3_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RB3_BODY_PATTERN),
            "mass_distribution_params": (0.98, 1.02),
            "operation": "scale",
        },
    )
    revo2_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=REVO2_BODY_PATTERN),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    rb3_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RB3_JOINT_PATTERN),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    revo2_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=REVO2_JOINT_PATTERN),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    table_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    object_com = EventTerm(
        func=randomize_rigid_object_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "com_range": {
                "x": (-0.002, 0.002),
                "y": (-0.002, 0.002),
                "z": (-0.001, 0.001),
            },
        },
    )
    init_obs_delay_buffers = EventTerm(func=init_obs_delay_buffers, mode="startup", params={})
    observation_time_lag = EventTerm(
        func=randomize_observation_time_lag,
        mode="reset",
        params={
            "time_lag_ranges": {
                "object_pos": (0, 2),
                "object_quat": (0, 2),
                "hand_wrist_pos": (0, 2),
                "hand_wrist_quat": (0, 2),
                "hand_joint_pos": (0, 2),
            },
            "consistent_groups": [
                ["object_pos", "object_quat"],
                ["hand_wrist_pos", "hand_wrist_quat"],
                ["hand_joint_pos"],
            ],
            "continuous_time_lag": True,
        },
    )
    curriculum_gravity = EventTerm(
        func=curriculum_gravity_with_randomization,
        mode="reset",
        params={
            "gravity_stages": [(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
            + [
                (i * 10_000 + 20_000, (0.0, 0.0, -g_max), (0.0, 0.0, -g_min))
                for i, (g_min, g_max) in enumerate(
                    [
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
                    ]
                )
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
            ],
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )


@configclass
class DeterministicEventsCfg:
    """No active event terms; Isaac Lab requires a config object, not None."""

    pass


@configclass
class CurriculumCfg:
    """Event-driven curriculum state is logged by the reference command."""

    pass


@configclass
class RB3Revo2TunaEnvCfg(ManagerBasedRLEnvCfg):
    scene: RB3Revo2TunaSceneCfg = RB3Revo2TunaSceneCfg(num_envs=4096, env_spacing=0.75)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    residual_scale: ResidualScaleCfg = ResidualScaleCfg()
    actuator_baseline: ActuatorBaselineCfg = ActuatorBaselineCfg()
    randomization_ranges: RandomizationRangesCfg = RandomizationRangesCfg()
    obs_max_lags: dict[str, int] = {
        "object_pos": 2,
        "object_quat": 2,
        "hand_wrist_pos": 2,
        "hand_wrist_quat": 2,
        "hand_joint_pos": 2,
    }

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.normalize_action = False
        self.sim = SimulationCfg(
            dt=1.0 / 120.0,
            render_interval=self.decimation,
            gravity=(0.0, 0.0, 0.0),
            physics=PhysxCfg(
                solver_type=1,
                max_position_iteration_count=64,
                max_velocity_iteration_count=4,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.00625,
                gpu_max_rigid_contact_count=2**20,
                gpu_max_rigid_patch_count=2**19,
            ),
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
            ),
        )
        self.actions.joint_pos.scale = {
            r"^(base|shoulder|elbow|wrist1|wrist2|wrist3)$": self.residual_scale.arm,
            r"^right_(thumb_metacarpal|thumb_proximal|index_proximal|middle_proximal|ring_proximal|pinky_proximal)_joint$": self.residual_scale.hand,
        }
        self.commands.reference.enable_reset_perturbation = True

        def _scale(value, factor):
            if isinstance(value, dict):
                return {key: item * factor for key, item in value.items()}
            return value * factor

        arm = self.scene.robot.actuators["rb3_arm"]
        arm.stiffness = _scale(arm.stiffness, self.actuator_baseline.rb3_stiffness_scale)
        arm.damping = _scale(arm.damping, self.actuator_baseline.rb3_damping_scale)
        arm.effort_limit_sim = _scale(arm.effort_limit_sim, self.actuator_baseline.rb3_effort_scale)
        arm.velocity_limit_sim = self.actuator_baseline.rb3_velocity_limit
        hand = self.scene.robot.actuators["revo2_hand"]
        hand.stiffness = _scale(hand.stiffness, self.actuator_baseline.revo2_stiffness_scale)
        hand.damping = _scale(hand.damping, self.actuator_baseline.revo2_damping_scale)
        hand.effort_limit_sim = _scale(hand.effort_limit_sim, self.actuator_baseline.revo2_effort_scale)
        hand.velocity_limit_sim = self.actuator_baseline.revo2_velocity_limit

        ranges = self.randomization_ranges
        for term_name, friction in (
            ("rb3_physics_material", ranges.rb3_friction),
            ("revo2_physics_material", ranges.revo2_friction),
            ("object_physics_material", ranges.object_friction),
            ("table_physics_material", ranges.table_friction),
        ):
            term = getattr(self.events, term_name)
            term.params["static_friction_range"] = friction
            term.params["dynamic_friction_range"] = friction
        self.events.rb3_mass.params["mass_distribution_params"] = ranges.rb3_mass_scale
        self.events.revo2_mass.params["mass_distribution_params"] = ranges.revo2_mass_scale
        self.events.rb3_actuator_gains.params["stiffness_distribution_params"] = ranges.rb3_gain_scale
        self.events.rb3_actuator_gains.params["damping_distribution_params"] = ranges.rb3_gain_scale
        self.events.revo2_actuator_gains.params["stiffness_distribution_params"] = ranges.revo2_gain_scale
        self.events.revo2_actuator_gains.params["damping_distribution_params"] = ranges.revo2_gain_scale
        self.events.object_mass.params["mass_distribution_params"] = ranges.object_mass_scale
        self.events.object_com.params["com_range"] = {
            "x": ranges.object_com_xy,
            "y": ranges.object_com_xy,
            "z": ranges.object_com_z,
        }
        self.viewer.eye = (1.25, 1.1, 0.9)
        self.viewer.lookat = (0.35, 0.0, 0.3)


@configclass
class RB3Revo2TunaEnvCfg_SMOKE(RB3Revo2TunaEnvCfg):
    """16-environment training configuration for PPO integration tests."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.2


@configclass
class RB3Revo2TunaEnvCfg_PLAY(RB3Revo2TunaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.5
        self.events = DeterministicEventsCfg()
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.commands.reference.rsi_enabled = False
        self.commands.reference.debug_output = True
        self.commands.reference.enable_reset_perturbation = False
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
