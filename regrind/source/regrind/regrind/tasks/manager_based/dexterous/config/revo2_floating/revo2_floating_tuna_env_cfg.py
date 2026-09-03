"""Floating Revo2 + rigid tuna-can REGRIND environment.

The policy controls a 6D Cartesian wrist residual and six Revo2 leader-joint
residuals. RB3 does not exist in this training environment; a downstream
strict-IK bridge maps the resulting wrist trajectory to the physical arm.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

import regrind.tasks.manager_based.dexterous.mdp as mdp
from regrind.data.rb3_revo2_reference import REVO2_JOINT_NAMES
from regrind.objects.tuna_can import TUNA_CAN_CFG
from regrind.robots.free_revo2_right_hand import (
    FREE_REVO2_RIGHT_HAND_CFG,
    REVO2_FINGERTIP_BODY_NAMES,
    REVO2_RELATIVE_ACTION_SCALE,
)
from regrind.tasks.manager_based.dexterous.config.rb3_revo2.rb3_revo2_tuna_env_cfg import (
    ActuatorBaselineCfg,
    CurriculumCfg,
    DEFAULT_OBJECT_KEYPOINTS_PATH,
    DEFAULT_REFERENCE_PATH,
    DeterministicEventsCfg,
    EventsCfg,
    RandomizationRangesCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class FloatingRevo2SceneCfg(InteractiveSceneCfg):
    robot = FREE_REVO2_RIGHT_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
class FloatingCommandsCfg:
    reference = mdp.RB3Revo2ReferenceCommandCfg(
        trajectory_path=str(DEFAULT_REFERENCE_PATH),
        object_keypoints_path=str(DEFAULT_OBJECT_KEYPOINTS_PATH),
        robot_asset_name="robot",
        object_asset_name="object",
        wrist_body_name="right_hand_base_link",
        fingertip_body_names=REVO2_FINGERTIP_BODY_NAMES,
        joint_reference="revo2",
        reset_floating_root=True,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        rsi_enabled=True,
        loop=False,
    )


@configclass
class FloatingActionsCfg:
    root_pose = mdp.SE3ImpedanceActionCfg(
        asset_name="robot",
        body_name="right_hand_base_link",
        base_action_source="motion_target",
        command_name="reference",
        raw_clip=(-1.0, 1.0),
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
class FloatingObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        object_pos = ObsTerm(
            func=mdp.object_pos,
            params={"command_name": "reference", "delay_key": "object_pos", "apply_noise": True},
        )
        object_ori = ObsTerm(
            func=mdp.object_ori,
            params={"command_name": "reference", "delay_key": "object_quat", "apply_noise": True},
        )
        hand_wrist_pos = ObsTerm(
            func=mdp.hand_wrist_pos,
            params={"command_name": "reference", "delay_key": "hand_wrist_pos", "apply_noise": True},
            history_length=2,
        )
        hand_wrist_rot6d = ObsTerm(
            func=mdp.hand_wrist_rot6d,
            params={"command_name": "reference", "delay_key": "hand_wrist_quat", "apply_noise": True},
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
        action_base_wrist_pos_and_rot6d = ObsTerm(
            func=mdp.action_base_wrist_pos_and_rot6d,
            params={"action_term_name": "root_pose"},
        )
        action_base_hand_joint_pos = ObsTerm(
            func=mdp.action_base_hand_joint_pos,
            params={"action_term_name": "joint_pos"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
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
        action_base_wrist_pos_and_rot6d = ObsTerm(
            func=mdp.action_base_wrist_pos_and_rot6d,
            params={"action_term_name": "root_pose"},
        )
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
class FloatingRevo2TunaEnvCfg(ManagerBasedRLEnvCfg):
    scene: FloatingRevo2SceneCfg = FloatingRevo2SceneCfg(num_envs=4096, env_spacing=0.75)
    observations: FloatingObservationsCfg = FloatingObservationsCfg()
    actions: FloatingActionsCfg = FloatingActionsCfg()
    commands: FloatingCommandsCfg = FloatingCommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
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

        control_dt = self.sim.dt * self.decimation
        self.actions.root_pose.scale_pos = 1.0 * control_dt
        self.actions.root_pose.scale_rot = 3.2 * control_dt
        self.actions.root_pose.kp_pos = 300.0
        self.actions.root_pose.kd_pos = 30.0
        self.actions.root_pose.kp_rot = 3.0
        self.actions.root_pose.kd_rot = 0.3
        self.actions.joint_pos.scale = {
            key: value * control_dt for key, value in REVO2_RELATIVE_ACTION_SCALE.items()
        }

        # The shared event config also describes the legacy RB3 branch. Remove
        # only those terms; all hand/object/noise/curriculum terms are reused.
        self.events.rb3_physics_material = None
        self.events.rb3_mass = None
        self.events.rb3_actuator_gains = None
        self.commands.reference.enable_reset_perturbation = True

        ranges = self.randomization_ranges
        for term_name, friction in (
            ("revo2_physics_material", ranges.revo2_friction),
            ("object_physics_material", ranges.object_friction),
            ("table_physics_material", ranges.table_friction),
        ):
            term = getattr(self.events, term_name)
            term.params["static_friction_range"] = friction
            term.params["dynamic_friction_range"] = friction
        self.events.revo2_mass.params["mass_distribution_params"] = ranges.revo2_mass_scale
        self.events.revo2_actuator_gains.params["stiffness_distribution_params"] = ranges.revo2_gain_scale
        self.events.revo2_actuator_gains.params["damping_distribution_params"] = ranges.revo2_gain_scale
        self.events.object_mass.params["mass_distribution_params"] = ranges.object_mass_scale
        self.events.object_com.params["com_range"] = {
            "x": ranges.object_com_xy,
            "y": ranges.object_com_xy,
            "z": ranges.object_com_z,
        }
        self.viewer.eye = (0.95, 0.75, 0.65)
        self.viewer.lookat = (0.4, 0.0, 0.16)


@configclass
class FloatingRevo2TunaEnvCfg_SMOKE(FloatingRevo2TunaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.0


@configclass
class FloatingRevo2TunaEnvCfg_PLAY(FloatingRevo2TunaEnvCfg):
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
