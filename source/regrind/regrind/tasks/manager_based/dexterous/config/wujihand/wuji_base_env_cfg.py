from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg

import regrind.tasks.manager_based.dexterous.mdp as mdp

from regrind.robots.free_wuji_right_hand import (
    ACTUATED_JOINT_NAMES,
    FREE_WUJI_RIGHT_HAND_CFG,
    RELATIVE_ACTION_SCALE,
    WUJI_FINGERTIP_BODY_NAMES,
)
from regrind.tasks.manager_based.dexterous.dexterous_env_cfg import DexterousEnvCfg
from regrind.tasks.manager_based.dexterous.mdp.wuji_fingertip_table_contact import (
    WUJI_FINGERTIP_TABLE_CONTACT_SENSOR_NAMES,
    install_wuji_fingertip_table_contact_sensors,
)


@configclass
class WujiHandBaseEnvCfg(DexterousEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FREE_WUJI_RIGHT_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.enable_contact_sensor or self.include_contact_forces_in_critic_obs:
            self.scene.robot = self.scene.robot.replace(
                spawn=self.scene.robot.spawn.replace(activate_contact_sensors=True),
            )

        self.tf_source_prim_path: str = "{ENV_REGEX_NS}/Robot/root/root"
        # NOTE: the source prim should not matter because we are reading target_pos_w
        self.tf_target_frames: dict[str, list] = {
            "fingertips": [
                FrameTransformerCfg.FrameCfg(  # thumb (mano_16)
                    prim_path="/World/envs/env_.*/Robot/root/right_finger1_link4",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.03)),
                ),
                FrameTransformerCfg.FrameCfg(  # index (mano_17)
                    prim_path="/World/envs/env_.*/Robot/root/right_finger2_link4",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.021)),
                ),
                FrameTransformerCfg.FrameCfg(  # middle (mano_18)
                    prim_path="/World/envs/env_.*/Robot/root/right_finger3_link4",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.021)),
                ),
                FrameTransformerCfg.FrameCfg(  # ring (mano_19)
                    prim_path="/World/envs/env_.*/Robot/root/right_finger4_link4",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.021)),
                ),
                FrameTransformerCfg.FrameCfg(  # little (mano_20)
                    prim_path="/World/envs/env_.*/Robot/root/right_finger5_link4",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0225)),
                ),
            ],
        }
        for key in self.tf_target_frames.keys():
            tf_name = f"tf_{key}"
            if tf_name not in self.commands.motion.tf_asset_names:
                self.commands.motion.tf_asset_names.append(tf_name)

        self.commands.motion.actuated_joint_names = ACTUATED_JOINT_NAMES

        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        from regrind.tasks.manager_based.dexterous.mdp import randomize_rigid_body_material

        self.events = self.events.replace(
            robot_wrist_translation_joint_pos_offset=None,
            robot_wrist_rotation_joint_pos_offset=None,

            robot_physics_material=EventTerm(
                func=randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.5, 1.5),
                    "dynamic_friction_range": (0.5, 1.5),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 250,
                    "make_consistent": True,
                },
            ),
            robot_fingers_joint_pos_offset=None,
        )

        self.actions.root_pose.body_name = "root"
        self.actions.root_pose.base_action_source = "motion_target"
        self.actions.root_pose.scale_pos = 1.0 * self.sim.dt * self.decimation
        self.actions.root_pose.scale_rot = 3.2 * self.sim.dt * self.decimation
        # Cartesian wrench PD on the floating root (world frame). Initial guess; retune for FREE_WUJI_RIGHT_HAND.
        self.actions.root_pose.kp_pos = 300.0
        self.actions.root_pose.kd_pos = 30.0
        self.actions.root_pose.kp_rot = 3.0
        self.actions.root_pose.kd_rot = 0.3

        self.actions.joint_pos.base_action_source = "motion_target"
        self.actions.joint_pos.scale = {
            key: value * self.sim.dt * self.decimation for key, value in RELATIVE_ACTION_SCALE.items()
        }
        self.actions.joint_pos.joint_names = ACTUATED_JOINT_NAMES

        # Smaller motion-command debug markers (MANO skeleton + object keypoints) than manager dexterous default.
        self.commands.motion.debug_vis_marker_scale = 0.5

        if self.include_contact_forces_in_critic_obs:
            self.observations.critic = self.observations.critic.replace(
                fingertip_contact_forces_w=ObsTerm(
                    func=mdp.contact_sensor_net_normal_forces_w_ordered,
                    params={
                        "sensor_name": "contact_forces",
                        "body_names_ordered": WUJI_FINGERTIP_BODY_NAMES,
                    },
                ),
            )

        self.rewards.object_keypoints_pos_error_exp.params["std"] = 0.015
        self.rewards.action_l2_exp.weight = 0.5

        install_wuji_fingertip_table_contact_sensors(self.scene)
        self.terminations.hit_table.params["sensor_names"] = list(WUJI_FINGERTIP_TABLE_CONTACT_SENSOR_NAMES)
        self.terminations.hit_table.params["force_threshold"] = 10.0
