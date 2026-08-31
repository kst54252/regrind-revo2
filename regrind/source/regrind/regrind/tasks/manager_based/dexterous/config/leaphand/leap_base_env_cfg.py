from isaaclab.utils.configclass import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg

import regrind.tasks.manager_based.dexterous.mdp as mdp
from regrind.robots.free_leap_right_hand import FREE_LEAP_RIGHT_HAND_CFG, RELATIVE_ACTION_SCALE, ACTUATED_JOINT_NAMES
from regrind.tasks.manager_based.dexterous.dexterous_env_cfg import DexterousEnvCfg


@configclass
class LeapHandBaseEnvCfg(DexterousEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FREE_LEAP_RIGHT_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.enable_contact_sensor or self.include_contact_forces_in_critic_obs:
            self.scene.robot = self.scene.robot.replace(
                spawn=self.scene.robot.spawn.replace(activate_contact_sensors=True),
            )

        self.tf_source_prim_path: str = "{ENV_REGEX_NS}/Robot/root/root"
        # NOTE: the source prim should not matter because we are reading target_pos_w
        self.tf_target_frames: dict[str, list] = {
            "fingertips": [
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/root/if_ds",
                    offset=OffsetCfg(pos=(0.0, -0.045, 0.0144)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/root/mf_ds",
                    offset=OffsetCfg(pos=(0.0, -0.045, 0.0144)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/root/rf_ds",
                    offset=OffsetCfg(pos=(0.0, -0.045, 0.0144)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/root/th_ds",
                    offset=OffsetCfg(pos=(0.0, -0.055, -0.015)),
                )
            ],
        }
        for key in self.tf_target_frames.keys():
            tf_name = f"tf_{key}"
            if tf_name not in self.commands.motion.tf_asset_names:
                self.commands.motion.tf_asset_names.append(tf_name)

        self.commands.motion.actuated_joint_names = ACTUATED_JOINT_NAMES

        # Floating root: no wrist translation/rotation joints on the asset (see DexterousEnvCfg.EventCfg).
        # TODO: support wrist translation/rotation offset in the action term
        self.events = self.events.replace(
            robot_wrist_translation_joint_pos_offset=None,
            robot_wrist_rotation_joint_pos_offset=None,
        )

        self.actions.root_pose.body_name = "root"
        self.actions.root_pose.base_action_source = "motion_target"
        self.actions.root_pose.scale_pos = 1.0 * self.sim.dt * self.decimation
        self.actions.root_pose.scale_rot = 3.2 * self.sim.dt * self.decimation
        # Cartesian wrench PD on the floating root (world frame). Units: N/m, N·s/m, N·m/rad, N·m·s/rad.
        # Tuned for FREE_LEAP_RIGHT_HAND (gravity off): stiff enough to track motion demos without huge lag;
        # kd ~ 0.11·kp (pos) / 0.10·kp (rot) to limit overshoot at contact. Reduce if jittery.
        self.actions.root_pose.kp_pos = 300.0
        self.actions.root_pose.kd_pos = 30.0
        self.actions.root_pose.kp_rot = 3.0
        self.actions.root_pose.kd_rot = 0.3

        self.actions.joint_pos.base_action_source = "motion_target"
        self.actions.joint_pos.scale = {
            key: value * self.sim.dt * self.decimation
            for key, value in RELATIVE_ACTION_SCALE.items()
        }
        self.actions.joint_pos.joint_names = ACTUATED_JOINT_NAMES

        if self.include_contact_forces_in_critic_obs:
            self.observations.critic = self.observations.critic.replace(
                fingertip_contact_forces_w=ObsTerm(
                    func=mdp.contact_sensor_net_forces_w,
                    params={
                        "sensor_cfg": SceneEntityCfg(
                            "contact_forces",
                            body_names=["if_ds", "mf_ds", "rf_ds", "th_ds"],
                        ),
                    },
                ),
            )
