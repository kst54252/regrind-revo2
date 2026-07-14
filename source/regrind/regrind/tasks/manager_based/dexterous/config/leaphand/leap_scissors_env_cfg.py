import os
from isaaclab.utils import configclass
from isaaclab.envs.mdp import randomize_rigid_body_com
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from .leap_base_env_cfg import LeapHandBaseEnvCfg
from regrind.objects.scissors import SCISSORS_2X_3D_PRINTED_OBJECT_CFG as SCISSORS_OBJECT_CFG


@configclass
class LeapHandScissorsEnvCfg(LeapHandBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 9.0

        self.scene.object = SCISSORS_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

        self.commands.motion.object_pos_lower_bound = (-0.25, -0.25, 0.95)
        self.commands.motion.object_pos_upper_bound = (0.25, 0.25, 1.5)

        self.commands.motion.keypoint_paths: dict[str, str] = {
            "bottom": f"{os.environ['REGRIND_DATA_DIR']}/keypoints/scissors_bottom_25.npy",
            "top": f"{os.environ['REGRIND_DATA_DIR']}/keypoints/scissors_top_25.npy",
        }
        self.commands.motion.keypoint_body_prim_paths: dict[str, str] = {
            "bottom": "{ENV_REGEX_NS}/Object/bottom/bottom",
            "top": "{ENV_REGEX_NS}/Object/bottom/top",
        }
        self.commands.motion.keypoint_scale: float = 2.0
        self.commands.motion.object_joint_axis = (0.0, 0.0, -1.0)
        self.scene.setup_frame_transformers(
            tf_source_prim_path=self.tf_source_prim_path,
            tf_target_frames=self.tf_target_frames,
        )

        self.commands.motion.demo_path = f"{os.environ['REGRIND_DATA_DIR']}/arctic_demo/arctic_leap_scissors_2x/demo_interp_120fps.h5"
        self.commands.motion.demo_dt = 1.0 / 120.0
        self.commands.motion.retargeted_traj_path = f"{os.environ['REGRIND_DATA_DIR']}/retargeted_traj/leaphand/scissors_2x/retargeted_120fps.h5"
        self.commands.motion.augmentation_traj_dir = None
        self.commands.motion.traj_aug_enabled = True
        self.commands.motion.traj_aug_translation_lower_bound = (-0.05, -0.05, 0.0)
        self.commands.motion.traj_aug_translation_upper_bound = (0.05, 0.05, 0.0)
        self.commands.motion.traj_aug_yaw_range = (-0.5235987756, 0.5235987756)
        self.commands.motion.traj_aug_start = 0.281
        self.commands.motion.traj_aug_use = 0.4375

        self.events = self.events.replace(
            object_com=EventTerm(
                func=randomize_rigid_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "com_range": {"x": (-0.025, 0.025), "y": (-0.01, 0.01), "z": (-0.005, 0.005)},
                },
            )
        )


@configclass
class LeapHandScissorsEnvCfg_PLAY(LeapHandScissorsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 16
        # disable curriculum
        self.events = self.events.replace(
            curriculum_gravity=None,
            curriculum_random_push_object=None,
            curriculum_random_push_robot=None,
        )
        # disable RSI
        self.commands.motion.rsi_enabled = False
        # # use a large threshold to avoid early termination due to deviation
        self.terminations.object_deviation.params["threshold"] = 1.0
        self.terminations.object_rotation_error = None
        self.terminations.hit_table = None
        # enable desired gravity
        self.sim.gravity = (0.0, 0.0, -9.81)
        # closer viewpoint
        self.viewer.eye = (1., 1., 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        # enable marker visualization
        self.commands.motion.debug_vis = True
        self.sync_contact_sensor_sphere_debug_vis()
