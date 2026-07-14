import os
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from regrind.tasks.manager_based.dexterous.mdp import randomize_rigid_body_material
from regrind.tasks.manager_based.dexterous.mdp import randomize_rigid_body_scale

from .wuji_base_env_cfg import WujiHandBaseEnvCfg
from regrind.objects.screwdriver import SCREWDRIVER_OBJECT_CFG
from regrind.envs.events import randomize_rigid_object_com


@configclass
class WujiHandScrewdriverEnvCfg(WujiHandBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 9.0

        self.scene.object = SCREWDRIVER_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")
        self.scene.table.init_state.pos = (0.0, 0.0, 0.012 - self.scene.table.spawn.size[2] / 2)

        self.commands.motion.object_pos_lower_bound = (-0.40, -0.25, 0.0)
        self.commands.motion.object_pos_upper_bound = (0.10, 0.25, 0.50)

        self.commands.motion.keypoint_paths: dict[str, str] = {
            "bottom": f"{os.environ['REGRIND_DATA_DIR']}/keypoints/screwdriver_50.npy",
        }
        self.commands.motion.keypoint_body_prim_paths: dict[str, str] = {
            "bottom": "{ENV_REGEX_NS}/Object/bottom/bottom",
        }
        self.commands.motion.keypoint_scale: float = 1.0
        self.commands.motion.object_joint_axis = None

        # disable object joint observation for rigid object
        self.observations.policy.object_joint = None
        self.observations.critic.object_joint = None
        self.observations.critic.target_object_joint = None

        self.scene.setup_frame_transformers(
            tf_source_prim_path=self.tf_source_prim_path,
            tf_target_frames=self.tf_target_frames,
        )

        self.commands.motion.demo_path = f"{os.environ['REGRIND_DATA_DIR']}/zed_mocap_demo/screwdriver/demo_interp_120fps.h5"
        self.commands.motion.demo_dt = 1.0 / 120.0
        self.commands.motion.retargeted_traj_path = f"{os.environ['REGRIND_DATA_DIR']}/retargeted_traj/wujihand/screwdriver/retargeted_120fps.h5"
        self.commands.motion.augmentation_traj_dir = None
        self.commands.motion.traj_aug_enabled = True
        self.commands.motion.traj_aug_translation_lower_bound = (-0.05, -0.05, 0.0)
        self.commands.motion.traj_aug_translation_upper_bound = (0.05, 0.05, 0.0)
        self.commands.motion.traj_aug_yaw_range = (-0.5235987756, 0.5235987756)
        self.commands.motion.traj_aug_start = 0.135
        self.commands.motion.traj_aug_use = 0.292

        self.scene.replicate_physics = False  # to enable randomizing object scale
        self.events = self.events.replace(
            object_com=EventTerm(
                func=randomize_rigid_object_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "com_range": {"x": (-0.005, 0.005), "y": (-0.02, 0.02), "z": (-0.005, 0.005)},
                },
            ),
            object_physics_material=EventTerm(
                func=randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                    "static_friction_range": (0.5, 1.5),
                    "dynamic_friction_range": (0.5, 1.5),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 250,
                    "make_consistent": True,
                },
            ),
            object_scale=EventTerm(
                func=randomize_rigid_body_scale,
                mode="prestartup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "scale_range": {"x": (1.0, 1.05), "y": (1.0, 1.05), "z": (1.0, 1.0)},
                },
            )
        )


@configclass
class WujiHandScrewdriverEnvCfg_PLAY(WujiHandScrewdriverEnvCfg):
    enable_contact_sensor: bool = True

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
        self.viewer.eye = (1., -1., 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # enable marker visualization
        self.commands.motion.debug_vis = True
        self.sync_contact_sensor_sphere_debug_vis()
