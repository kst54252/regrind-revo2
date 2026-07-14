from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    normalize,
    sample_uniform,
)
from regrind.utils.math import euler_xyz_intrinsic_from_quat, quat_from_euler_xyz_intrinsic

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from regrind.data.utils import read_h5_to_dict
from regrind.data.arctic import QPOS_INDEX as DEMO_QPOS_INDEX_NUMPY
from regrind.data.arctic import QVEL_INDEX as DEMO_QVEL_INDEX_NUMPY
from regrind.data.arctic import QPOS_INDEX_WUJI, QVEL_INDEX_WUJI, QPOS_DIM_WUJI
from regrind.data.arctic import load_retargeted_traj, QPOS_DIM
from regrind.utils.motion_interp import rescale_motion
from regrind.utils.markers import ManoMarker, MANO_TO_MANUS_MAPPING


# Convert numpy arrays to torch tensors for indexing
DEMO_QPOS_INDEX = {k: torch.tensor(v, dtype=torch.long) for k, v in DEMO_QPOS_INDEX_NUMPY.items()}
DEMO_QVEL_INDEX = {k: torch.tensor(v, dtype=torch.long) for k, v in DEMO_QVEL_INDEX_NUMPY.items()}


class MotionLoader:
    """
    The class loads the motion data from the demo and retargeted trajectory.
    It rescales the motion data if necessary.
    """

    def __init__(
        self,
        demo_path: str,
        demo_dt: float,
        env_dt: float,
        env_episode_length: int,
        retargeted_traj_path: str,
        object_joint_axis: np.ndarray,
        augmentation_traj_dir: str | None = None,
        object_keypoints_local: dict[str, np.ndarray] = None,
        is_articulated_object: bool = True,
        device: str = "cuda:0"
    ):
        """
        Args:
            demo_path: path to the demo h5 file
            demo_dt: demo time step size in seconds (1/fps)
            env_dt: environment time step size in seconds.
            retargeted_traj_path: path to the retargeted trajectory file
            object_joint_axis: axis of the object joint.
            augmentation_traj_dir: path to the augmentation trajectory directory
            env_episode_length: number of environment timesteps in an episode.
                NOTE: The motion data will be resampled according to the environment episode length.
                      Make sure all demos have the same length and are properly aligned.
            object_keypoints_local: local keypoints of the object.
                Keys: "bottom", "top"
                Values: numpy array of shape (num_keypoints, 3)
            is_articulated_object: whether the object is articulated.
            device: device to use for the tensors
        """
        assert os.path.isfile(demo_path), f"Invalid demo path: {demo_path}"
        assert os.path.isfile(retargeted_traj_path), f"Invalid retargeted trajectory path: {retargeted_traj_path}"
        if augmentation_traj_dir is not None:
            assert os.path.isdir(augmentation_traj_dir), f"Invalid augmentation trajectory directory: {augmentation_traj_dir}"
        self.device = device
        self.demo_dt = demo_dt
        self.env_dt = env_dt
        self.env_episode_length = env_episode_length
        self.object_joint_axis = object_joint_axis
        self.object_keypoints_local = object_keypoints_local
        self.is_articulated_object = is_articulated_object
        # Set by first loaded retargeted trajectory (16-DoF Leap vs 20-DoF Wuji).
        self._layout_n_finger: int | None = None
        self.qpos_dim = QPOS_DIM
        self.demo_qpos_index = {k: v.clone() for k, v in DEMO_QPOS_INDEX.items()}
        self.demo_qvel_index = {k: v.clone() for k, v in DEMO_QVEL_INDEX.items()}
        self._load_data(demo_path, retargeted_traj_path, augmentation_traj_dir)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    def _load_data(self, demo_path: str, retargeted_traj_path: str, augmentation_traj_dir: str | None = None):

        # Load demo
        demo = read_h5_to_dict(demo_path)
        print(f"Loaded demo from {demo_path}.")
        if "meta" in demo:
            demo_meta = demo["meta"]
            print("-" * 50, "Demo Meta", "-" * 50)
            # print meta dict line by line
            for k, v in demo_meta.items():
                print(f"{k}: {v}")
            print("-" * 50, "End of Demo Meta", "-" * 50)
        if "qpos" in demo:
            original_demo_len = demo["qpos"].shape[0]
        elif "wrist_pos_world" in demo:
            original_demo_len = demo["wrist_pos_world"].shape[0]
        elif "mano_joint_coords" in demo:
            original_demo_len = demo["mano_joint_coords"].shape[0]
        else:
            raise ValueError("No qpos or wrist_pos_world found in demo to infer original demo length.")

        speed_factor = ((original_demo_len - 1) * self.demo_dt) / ((self.env_episode_length - 1) * self.env_dt)  # demo duration / env episode duration
        print(f"Original demo length: {original_demo_len}, demo dt: {self.demo_dt}, env episode length: {self.env_episode_length}, env dt: {self.env_dt}")
        print(f"Motion will be scaled by speed factor: {speed_factor}")
        if "mano_joint_coord" in demo:
            demo_mano_joint_coord = demo["mano_joint_coord"]  # (demo_length, 21, 3)
        elif "mano_joint_coords" in demo:
            demo_mano_joint_coord = demo["mano_joint_coords"]
        elif "manus_hand_keypoints_world" in demo:
            demo_manus_joint_coord = demo["manus_hand_keypoints_world"]  # (demo_length, 21, 3)
            indices = [MANO_TO_MANUS_MAPPING[i] for i in range(21)]
            demo_mano_joint_coord = demo_manus_joint_coord[:, indices, :]  # (demo_length, 21, 3)
        else:
            raise ValueError("No hand keypoints found in demo.")
        demo_mano_joint_coord, demo_mano_joint_coord_vel = rescale_motion(
            demo_mano_joint_coord, motion_type="position", speed_factor=speed_factor, dt=self.demo_dt, target_dt=self.env_dt
        )

        demo_qpos_list, demo_qvel_list, retargeted_hand_keypoints_list, object_keypoints_pos_list = [], [], [], []
        demo_length_list, accumulated_demo_length_list = [], []

        # Load retargeted trajectory of the original trajectory
        assert retargeted_traj_path is not None
        qpos, qvel, retargeted_hand_keypoints, object_keypoints_pos = self._load_retargeted_traj(
            retargeted_traj_path, speed_factor=speed_factor
        )
        demo_qpos_list.append(qpos)
        demo_qvel_list.append(qvel)
        retargeted_hand_keypoints_list.append(retargeted_hand_keypoints)
        object_keypoints_pos_list.append(object_keypoints_pos)
        demo_length_list.append(qpos.shape[0])
        accumulated_demo_length_list.append(qpos.shape[0])

        # Load augmentation trajectories
        if augmentation_traj_dir is not None:
            augmentation_traj_files = os.listdir(augmentation_traj_dir)
            for augmentation_traj_file in augmentation_traj_files:
                qpos, qvel, retargeted_hand_keypoints, object_keypoints_pos = self._load_retargeted_traj(
                    os.path.join(augmentation_traj_dir, augmentation_traj_file),
                    speed_factor=speed_factor,
                )
                demo_qpos_list.append(qpos)
                demo_qvel_list.append(qvel)
                retargeted_hand_keypoints_list.append(retargeted_hand_keypoints)
                object_keypoints_pos_list.append(object_keypoints_pos)
                demo_length_list.append(qpos.shape[0])
                accumulated_demo_length_list.append(accumulated_demo_length_list[-1] + qpos.shape[0])

        # Concatenate all trajectories (total_demo_len = sum(len(demo_qpos_list)))
        self.demo_qpos = torch.cat(demo_qpos_list, dim=0)  # (total_demo_len, qpos_dim)
        self.demo_qvel = torch.cat(demo_qvel_list, dim=0)  # (total_demo_len, qvel_dim)  # qvel_dim is the same as qpos_dim
        self.retargeted_hand_keypoints = torch.cat(retargeted_hand_keypoints_list, dim=0)  # (total_demo_len, num_hand_keypoints, 3)
        self.object_keypoints_pos = torch.cat(object_keypoints_pos_list, dim=0)  # (total_demo_len, num_keypoints, 3)
        # self.demo_lengths is a convenient tensor for getting the lengths / bounds of the demos.
        self.demo_lengths = torch.cat(
            [torch.full((len_i,), len_i, dtype=torch.long, device=self.device) for len_i in demo_length_list],
        )  # (total_demo_len,)
        self.demo_start_indices = torch.cat([
            torch.full((len_i,), start_i, dtype=torch.long, device=self.device)
            for len_i, start_i in zip(demo_length_list, [0] + accumulated_demo_length_list[:-1])
        ])  # (total_demo_len,)
        # TODO: for now we assume all demos have the same length. So we just repeat the mano joint coord for all demos.
        # NOTE: Also note that the mano joint coord is only available for the original demo.
        #       So they will be off for the augmentation demos and should not be used as e.g. tracking targets.
        self.demo_mano_joint_coord = torch.from_numpy(demo_mano_joint_coord).float().to(self.device)  # (original_demo_len, 21, 3)
        self.demo_mano_joint_coord = self.demo_mano_joint_coord.repeat(len(demo_qpos_list), 1, 1)  # (total_demo_len, 21, 3)

        self.total_demo_len = self.demo_qpos.shape[0]
        assert self.demo_qvel.shape[0] == self.total_demo_len
        assert self.retargeted_hand_keypoints.shape[0] == self.total_demo_len
        assert self.demo_lengths.shape[0] == self.total_demo_len
        assert self.demo_mano_joint_coord.shape[0] == self.total_demo_len
        assert self.demo_start_indices.shape[0] == self.total_demo_len
        assert self.object_keypoints_pos.shape[0] == self.total_demo_len
        print(f"Total demo length: {self.total_demo_len}")

    def _ensure_qpos_layout(self, n_finger: int) -> None:
        """Select qpos/qvel indexing for Leap (16) vs Wuji (20) finger DoFs."""
        if self._layout_n_finger is None:
            if n_finger == 16:
                self.qpos_dim = QPOS_DIM
                qnp, vnp = DEMO_QPOS_INDEX_NUMPY, DEMO_QVEL_INDEX_NUMPY
            elif n_finger == 20:
                self.qpos_dim = QPOS_DIM_WUJI
                qnp, vnp = QPOS_INDEX_WUJI, QVEL_INDEX_WUJI
            else:
                raise ValueError(
                    f"Unsupported robot_joints width {n_finger}; expected 16 (Leap) or 20 (Wuji)."
                )
            self._layout_n_finger = n_finger
            self.demo_qpos_index = {
                k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in qnp.items()
            }
            self.demo_qvel_index = {
                k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in vnp.items()
            }
        elif n_finger != self._layout_n_finger:
            raise ValueError(
                f"Inconsistent robot_joints width {n_finger} (expected {self._layout_n_finger} from earlier segment)."
            )

    def _load_retargeted_traj(self, retargeted_traj_path: str, speed_factor: float):
        retargeted_traj = load_retargeted_traj(retargeted_traj_path)
        assert retargeted_traj is not None
        if self.is_articulated_object and "object_joint" not in retargeted_traj:
            raise ValueError("Object joint not found in retargeted trajectory.")
        original_traj_len = retargeted_traj["object_pos"].shape[0]
        print(f"Loaded retargeted trajectory with {original_traj_len} steps from {retargeted_traj_path}.")

        n_finger = int(retargeted_traj["robot_joints"].shape[1])
        self._ensure_qpos_layout(n_finger)
        dq, dv = self.demo_qpos_index, self.demo_qvel_index

        qpos = torch.zeros((self.env_episode_length, self.qpos_dim), dtype=torch.float32, device=self.device)
        qvel = torch.zeros_like(qpos, dtype=torch.float32, device=self.device)

        obj_pos, obj_lin_vel = rescale_motion(
            retargeted_traj["object_pos"], "position", speed_factor, dt=self.demo_dt, target_dt=self.env_dt
        )
        obj_quat, obj_ang_vel = rescale_motion(
            retargeted_traj["object_quat"], "orientation", speed_factor, dt=self.demo_dt, target_dt=self.env_dt
        )
        if self.is_articulated_object:
            obj_joint, obj_joint_vel = rescale_motion(
                retargeted_traj["object_joint"], "joint", speed_factor, dt=self.demo_dt, target_dt=self.env_dt
            )
            # Ensure shape is (T, J) for consistent assignment into qpos/qvel.
            # Some trajectories store a single joint as (T,), while our indexing expects (T, 1).
            obj_joint = np.asarray(obj_joint)
            obj_joint_vel = np.asarray(obj_joint_vel)
            if obj_joint.ndim == 1:
                obj_joint = obj_joint[:, None]
            if obj_joint_vel.ndim == 1:
                obj_joint_vel = obj_joint_vel[:, None]
        else:
            obj_joint, obj_joint_vel = None, None
        hand_wrist_pos, hand_wrist_lin_vel = rescale_motion(
            retargeted_traj["robot_pos"], "position", speed_factor, dt=self.demo_dt, target_dt=self.env_dt
        )
        hand_wrist_quat, hand_wrist_ang_vel = rescale_motion(
            retargeted_traj["robot_quat"], "orientation", speed_factor, dt=self.demo_dt, target_dt=self.env_dt
        )
        hand_wrist_euler_XYZ = R.from_quat(hand_wrist_quat, scalar_first=True).as_euler("XYZ", degrees=False)
        finger_joints, finger_joints_vel = rescale_motion(
            retargeted_traj["robot_joints"], "joint", speed_factor, dt=self.demo_dt, target_dt=self.env_dt
        )
        qpos[:, dq["object_pos"]] = torch.from_numpy(obj_pos).float().to(self.device)
        qpos[:, dq["object_quat"]] = torch.from_numpy(obj_quat).float().to(self.device)
        if self.is_articulated_object:
            qpos[:, dq["object_joint"]] = torch.from_numpy(obj_joint).float().to(self.device)
        qpos[:, dq["hand_wrist_pos"]] = torch.from_numpy(hand_wrist_pos).float().to(self.device)
        qpos[:, dq["hand_wrist_rot"]] = torch.from_numpy(hand_wrist_euler_XYZ).float().to(self.device)
        qpos[:, dq["fingers"]] = torch.from_numpy(finger_joints).float().to(self.device)

        qvel[:, dv["object_pos"]] = torch.from_numpy(obj_lin_vel).float().to(self.device)
        qvel[:, dv["object_root_angvel"]] = torch.from_numpy(obj_ang_vel).float().to(self.device)
        if self.is_articulated_object:
            qvel[:, dv["object_joint"]] = torch.from_numpy(obj_joint_vel).float().to(self.device)
        qvel[:, dv["hand_wrist_pos"]] = torch.from_numpy(hand_wrist_lin_vel).float().to(self.device)
        qvel[:, dv["hand_wrist_rot"]] = torch.from_numpy(hand_wrist_ang_vel).float().to(self.device)
        qvel[:, dv["fingers"]] = torch.from_numpy(finger_joints_vel).float().to(self.device)

        retargeted_hand_keypoints, _ = rescale_motion(
            retargeted_traj["robot_keypoints"], "position", speed_factor, dt=self.demo_dt, target_dt=self.env_dt
        )
        retargeted_hand_keypoints = torch.from_numpy(retargeted_hand_keypoints).float().to(self.device)  # (env_episode_length, num_hand_keypoints, 3)
        # NOTE: the first 4 hand keypoints are the fingertips (see the retargeter's keypoint ordering).

        object_keypoints_pos = self._get_object_keypoints_pos(
            obj_pos, obj_quat, obj_joint,
            self.object_joint_axis,
            self.object_keypoints_local["bottom"],
            self.object_keypoints_local.get("top", None),
        )
        object_keypoints_pos = torch.from_numpy(object_keypoints_pos).float().to(self.device)  # (env_episode_length, num_keypoints, 3)

        return qpos, qvel, retargeted_hand_keypoints, object_keypoints_pos

    def _get_object_keypoints_pos(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray | None,
        joint_axis: np.ndarray | None,
        bottom_keypoints_local: np.ndarray,
        top_keypoints_local: np.ndarray | None,
    ) -> np.ndarray:

        root_pos = np.asarray(root_pos)
        root_quat = np.asarray(root_quat)
        bottom_local = np.asarray(bottom_keypoints_local)

        if root_pos.ndim == 1:
            root_pos = root_pos[None, ...]  # (B,3)
        if root_quat.ndim == 1:
            root_quat = root_quat[None, ...]  # (B,4) scalar-first [w,x,y,z]

        # Bottom rotation (world-from-bottom), applied per batch
        Rb = R.from_quat(root_quat, scalar_first=True)  # (B,) rotations
        B = root_pos.shape[0]
        bottom_world = np.stack(
            [Rb[i].apply(bottom_local) + root_pos[i][None, :] for i in range(B)],
            axis=0,
        )  # (B, Nb, 3)

        if joint_pos is None:
            return bottom_world

        joint_pos = np.asarray(joint_pos)
        joint_axis = np.asarray(joint_axis)
        top_local = np.asarray(top_keypoints_local)

        if joint_pos.ndim == 0:
            joint_pos = joint_pos[None, ...]  # (B,)
        if joint_pos.ndim == 2 and joint_pos.shape[-1] == 1:
            joint_pos = np.squeeze(joint_pos, axis=-1)  # (B,)
        if joint_axis.ndim == 1:
            joint_axis = np.broadcast_to(joint_axis[None, ...], (B, 3))  # (3,) -> (B,3)

        # Relative rotation about joint_axis by joint_pos (in bottom frame)
        axis_norm = np.linalg.norm(joint_axis, axis=1, keepdims=True)
        axis_norm = np.clip(axis_norm, 1e-12, None)
        axis_unit = joint_axis / axis_norm                          # (B,3)
        rotvec = axis_unit * joint_pos[:, None]                     # (B,3) axis-angle as rotvec
        Rrel = R.from_rotvec(rotvec)                                  # (B,) rotations

        # World-from-top rotation: R_top = Rb * Rrel (apply Rrel, then Rb), per batch
        Rtop = Rb * Rrel  # (B,) rotations
        top_world = np.stack(
            [Rtop[i].apply(top_local) + root_pos[i][None, :] for i in range(B)],
            axis=0,
        )  # (B, Nt, 3)

        # Concatenate bottom and top
        return np.concatenate([bottom_world, top_world], axis=1)     # (B,Nb+Nt,3)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object: Articulation = env.scene[cfg.object_asset_name]
        self.robot: Articulation = env.scene[cfg.robot_asset_name]

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))

        self.tfs: dict[str, FrameTransformer] = {tf_name: env.scene[tf_name] for tf_name in cfg.tf_asset_names}

        object_keypoints_local = {
            key: np.array(np.load(path), dtype=np.float32) * cfg.keypoint_scale for key, path in cfg.keypoint_paths.items()
        }
        self.object_keypoints_local = object_keypoints_local
        self._object_keypoints_local_torch = {
            key: torch.from_numpy(value).float().to(self.device) for key, value in object_keypoints_local.items()
        }
        self._object_joint_axis = torch.tensor(self.cfg.object_joint_axis, dtype=torch.float32, device=self.device) if self.cfg.object_joint_axis is not None else None
        self.motion: MotionLoader = MotionLoader(
            demo_path=self.cfg.demo_path,
            demo_dt=self.cfg.demo_dt,
            env_dt=env.step_dt,
            env_episode_length=env.max_episode_length,
            retargeted_traj_path=self.cfg.retargeted_traj_path,
            augmentation_traj_dir=self.cfg.augmentation_traj_dir,
            object_joint_axis=self.cfg.object_joint_axis,
            object_keypoints_local=object_keypoints_local,
            is_articulated_object=isinstance(self.object, Articulation),
            device=self.device
        )
        # If debug visuals were requested during base init, defer marker creation until motion exists.
        if self.cfg.debug_vis and self.cfg.debug_vis_keypoints and getattr(self, "object_keypoint_markers", None) is None:
            self.object_keypoint_markers = self._create_object_keypoint_markers()

        self._contact_force_markers: VisualizationMarkers | None = None

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.bin_count = int(self.motion.total_demo_len // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self._object_pos_lower_bound = torch.tensor(self.cfg.object_pos_lower_bound, device=self.device)
        self._object_pos_upper_bound = torch.tensor(self.cfg.object_pos_upper_bound, device=self.device)
        self._traj_aug_pos_lower = torch.tensor(
            self.cfg.traj_aug_translation_lower_bound, dtype=torch.float32, device=self.device
        )
        self._traj_aug_pos_upper = torch.tensor(
            self.cfg.traj_aug_translation_upper_bound, dtype=torch.float32, device=self.device
        )
        self._traj_aug_yaw_range = torch.tensor(
            self.cfg.traj_aug_yaw_range, dtype=torch.float32, device=self.device
        )
        self._traj_aug_delta_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._traj_aug_delta_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._traj_aug_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._traj_aug_use_step = torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_quat"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_joint"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_keypoints_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_fingertips_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_hand_wrist_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_hand_wrist_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_hand_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_hand_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["gravity_z"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.phi.reshape(self.num_envs, -1)

    @property
    def timesteps_in_current_demo(self) -> torch.Tensor:
        return self.time_steps - self.motion.demo_start_indices[self.time_steps]

    @property
    def current_demo_length(self) -> torch.Tensor:
        return self.motion.demo_lengths[self.time_steps]

    @property
    def phi(self) -> torch.Tensor:
        return self.timesteps_in_current_demo / self.current_demo_length

    @property
    def object_pos_lower_bound(self) -> torch.Tensor:
        return self._object_pos_lower_bound

    @property
    def object_pos_upper_bound(self) -> torch.Tensor:
        return self._object_pos_upper_bound

    # -- Target values --
    def _target_qpos(self, time_steps: torch.Tensor, key: str) -> torch.Tensor:
        return self.motion.demo_qpos[time_steps][:, self.motion.demo_qpos_index[key]]

    def _target_qvel(self, time_steps: torch.Tensor, key: str) -> torch.Tensor:
        return self.motion.demo_qvel[time_steps][:, self.motion.demo_qvel_index[key]]

    def _traj_aug_weight(self, time_steps: torch.Tensor) -> torch.Tensor:
        if not self.cfg.traj_aug_enabled:
            return torch.zeros((time_steps.shape[0],), dtype=torch.float32, device=self.device)

        demo_start = self.motion.demo_start_indices[time_steps]
        t_in_demo = (time_steps - demo_start).to(torch.float32)
        start_step = self._traj_aug_start_step.to(torch.float32)
        use_step = self._traj_aug_use_step.to(torch.float32)
        blend_len = (use_step - start_step).clamp_min(1.0)
        phase = (t_in_demo - start_step) / blend_len
        return torch.where(t_in_demo < start_step, 1.0, (1.0 - phase).clamp(0.0, 1.0))

    def _traj_aug_delta_quat(self, time_steps: torch.Tensor) -> torch.Tensor:
        weight = self._traj_aug_weight(time_steps)
        yaw = self._traj_aug_delta_yaw * weight
        axis = torch.zeros((time_steps.shape[0], 3), dtype=torch.float32, device=self.device)
        axis[:, 2] = 1.0
        return quat_from_angle_axis(yaw, axis)

    def _target_object_pos_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        object_pos = self._target_qpos(time_steps, "object_pos")
        if not self.cfg.traj_aug_enabled:
            return object_pos
        weight = self._traj_aug_weight(time_steps)
        return object_pos + self._traj_aug_delta_pos * weight.unsqueeze(-1)

    def _target_object_quat_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        object_quat = self._target_qpos(time_steps, "object_quat")
        if not self.cfg.traj_aug_enabled:
            return object_quat
        aug_quat = quat_mul(self._traj_aug_delta_quat(time_steps), object_quat)
        return aug_quat / (aug_quat.norm(dim=-1, keepdim=True) + 1e-8)

    def _apply_traj_aug_to_points(
        self,
        points: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.traj_aug_enabled or not self.cfg.traj_aug_apply_to_hand:
            return points

        object_pos = self._target_qpos(time_steps, "object_pos")
        weight = self._traj_aug_weight(time_steps)
        delta_pos = self._traj_aug_delta_pos * weight.unsqueeze(-1)
        delta_quat = self._traj_aug_delta_quat(time_steps)
        if points.ndim == 2:
            return quat_apply(delta_quat, points - object_pos) + object_pos + delta_pos
        if points.ndim == 3:
            delta_quat = delta_quat.unsqueeze(1).expand(-1, points.shape[1], -1)
            object_pos = object_pos.unsqueeze(1)
            delta_pos = delta_pos.unsqueeze(1)
            return quat_apply(delta_quat, points - object_pos) + object_pos + delta_pos
        raise ValueError(f"Unsupported points shape for trajectory augmentation: {points.shape}")

    def _target_hand_wrist_pos_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        hand_wrist_pos = self._target_qpos(time_steps, "hand_wrist_pos")
        return self._apply_traj_aug_to_points(hand_wrist_pos, time_steps)

    def _target_hand_wrist_quat_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        wrist_rot = self._target_qpos(time_steps, "hand_wrist_rot")
        wrist_quat = quat_from_euler_xyz_intrinsic(wrist_rot[:, 0], wrist_rot[:, 1], wrist_rot[:, 2])
        if self.cfg.traj_aug_enabled and self.cfg.traj_aug_apply_to_hand:
            wrist_quat = quat_mul(self._traj_aug_delta_quat(time_steps), wrist_quat)
            wrist_quat = wrist_quat / (wrist_quat.norm(dim=-1, keepdim=True) + 1e-8)
        return wrist_quat

    def _zero_initial_demo_velocity(self, velocity: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        is_first_step = (time_steps - self.motion.demo_start_indices[time_steps]) == 0
        if velocity.ndim == 2:
            return torch.where(is_first_step.unsqueeze(-1), torch.zeros_like(velocity), velocity)
        if velocity.ndim == 3:
            return torch.where(is_first_step[:, None, None], torch.zeros_like(velocity), velocity)
        raise ValueError(f"Unsupported velocity shape: {velocity.shape}")

    def _finite_difference_time_steps(self, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        demo_start = self.motion.demo_start_indices[time_steps]
        demo_end = demo_start + self.motion.demo_lengths[time_steps] - 1
        prev_steps = torch.maximum(time_steps - 1, demo_start)
        next_steps = torch.minimum(time_steps + 1, demo_end)
        dt = (next_steps - prev_steps).clamp_min(1).to(torch.float32) * self._env.step_dt
        return prev_steps, next_steps, dt

    def _finite_difference_position(self, position_fn, time_steps: torch.Tensor) -> torch.Tensor:
        prev_steps, next_steps, dt = self._finite_difference_time_steps(time_steps)
        velocity = (position_fn(next_steps) - position_fn(prev_steps)) / dt.unsqueeze(-1)
        return self._zero_initial_demo_velocity(velocity, time_steps)

    def _finite_difference_quat(self, quat_fn, time_steps: torch.Tensor) -> torch.Tensor:
        prev_steps, next_steps, dt = self._finite_difference_time_steps(time_steps)
        quat_diff = quat_mul(quat_fn(next_steps), quat_inv(quat_fn(prev_steps)))
        quat_diff = torch.where(quat_diff[:, 0:1] < 0.0, -quat_diff, quat_diff)
        angular_velocity = axis_angle_from_quat(quat_diff) / dt.unsqueeze(-1)
        return self._zero_initial_demo_velocity(angular_velocity, time_steps)

    @property
    def target_object_pos(self) -> torch.Tensor:
        return self._target_object_pos_at(self.time_steps)

    @property
    def target_object_quat(self) -> torch.Tensor:
        return self._target_object_quat_at(self.time_steps)

    @property
    def target_object_joint(self) -> torch.Tensor:
        return self._target_qpos(self.time_steps, "object_joint")

    @property
    def target_object_lin_vel(self) -> torch.Tensor:
        return self._finite_difference_position(self._target_object_pos_at, self.time_steps)

    @property
    def target_object_ang_vel(self) -> torch.Tensor:
        return self._finite_difference_quat(self._target_object_quat_at, self.time_steps)

    @property
    def target_object_joint_vel(self) -> torch.Tensor:
        return self._zero_initial_demo_velocity(self._target_qvel(self.time_steps, "object_joint"), self.time_steps)

    @property
    def target_object_keypoints_pos(self) -> torch.Tensor:
        if not self.cfg.traj_aug_enabled:
            return self.motion.object_keypoints_pos[self.time_steps]
        object_joint = self.target_object_joint if self._object_joint_axis is not None else None
        return self._get_object_keypoints_pos_torch(
            self.target_object_pos,
            self.target_object_quat,
            object_joint,
        )

    @property
    def target_hand_wrist_pos(self) -> torch.Tensor:
        return self._target_hand_wrist_pos_at(self.time_steps)

    @property
    def target_hand_wrist_rot(self) -> torch.Tensor:
        if not self.cfg.traj_aug_enabled or not self.cfg.traj_aug_apply_to_hand:
            return self._target_qpos(self.time_steps, "hand_wrist_rot")
        return euler_xyz_intrinsic_from_quat(self.target_hand_wrist_quat)

    @property
    def target_hand_wrist_quat(self) -> torch.Tensor:
        return self._target_hand_wrist_quat_at(self.time_steps)

    @property
    def target_hand_wrist_lin_vel(self) -> torch.Tensor:
        return self._finite_difference_position(self._target_hand_wrist_pos_at, self.time_steps)

    @property
    def target_hand_wrist_ang_vel(self) -> torch.Tensor:
        return self._finite_difference_quat(self._target_hand_wrist_quat_at, self.time_steps)

    @property
    def target_hand_joint_pos(self) -> torch.Tensor:
        return self._target_qpos(self.time_steps, "fingers")

    @property
    def target_hand_joint_vel(self) -> torch.Tensor:
        return self._zero_initial_demo_velocity(self._target_qvel(self.time_steps, "fingers"), self.time_steps)

    @property
    def target_mano_joint_coord(self) -> torch.Tensor:
        return self.motion.demo_mano_joint_coord[self.time_steps]

    @property
    def target_fingertips_pos(self) -> torch.Tensor:
        fingertips = self.motion.retargeted_hand_keypoints[self.time_steps][:, [1, 2, 3, 0], :]
        return self._apply_traj_aug_to_points(fingertips, self.time_steps)  # TODO: hardcoded indices for now

    # -- Current values --
    @property
    def current_object_pos(self) -> torch.Tensor:
        return self.object.data.root_pos_w - self._env.scene.env_origins

    @property
    def current_object_quat(self) -> torch.Tensor:
        return self.object.data.root_quat_w

    @property
    def current_object_joint(self) -> torch.Tensor | None:
        if isinstance(self.object, Articulation):
            return self.object.data.joint_pos
        else:
            return None

    @property
    def current_object_lin_vel(self) -> torch.Tensor:
        return self.object.data.root_lin_vel_w

    @property
    def current_object_ang_vel(self) -> torch.Tensor:
        return self.object.data.root_ang_vel_w

    def _get_object_keypoints_pos_torch(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        if root_pos.ndim == 1:
            root_pos = root_pos.unsqueeze(0)
        if root_quat.ndim == 1:
            root_quat = root_quat.unsqueeze(0)

        batch_size = root_pos.shape[0]

        bottom_local = self._object_keypoints_local_torch["bottom"].to(root_pos.dtype)
        bottom_local = bottom_local.unsqueeze(0).expand(batch_size, -1, -1)
        bottom_quat = root_quat.unsqueeze(1).expand(-1, bottom_local.shape[1], -1)
        bottom_world = quat_apply(bottom_quat, bottom_local) + root_pos.unsqueeze(1)

        if joint_pos is None:
            return bottom_world

        if joint_pos.ndim == 0:
            joint_pos = joint_pos.unsqueeze(0)
        if joint_pos.ndim == 2 and joint_pos.shape[-1] == 1:
            joint_pos = joint_pos.squeeze(-1)

        joint_axis = self._object_joint_axis
        if joint_axis.ndim == 1:
            joint_axis = joint_axis.unsqueeze(0).expand(batch_size, -1)

        axis_unit = normalize(joint_axis)
        rel_quat = quat_from_angle_axis(joint_pos, axis_unit)
        top_quat = quat_mul(root_quat, rel_quat)

        top_local = self._object_keypoints_local_torch["top"].to(root_pos.dtype)
        top_local = top_local.unsqueeze(0).expand(batch_size, -1, -1)
        top_quat = top_quat.unsqueeze(1).expand(-1, top_local.shape[1], -1)
        top_world = quat_apply(top_quat, top_local) + root_pos.unsqueeze(1)

        return torch.cat([bottom_world, top_world], dim=1)

    @property
    def current_object_keypoints_pos(self) -> torch.Tensor:
        return self._get_object_keypoints_pos_torch(
            self.current_object_pos,
            self.current_object_quat,
            self.current_object_joint,
        )

    @property
    def current_hand_wrist_pos(self) -> torch.Tensor:
        return self.robot.data.root_pos_w - self._env.scene.env_origins

    @property
    def current_hand_wrist_quat(self) -> torch.Tensor:
        return self.robot.data.root_quat_w

    @property
    def current_hand_wrist_rot(self) -> torch.Tensor:
        return euler_xyz_intrinsic_from_quat(self.current_hand_wrist_quat)

    @property
    def current_fingertips_pos(self) -> torch.Tensor:
        # World positions minus env origin (same convention as :meth:`current_object_pos`), not robot-root frame.
        pos_w = self.tfs["tf_fingertips"].data.target_pos_w
        return pos_w - self._env.scene.env_origins.unsqueeze(1)

    @property
    def current_hand_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.actuated_dof_indices]

    @property
    def current_hand_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.actuated_dof_indices]

    @property
    def default_hand_joint_pos(self) -> torch.Tensor:
        return self.robot.data.default_joint_pos[:, self.actuated_dof_indices]

    def _update_metrics(self):
        self.metrics["error_object_pos"] = torch.norm(self.target_object_pos - self.current_object_pos, p=2, dim=-1)
        self.metrics["error_object_quat"] = quat_error_magnitude(self.target_object_quat, self.current_object_quat)
        if self.current_object_joint is not None:
            self.metrics["error_object_joint"] = torch.norm(self.target_object_joint - self.current_object_joint, p=2, dim=-1)
        # Reward terms compute and log the hand/object tracking errors.
        self.metrics["error_hand_joint_pos"] = torch.norm(self.target_hand_joint_pos - self.current_hand_joint_pos, dim=-1)
        self.metrics["error_hand_joint_vel"] = torch.norm(self.target_hand_joint_vel - self.current_hand_joint_vel, dim=-1)

        gravity_z = self._env.sim.physics_sim_view.get_gravity()[2]
        self.metrics["gravity_z"].fill_(gravity_z)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self.cfg.adaptive_sampling_enabled:
            terminated = self._env.termination_manager.terminated[env_ids]
            demo_end_reached = self._env.termination_manager.get_term("demo_end_reached")[env_ids]
            episode_failed = terminated * (~demo_end_reached)
            if torch.any(episode_failed):
                current_bin_index = torch.clamp(
                    (self.time_steps * self.bin_count) // max(self.motion.total_demo_len, 1), 0, self.bin_count - 1
                )
                fail_bins = current_bin_index[env_ids][episode_failed]
                self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

            # Normalize failure counts to a distribution, then mix with uniform.
            # This makes the failure signal independent of num_envs and episode length.
            failure_total = self.bin_failed_count.sum()
            if failure_total > 1e-8:
                failure_dist = self.bin_failed_count / failure_total
            else:
                failure_dist = torch.full_like(self.bin_failed_count, 1.0 / self.bin_count)

            sampling_probabilities = (
                (1.0 - self.cfg.adaptive_uniform_ratio) * failure_dist
                + self.cfg.adaptive_uniform_ratio / self.bin_count
            )

            if self.cfg.adaptive_kernel_size > 1:
                sampling_probabilities = torch.nn.functional.pad(
                    sampling_probabilities.unsqueeze(0).unsqueeze(0),
                    (0, self.cfg.adaptive_kernel_size - 1),
                    mode="replicate",
                )
                sampling_probabilities = torch.nn.functional.conv1d(
                    sampling_probabilities, self.kernel.view(1, 1, -1)
                ).view(-1)

            sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        else:
            sampling_probabilities = torch.full(
                (self.bin_count,), 1.0 / float(self.bin_count), device=self.device
            )

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        sampled_timesteps = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.total_demo_len - 1)
        ).long()

        if not self.cfg.rsi_enabled:
            sampled_timesteps = self.motion.demo_start_indices[self.time_steps[env_ids]]

        self.time_steps[env_ids] = sampled_timesteps

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

        try:
            import wandb
            import numpy as np

            if wandb.run is not None:
                # NOTE: assume all demos have the same length
                demo_len = self.motion.demo_lengths[0]
                assert all(self.motion.demo_lengths == demo_len), "All demos must have the same length"
                wandb.log(
                    {
                        "Metrics/motion/sampled_timesteps": wandb.Histogram(
                            (sampled_timesteps % demo_len).cpu().numpy(),
                            num_bins=demo_len,
                        ),
                        "Metrics/motion/sampling_bin_probs": wandb.Histogram(
                            np_histogram=(
                                sampling_probabilities.cpu().numpy(),
                                np.linspace(0, self.bin_count, self.bin_count + 1),
                            ),
                            num_bins=self.bin_count,
                        ),
                    },
                    commit=False,
                )
        except ImportError:
            pass

    def _sample_traj_aug(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        self._traj_aug_delta_pos[env_ids] = 0.0
        self._traj_aug_delta_yaw[env_ids] = 0.0
        self._traj_aug_start_step[env_ids] = 0
        self._traj_aug_use_step[env_ids] = 1
        if not self.cfg.traj_aug_enabled:
            return

        num_resets = len(env_ids)
        self._traj_aug_delta_pos[env_ids] = sample_uniform(
            self._traj_aug_pos_lower,
            self._traj_aug_pos_upper,
            (num_resets, 3),
            device=self.device,
        )
        self._traj_aug_delta_yaw[env_ids] = sample_uniform(
            self._traj_aug_yaw_range[0],
            self._traj_aug_yaw_range[1],
            (num_resets,),
            device=self.device,
        )

        demo_length = self.motion.demo_lengths[self.time_steps[env_ids]]
        max_offset = (demo_length - 1).clamp(min=1)
        start_scalar = min(max(float(self.cfg.traj_aug_start), 0.0), 1.0)
        use_scalar = min(max(float(self.cfg.traj_aug_use), 0.0), 1.0)
        start_step = (start_scalar * max_offset.to(torch.float32)).long()
        start_step = torch.minimum(start_step, (max_offset - 1).clamp(min=0))
        use_step = (use_scalar * max_offset.to(torch.float32)).long()
        use_step = torch.maximum(use_step, start_step + 1)
        use_step = torch.minimum(use_step, max_offset)

        self._traj_aug_start_step[env_ids] = start_step
        self._traj_aug_use_step[env_ids] = use_step

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        self._sample_traj_aug(env_ids)

        object_pos = self.target_object_pos.clone()
        object_quat = self.target_object_quat.clone()
        object_joint = self.target_object_joint.clone()
        object_lin_vel = self.target_object_lin_vel.clone()
        object_ang_vel = self.target_object_ang_vel.clone()
        object_joint_vel = self.target_object_joint_vel.clone()

        if self.cfg.enable_reset_perturbation:
            # Apply random perturbation to object pose (same scale as observation noise).
            num_resets = len(env_ids)
            object_pos[env_ids] += sample_uniform(-0.002, 0.002, (num_resets, 3), device=self.device)

            rand_axis = normalize(torch.randn((num_resets, 3), device=self.device))
            rand_angle = sample_uniform(-0.02, 0.02, (num_resets,), device=self.device)
            delta_quat = quat_from_angle_axis(rand_angle, rand_axis)
            object_quat[env_ids] = quat_mul(delta_quat, object_quat[env_ids])
            object_quat[env_ids] = object_quat[env_ids] / (object_quat[env_ids].norm(dim=-1, keepdim=True) + 1e-8)

            if object_joint is not None:
                object_joint[env_ids] += sample_uniform(-0.02, 0.02, object_joint[env_ids].shape, device=self.device)

        self.object.write_root_state_to_sim(
            torch.cat([
                object_pos[env_ids] + self._env.scene.env_origins[env_ids],
                object_quat[env_ids],
                object_lin_vel[env_ids],
                object_ang_vel[env_ids],
            ], dim=-1),
            env_ids=env_ids
        )
        if isinstance(self.object, Articulation):
            self.object.write_joint_state_to_sim(object_joint[env_ids], object_joint_vel[env_ids], env_ids=env_ids)

        hand_pos = self.target_hand_wrist_pos.clone()
        hand_quat = self.target_hand_wrist_quat.clone()
        hand_lin_vel = self.target_hand_wrist_lin_vel.clone()
        hand_ang_vel = self.target_hand_wrist_ang_vel.clone()

        joint_pos = self.target_hand_joint_pos.clone()
        joint_vel = self.target_hand_joint_vel.clone()

        if self.cfg.enable_reset_perturbation:
            num_resets = len(env_ids)
            hand_pos[env_ids] += sample_uniform(-0.002, 0.002, (num_resets, 3), device=self.device)
            rand_axis = normalize(torch.randn((num_resets, 3), device=self.device))
            rand_angle = sample_uniform(-0.02, 0.02, (num_resets,), device=self.device)
            delta_quat = quat_from_angle_axis(rand_angle, rand_axis)
            hand_quat[env_ids] = quat_mul(delta_quat, hand_quat[env_ids])
            hand_quat[env_ids] = hand_quat[env_ids] / (hand_quat[env_ids].norm(dim=-1, keepdim=True) + 1e-8)

            hand_noise = sample_uniform(-0.02, 0.02, joint_pos[env_ids].shape, device=self.device)
            joint_pos[env_ids] += hand_noise

        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    hand_pos[env_ids] + self._env.scene.env_origins[env_ids],
                    hand_quat[env_ids],
                    hand_lin_vel[env_ids],
                    hand_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

        dof_ids = torch.tensor(self.actuated_dof_indices, device=self.device, dtype=torch.long)
        soft_lo = self.robot.data.soft_joint_pos_limits[env_ids][:, dof_ids, 0]
        soft_hi = self.robot.data.soft_joint_pos_limits[env_ids][:, dof_ids, 1]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], soft_lo, soft_hi)
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], joint_ids=self.actuated_dof_indices, env_ids=env_ids
        )

    def _update_command(self):
        # Prevent time_steps from stepping past the current demo end.
        prev_steps = self.time_steps
        self.time_steps = self.time_steps + 1
        demo_start = self.motion.demo_start_indices[prev_steps]
        demo_end = demo_start + self.motion.demo_lengths[prev_steps] - 1
        self.time_steps = torch.minimum(self.time_steps, demo_end)
        # End-of-demo resets are handled by termination terms.

        # Scale alpha so the EMA half-life is measured in episodes, not steps.
        # With ~num_envs/max_episode_length resets per step, adaptive_alpha now
        # controls: half_life_episodes ≈ ln(2) / adaptive_alpha.
        resets_per_step = self.num_envs / self._env.max_episode_length
        effective_alpha = min(self.cfg.adaptive_alpha * resets_per_step, 1.0)
        self.bin_failed_count = (
            effective_alpha * self._current_bin_failed + (1 - effective_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            self.mano_marker = ManoMarker(
                prim_path="/World/Visuals/mano_marker",
                device=self.device,
                scale=self.cfg.debug_vis_marker_scale,
            )
            if self.cfg.debug_vis_keypoints and hasattr(self, "motion"):
                # Motion may not be initialized yet when base class calls this.
                self.object_keypoint_markers = self._create_object_keypoint_markers()
            else:
                self.object_keypoint_markers = None

        else:
            self.mano_marker = None
            self.object_keypoint_markers = None
            self._contact_force_markers = None

    def _create_contact_force_markers(self, num_bodies: int) -> VisualizationMarkers:
        s = self.cfg.debug_vis_marker_scale
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/contact_force_arrows",
            markers={
                f"arrow_{i}": sim_utils.CylinderCfg(
                    radius=0.0025 * s,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.15, 0.05), roughness=0.35, metallic=0.0
                    ),
                )
                for i in range(num_bodies)
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _create_object_keypoint_markers(self) -> VisualizationMarkers:
        num_keypoints = self.motion.object_keypoints_pos.shape[1]
        s = self.cfg.debug_vis_marker_scale
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/target_object_keypoints",
            markers={
                **{
                    f"target_keypoint_{i}": sim_utils.SphereCfg(
                        radius=0.006 * s,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2), roughness=1.0),
                    )
                    for i in range(num_keypoints)
                },
                **{
                    f"current_keypoint_{i}": sim_utils.SphereCfg(
                        radius=0.006 * s,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2), roughness=1.0),
                    )
                    for i in range(num_keypoints)
                },
                **{
                    f"keypoint_line_{i}": sim_utils.CylinderCfg(
                        radius=0.002 * s,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.2), roughness=1.0),
                    )
                    for i in range(num_keypoints)
                },
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        if self.mano_marker is not None:
            self.mano_marker.visualize(self.target_mano_joint_coord + self._env.scene.env_origins.unsqueeze(1).expand(self.num_envs, -1, -1))

        if self.object_keypoint_markers is not None:
            env_origins = self._env.scene.env_origins.unsqueeze(1)
            target_keypoints_w = self.target_object_keypoints_pos + env_origins
            current_keypoints_w = self.current_object_keypoints_pos + env_origins

            num_keypoints = target_keypoints_w.shape[1]
            num_envs = self.num_envs
            flat_count = num_envs * num_keypoints

            line_pos, line_quat, line_length = self._get_connecting_lines(
                start_pos=current_keypoints_w.reshape(-1, 3),
                end_pos=target_keypoints_w.reshape(-1, 3),
            )

            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                2 * flat_count, 1
            )
            marker_indices = torch.cat(
                [
                    torch.arange(num_keypoints, device=self.device).repeat(num_envs),
                    (torch.arange(num_keypoints, device=self.device) + num_keypoints).repeat(num_envs),
                    (torch.arange(num_keypoints, device=self.device) + 2 * num_keypoints).repeat(num_envs),
                ],
                dim=0,
            )
            marker_scales = torch.ones((2 * flat_count + line_pos.size(0), 3), device=self.device)
            marker_scales[-line_length.size(0) :, -1] = line_length

            self.object_keypoint_markers.visualize(
                translations=torch.cat(
                    [target_keypoints_w.reshape(-1, 3), current_keypoints_w.reshape(-1, 3), line_pos], dim=0
                ),
                orientations=torch.cat([default_quat, line_quat], dim=0),
                scales=marker_scales,
                marker_indices=marker_indices,
            )

        if self.cfg.debug_vis_contact_force_arrows and "contact_forces" in self._env.scene.sensors:
            cs = self._env.scene.sensors["contact_forces"]
            if not cs.is_initialized or cs.body_physx_view is None:
                return
            n_bodies = cs.num_bodies
            if self._contact_force_markers is None:
                self._contact_force_markers = self._create_contact_force_markers(n_bodies)

            _ = cs.data.net_forces_w
            pose = cs.body_physx_view.get_transforms()
            origins = pose.view(self.num_envs, n_bodies, 7)[:, :, :3]
            forces = cs.data.net_forces_w
            mag = torch.norm(forces, dim=-1)
            scale = self.cfg.contact_force_arrow_scale
            min_l = self.cfg.contact_force_arrow_min_length
            max_l = self.cfg.contact_force_arrow_max_length
            length = (mag * scale).clamp(min=min_l, max=max_l)
            default_up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=forces.dtype).expand_as(forces)
            dir_unit = torch.where(
                mag.unsqueeze(-1) > 1.0e-5,
                forces / mag.unsqueeze(-1).clamp(min=1.0e-9),
                default_up,
            )
            scaled = dir_unit * length.unsqueeze(-1)
            ends = origins + scaled
            # Avoid zero-length segments (degenerate quats in _get_connecting_lines) when force ~ 0.
            tiny = torch.tensor([0.0, 0.0, 2.0e-4], device=self.device, dtype=origins.dtype).view(1, 1, 3)
            ends = torch.where(mag.unsqueeze(-1) > 1.0e-5, ends, origins + tiny)
            starts = origins.reshape(-1, 3)
            ends = ends.reshape(-1, 3)
            line_pos, line_quat, line_len = self._get_connecting_lines(start_pos=starts, end_pos=ends)
            marker_scales = torch.ones((line_pos.size(0), 3), device=self.device)
            marker_scales[:, -1] = torch.clamp(line_len, min=1.0e-4)
            marker_indices = torch.arange(n_bodies, device=self.device).repeat(self.num_envs)
            self._contact_force_markers.visualize(
                translations=line_pos,
                orientations=line_quat,
                scales=marker_scales,
                marker_indices=marker_indices,
            )

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = end_pos - start_pos
        lengths = torch.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2

        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(start_pos.size(0), -1)
        direction_norm = normalize(direction)
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1)

        mask = rotation_axis_norm > 1e-6
        rotation_axis = torch.where(
            mask.unsqueeze(-1),
            normalize(rotation_axis),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(start_pos.size(0), -1),
        )

        cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        orientations = quat_from_angle_axis(angle, rotation_axis)

        return positions, orientations, lengths


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    object_asset_name: str = MISSING
    robot_asset_name: str = MISSING
    tf_asset_names: list[str] = MISSING

    demo_path: str = MISSING
    demo_dt: float = MISSING
    retargeted_traj_path: str = MISSING
    augmentation_traj_dir: str | None = None

    keypoint_paths: dict[str, str] = MISSING
    keypoint_body_prim_paths: dict[str, str] = MISSING
    keypoint_scale: float = 1.0
    object_joint_axis: tuple[float, float, float] = MISSING

    # anchor_body_name: str = MISSING
    # body_names: list[str] = MISSING
    actuated_joint_names: list[str] = MISSING

    enable_reset_perturbation: bool = False

    traj_aug_enabled: bool = False
    traj_aug_translation_lower_bound: tuple[float, float, float] = (0.0, 0.0, 0.0)
    traj_aug_translation_upper_bound: tuple[float, float, float] = (0.0, 0.0, 0.0)
    traj_aug_yaw_range: tuple[float, float] = (0.0, 0.0)
    traj_aug_start: float = 0.0
    traj_aug_use: float = 1.0
    traj_aug_apply_to_hand: bool = True

    rand_wrist_pos_range: tuple[float, float] = (-0.005, 0.005)
    rand_wrist_rot_range: tuple[float, float] = (-0.05, 0.05)
    rand_finger_joint_pos_range: tuple[float, float] = (-0.05, 0.05)

    rand_object_pos_range: tuple[float, float] = (-0.005, 0.005)
    rand_object_rot_range: tuple[float, float] = (-0.05, 0.05)
    rand_object_joint_range: tuple[float, float] = (-0.05, 0.05)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_sampling_enabled: bool = False
    rsi_enabled: bool = True

    # Uniform scale for motion debug markers (MANO skeleton + object keypoints). 1.0 = legacy size.
    debug_vis_marker_scale: float = 1.0
    debug_vis_keypoints: bool = False

    # When True (and scene has ``contact_forces`` sensor), draw net contact force as cylinders (arrow along F, length ~ ||F||).
    # Disables Isaac Lab ContactSensor sphere debug_vis (see DexterousEnvCfg) to avoid duplicate markers.
    # Default False; enable only on *-Play-* env cfgs.
    debug_vis_contact_force_arrows: bool = False
    # World-frame force (N) → arrow length (m), before min/max clamp.
    contact_force_arrow_scale: float = 5.0e-4
    contact_force_arrow_min_length: float = 0.008
    contact_force_arrow_max_length: float = 0.12

    # anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    # anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    # body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    # body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    object_pos_lower_bound: tuple[float, float, float] = MISSING  # [-0.25, -0.25, 0.95]
    object_pos_upper_bound: tuple[float, float, float] = MISSING  # [0.25, 0.25, 1.5]
