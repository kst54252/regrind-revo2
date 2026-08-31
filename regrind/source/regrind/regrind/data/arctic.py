import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict

QPOS_DIM = 30
QPOS_INDEX = {
    "object_pos": np.arange(0, 3),
    "object_quat": np.arange(3, 7),
    "object_joint": np.arange(7, 8),
    "hand_wrist_pos": np.arange(8, 11),
    "hand_wrist_rot": np.arange(11, 14),
    "hand_index": np.arange(14, 18),
    "hand_middle": np.arange(18, 22),
    "hand_ring": np.arange(22, 26),
    "hand_thumb": np.arange(26, 30),

    # convenience
    "object_pose": np.arange(0, 7),
    "hand": np.arange(8, 30),
    "fingers": np.arange(14, 30),  # hand excluding wrist joints

}

QVEL_INDEX = QPOS_INDEX.copy()
QVEL_INDEX["object_root_vel"] = np.arange(0, 6)
QVEL_INDEX["object_root_linvel"] = np.arange(0, 3)
QVEL_INDEX["object_root_angvel"] = np.arange(3, 6)  # NOTE: only used to index a qvel array.
# We construct qvel to be the same shape as qpos, qvel[3:6] is the angular velocity, qvel[6] is never used.

# Wuji right hand: 20 finger DoFs (same prefix as Leap through wrist; fingers span 20 scalars).
QPOS_DIM_WUJI = 34
QPOS_INDEX_WUJI = {
    "object_pos": np.arange(0, 3),
    "object_quat": np.arange(3, 7),
    "object_joint": np.arange(7, 8),
    "hand_wrist_pos": np.arange(8, 11),
    "hand_wrist_rot": np.arange(11, 14),
    "fingers": np.arange(14, 34),
    "object_pose": np.arange(0, 7),
    "hand": np.arange(8, 34),
}
QVEL_INDEX_WUJI = QPOS_INDEX_WUJI.copy()
QVEL_INDEX_WUJI["object_root_vel"] = np.arange(0, 6)
QVEL_INDEX_WUJI["object_root_linvel"] = np.arange(0, 3)
QVEL_INDEX_WUJI["object_root_angvel"] = np.arange(3, 6)


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[..., [1, 2, 3, 0]]


def _ensure_retargeted_traj_xyzw(traj: dict) -> dict:
    """Normalize retargeted trajectory quaternion datasets to Isaac Lab XYZW order."""
    quat_convention = _normalize_quat_convention(traj.get("quat_convention", "wxyz"))
    if quat_convention == "wxyz":
        traj["robot_quat"] = _quat_wxyz_to_xyzw(traj["robot_quat"])
        traj["object_quat"] = _quat_wxyz_to_xyzw(traj["object_quat"])
        traj["quat_convention"] = "xyzw"
    elif quat_convention != "xyzw":
        raise ValueError(f"Unsupported retargeted trajectory quaternion convention: {quat_convention!r}")
    if "robot_euler_XYZ" not in traj:
        traj["robot_euler_XYZ"] = R.from_quat(traj["robot_quat"]).as_euler("XYZ", degrees=False)
    return traj


def _normalize_quat_convention(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return str(value).lower()


def load_object_traj(path: str) -> Dict[str, np.ndarray]:
    """Load object trajectory from a npy file (raw_seqs)."""
    # raw sequence is a 2D array of shape (T, 7), 1 dim for articulation, 3 dims for rotation in axis-angle, 3 dims for translation
    data = np.load(path)
    print(f"Loaded raw object trajectory with shape {data.shape}")
    traj = {}
    traj["arti"] = np.array(data[:, 0:1])
    traj["axis_angle"] = np.array(data[:, 1:4])
    traj["pos"] = np.array(data[:, 4:7]) / 1000.0  # convert to meters
    quat = np.zeros((data.shape[0], 4))
    for i, axis_angle in enumerate(traj["axis_angle"]):
        quat[i] = R.from_rotvec(axis_angle).as_quat()  # (x, y, z, w)
    traj["quat"] = quat
    return traj


def load_mano_traj(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load mano trajectory from a npy file (raw_seqs)."""
    data = np.load(path, allow_pickle=True).item()
    traj = {}
    for hand in ["left", "right"]:
        traj[hand] = {}
        traj[hand]["pos"] = np.array(data[hand]["trans"])
        traj[hand]["axis_angle"] = np.array(data[hand]["rot"])
        quat = np.zeros((data[hand]["rot"].shape[0], 4))
        euler_XYZ = np.zeros((data[hand]["rot"].shape[0], 3))
        euler_xyz = np.zeros((data[hand]["rot"].shape[0], 3))
        for i, axis_angle in enumerate(traj[hand]["axis_angle"]):
            quat[i] = R.from_rotvec(axis_angle).as_quat()  # (x, y, z, w)
            euler_XYZ[i] = R.from_rotvec(axis_angle).as_euler("XYZ", degrees=False)
            euler_xyz[i] = R.from_rotvec(axis_angle).as_euler("xyz", degrees=False)
        traj[hand]["quat"] = quat
        traj[hand]["euler_XYZ"] = euler_XYZ
        traj[hand]["euler_xyz"] = euler_xyz
    return traj


def load_processed_traj(path: str) -> Dict[str, np.ndarray]:
    """Load processed trajectory from a npy file (processed_verts)."""
    data = np.load(path, allow_pickle=True).item()
    # data: dict, keys: ['world_coord', 'cam_coord', '2d', 'bbox', 'params']
    # world_coord: dict, keys: ['verts.right', 'joints.right', 'verts.left', 'joints.left', 'diameter', 'f', 'f_len', 'v_len', 'mask', 'parts_ids', 'bbox3d', 'kp3d', 'verts.object', 'verts.smplx', 'joints.smplx', 'rot_r', 'rot_l', 'obj_rot']
    traj = {}
    for hand in ["left", "right"]:
        traj[hand] = {}
        traj[hand]["joint_coord"] = np.array(data["world_coord"][f"joints.{hand}"])  # (traj_len, 21, 3)
        traj[hand]["rot"] = data["world_coord"][f"rot_{hand[0]}"]  # (traj_len, 3) # wrist rot in axis-angle?

    return traj


def load_retargeted_traj(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load the retargeted trajectory from a npy file or h5 file.
    The data is a dict:
        'robot_pos': (traj_len, 3)
        'robot_quat': (traj_len, 4), XYZW after loading
        'robot_joints': (traj_len, 16) Leap, or (traj_len, 20) Wuji
        'object_pos': (traj_len, 3)
        'object_quat': (traj_len, 4), XYZW after loading
        'object_joint': (traj_len, 1)
    """
    if path.endswith(".h5"):
        from regrind.data.utils import read_h5_to_dict
        traj = read_h5_to_dict(path)
        return _ensure_retargeted_traj_xyzw(traj)
    elif path.endswith(".npy"):
        traj = np.load(path, allow_pickle=True).item()
        return _ensure_retargeted_traj_xyzw(traj)
    else:
        raise ValueError(f"Unsupported file: {path}")
