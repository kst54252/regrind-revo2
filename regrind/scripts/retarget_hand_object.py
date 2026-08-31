"""Hand-object interaction-mesh retargeting, producing regrind-format trajectories.

Maps a MANO hand-joint demo onto the LeapHand, WujiHand, or Revo2 using the
Drake-based :class:`~regrind.retargeting.HandInteractionMeshOneStageRetargeter`, then writes a
regrind-consumable HDF5/NPZ trajectory. In addition to robot, object, and MANO
arrays, sequence output contains per-frame solver/objective/limit diagnostics.

Requires ``pydrake`` and (for the default solver) a licensed Mosek install.

Examples:
    python scripts/retarget_hand_object.py --robot leaphand --object scissors
    python scripts/retarget_hand_object.py --robot leaphand --object scissors --num-timesteps 20 \
        --out /tmp/scissors_test.h5 --no-visualize
"""

import argparse
import os
from datetime import datetime
from typing import Literal

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from pydrake.all import ClarabelSolver, MosekSolver

from regrind.data.utils import read_h5_to_dict
from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert Drake/retargeter quaternions to Isaac Lab runtime order."""
    return np.asarray(quat)[..., [1, 2, 3, 0]]


def _load_robot_constants_module(robot_name: str):
    """Load the per-hand constants module and its MANO->robot link mapping."""
    if robot_name == "leaphand":
        from regrind.retargeting import leaphand_constants as rc

        return rc, rc.MANO_TO_LEAP_MAPPING
    if robot_name == "wujihand":
        from regrind.retargeting import wujihand_constants as rc

        return rc, rc.MANO_TO_WUJI_MAPPING
    if robot_name == "revo2":
        from regrind.retargeting import revo2_constants as rc

        return rc, rc.MANO_TO_REVO2_MAPPING
    raise ValueError(
        f"Unknown robot {robot_name!r}; expected 'leaphand', 'wujihand', or 'revo2'."
    )


# ---------------------------------------------------------------------------
# Demo loading
# ---------------------------------------------------------------------------
def load_demo_data(
    demo_path: str,
    data_type: Literal["arctic", "zed_mocap", "dexycb"],
    input_quat_convention: str = "wxyz",
    input_pose_layout: str = "pos_quat",
):
    if data_type == "arctic":
        return load_arctic_demo(demo_path)
    if data_type == "zed_mocap":
        return load_custom_zed_mocap_demo(demo_path)
    if data_type == "dexycb":
        return load_dexycb_demo(
            demo_path,
            input_quat_convention=input_quat_convention,
            input_pose_layout=input_pose_layout,
        )
    raise ValueError(f"Invalid data type: {data_type}")


def _first_present(data: dict, names: tuple[str, ...], description: str):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"DexYCB demo has no {description}; tried datasets {names}")


def _to_wxyz(quaternion: np.ndarray, convention: str) -> np.ndarray:
    quaternion = np.asarray(quaternion)
    if convention == "wxyz":
        return quaternion
    if convention == "xyzw":
        return quaternion[..., [3, 0, 1, 2]]
    raise ValueError(f"unsupported quaternion convention: {convention}")


def _estimate_rigid_pose_from_keypoints(
    robot_points_local: np.ndarray,
    human_points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the proper rigid transform aligning robot points to MANO points.

    The SVD correction explicitly enforces ``det(R)=+1``.  This is important
    for left-hand DexYCB observations: a reflection may fit their chirality
    better, but it is not a realizable wrist SE(3) pose for a right-hand robot.
    """
    robot_points_local = np.asarray(robot_points_local, dtype=float)
    human_points_world = np.asarray(human_points_world, dtype=float)
    if robot_points_local.shape != human_points_world.shape:
        raise ValueError(
            "robot and human keypoints must have matching shapes, got "
            f"{robot_points_local.shape} and {human_points_world.shape}"
        )
    if (
        robot_points_local.ndim != 2
        or robot_points_local.shape[1] != 3
        or len(robot_points_local) < 3
    ):
        raise ValueError("keypoints must have shape (N,3) with N >= 3")
    if not (
        np.isfinite(robot_points_local).all()
        and np.isfinite(human_points_world).all()
    ):
        raise ValueError("keypoints used for wrist initialization contain NaN/Inf")

    robot_center = robot_points_local.mean(axis=0)
    human_center = human_points_world.mean(axis=0)
    covariance = (robot_points_local - robot_center).T @ (
        human_points_world - human_center
    )
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = human_center - rotation @ robot_center
    aligned = robot_points_local @ rotation.T + translation
    mean_error = float(np.linalg.norm(aligned - human_points_world, axis=1).mean())
    quaternion_xyzw = R.from_matrix(rotation).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return quaternion_wxyz, translation, mean_error


def _keypoint_aligned_wrist_initialization(
    retargeter,
    human_points_world: np.ndarray,
    object_pose_wxyz_xyz: np.ndarray,
    default_joints: np.ndarray,
    mano_to_robot,
    local_rpy_correction_deg: np.ndarray | tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """Build wrist + joint initialization from semantic point alignment."""
    default_joints = np.asarray(default_joints, dtype=float)
    if default_joints.shape != (retargeter.robot_dof,):
        raise ValueError(
            f"default robot joints must have shape ({retargeter.robot_dof},), "
            f"got {default_joints.shape}"
        )
    q_probe = np.zeros(retargeter.nq, dtype=float)
    q_probe[0] = 1.0
    q_probe[retargeter.actuated_position_indices] = default_joints
    object_start = retargeter.robot_position_count
    q_probe[object_start : object_start + 7] = np.asarray(
        object_pose_wxyz_xyz, dtype=float
    )
    retargeter._apply_mimic_positions(q_probe)
    robot_points_local = retargeter._get_robot_link_positions(
        q_probe, mano_to_robot
    )
    quaternion_wxyz, translation, mean_error_before = _estimate_rigid_pose_from_keypoints(
        robot_points_local, human_points_world
    )
    local_rpy_correction_deg = np.asarray(local_rpy_correction_deg, dtype=float)
    if (
        local_rpy_correction_deg.shape != (3,)
        or not np.isfinite(local_rpy_correction_deg).all()
    ):
        raise ValueError("initial wrist local RPY correction must be three finite values")
    rotation = R.from_quat(quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    local_correction = R.from_euler(
        "XYZ", local_rpy_correction_deg, degrees=True
    ).as_matrix()
    rotation = rotation @ local_correction
    # Keep the semantic-point centroids coincident after rotating about the
    # wrist frame. This changes orientation without introducing a translation
    # artifact from the robot keypoints' nonzero local centroid.
    translation = (
        human_points_world.mean(axis=0)
        - rotation @ robot_points_local.mean(axis=0)
    )
    aligned = robot_points_local @ rotation.T + translation
    mean_error = float(
        np.linalg.norm(aligned - human_points_world, axis=1).mean()
    )
    quaternion_xyzw = R.from_matrix(rotation).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    print("[DexYCB wrist initialization]")
    print(f"  method:                    semantic-keypoint Kabsch alignment")
    print(f"  local RPY correction:      {local_rpy_correction_deg} deg")
    print(f"  quaternion wxyz:           {quaternion_wxyz}")
    print(f"  position:                  {translation}")
    print(f"  det(R):                    {np.linalg.det(R.from_quat(quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()):.9g}")
    print(f"  Kabsch mean point error:   {mean_error_before:.9g} m")
    print(f"  corrected mean error:      {mean_error:.9g} m")
    return np.concatenate((quaternion_wxyz, translation, default_joints))


def load_dexycb_demo(
    demo_path: str,
    input_quat_convention: str = "wxyz",
    input_pose_layout: str = "pos_quat",
):
    """Load a preprocessed DexYCB HDF5/NPZ trajectory.

    Supported hand datasets are ``human_hand_keypoints``, ``mano_joint_coords``,
    and ``mano_joint_coord``. Object pose may be separate position/quaternion
    arrays or a combined ``object_pose(_trajectory)`` array with seven columns.
    """
    if str(demo_path).lower().endswith(".npz"):
        with np.load(demo_path) as archive:
            data = {key: archive[key] for key in archive.files}
    else:
        data = read_h5_to_dict(demo_path)

    mano_joint_coords = _first_present(
        data,
        ("human_hand_keypoints", "mano_joint_coords", "mano_joint_coord"),
        "human hand keypoints",
    )
    if mano_joint_coords.ndim != 3 or mano_joint_coords.shape[1:] != (21, 3):
        raise ValueError(
            "DexYCB human keypoints must have shape (T, 21, 3), got "
            f"{mano_joint_coords.shape}"
        )
    if not np.isfinite(mano_joint_coords).all():
        raise ValueError("DexYCB human keypoints contain NaN/Inf")

    object_position_names = ("object_pos", "obj_pos", "object_positions")
    object_quaternion_names = ("object_quat", "obj_quat", "object_quaternions")
    try:
        object_pos = _first_present(data, object_position_names, "object position")
        object_quat = _first_present(data, object_quaternion_names, "object quaternion")
    except KeyError:
        object_pose = _first_present(
            data,
            ("object_pose_trajectory", "object_pose", "object_poses"),
            "object pose trajectory",
        )
        if object_pose.ndim != 2 or object_pose.shape[1] != 7:
            raise ValueError(f"combined object pose must have shape (T, 7), got {object_pose.shape}")
        if input_pose_layout == "pos_quat":
            object_pos, object_quat = object_pose[:, :3], object_pose[:, 3:]
        else:
            object_quat, object_pos = object_pose[:, :4], object_pose[:, 4:]
    object_quat = _to_wxyz(object_quat, input_quat_convention)
    object_poses = np.concatenate((object_quat, object_pos), axis=1)

    wrist_pos = next(
        (np.asarray(data[name]) for name in ("wrist_pos", "robot_pos") if name in data),
        mano_joint_coords[:, 0],
    )
    wrist_quat = next(
        (np.asarray(data[name]) for name in ("wrist_quat", "robot_quat") if name in data),
        np.tile(np.array((1.0, 0.0, 0.0, 0.0)), (len(mano_joint_coords), 1)),
    )
    wrist_quat = _to_wxyz(wrist_quat, input_quat_convention)
    mano_poses = np.concatenate((wrist_quat, wrist_pos), axis=1)

    object_joints = next(
        (np.asarray(data[name]) for name in ("object_joint", "obj_joint") if name in data),
        None,
    )
    if object_joints is not None and object_joints.ndim == 1:
        object_joints = object_joints[:, None]

    lengths = {len(mano_joint_coords), len(mano_poses), len(object_poses)}
    if object_joints is not None:
        lengths.add(len(object_joints))
    if len(lengths) != 1:
        raise ValueError(f"DexYCB trajectory lengths do not match: {sorted(lengths)}")
    print(f"Loading DexYCB demo with {len(mano_joint_coords)} timesteps from {demo_path}.")
    return mano_poses, mano_joint_coords, object_poses, object_joints


def load_arctic_demo(arctic_demo_path: str):
    arctic_demo_data = read_h5_to_dict(arctic_demo_path)
    mano_joint_coords = arctic_demo_data["mano_joint_coord"]  # (T, 21, 3)
    print(f"Loading demo with {mano_joint_coords.shape[0]} timesteps from {arctic_demo_path}.")

    object_pos_list = arctic_demo_data["qpos"][:, :3]
    object_quat_list = arctic_demo_data["qpos"][:, 3:7]
    object_arti_list = arctic_demo_data["qpos"][:, 7:8]

    object_poses = np.concatenate([object_quat_list, object_pos_list], axis=1)  # (T, 7)
    object_joints = np.array(object_arti_list)  # (T, 1)

    root_positions = arctic_demo_data["qpos"][:, 8:11]
    root_rotations = arctic_demo_data["qpos"][:, 11:14]
    root_rotations = R.from_euler("XYZ", root_rotations).as_quat(scalar_first=True)
    mano_poses = np.concatenate([root_rotations, root_positions], axis=1)  # (T, 7)

    return mano_poses, mano_joint_coords, object_poses, object_joints


def load_custom_zed_mocap_demo(demo_path: str):
    demo_data = read_h5_to_dict(demo_path)
    mano_joint_coords = demo_data["mano_joint_coords"]  # (T, 21, 3)
    print(f"Loading demo with {mano_joint_coords.shape[0]} timesteps from {demo_path}.")

    object_pos_list = demo_data["obj_pos"]  # (T, 3)
    object_quat_list = demo_data["obj_quat"]  # (T, 4), WXYZ source boundary for Drake
    object_joints = demo_data["obj_joint"] if "obj_joint" in demo_data else None  # (T, 1)

    object_poses = np.concatenate([object_quat_list, object_pos_list], axis=1)  # (T, 7)
    object_joints = np.array(object_joints) if object_joints is not None else None

    root_positions = demo_data["wrist_pos"]  # (T, 3)
    root_rotations = demo_data["wrist_quat"]  # (T, 4), WXYZ source boundary for Drake
    mano_poses = np.concatenate([root_rotations, root_positions], axis=1)  # (T, 7)

    return mano_poses, mano_joint_coords, object_poses, object_joints


# ---------------------------------------------------------------------------
# Interpolation (30 -> 120 fps) and keypoint loading
# ---------------------------------------------------------------------------
def interpolate_poses(poses, factor):
    """Interpolate retargeter pose arrays [quat_wxyz(4), pos(3)] by ``factor`` using slerp + linear."""
    N = poses.shape[0]
    old_times = np.arange(N)
    new_times = np.linspace(0, N - 1, (N - 1) * factor + 1)

    quats_wxyz = poses[:, :4]
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    slerp = Slerp(old_times, R.from_quat(quats_xyzw))
    new_quats_xyzw = slerp(new_times).as_quat()
    new_quats_wxyz = new_quats_xyzw[:, [3, 0, 1, 2]]

    new_pos = np.column_stack([np.interp(new_times, old_times, poses[:, i]) for i in range(4, 7)])
    return np.concatenate([new_quats_wxyz, new_pos], axis=1)


def interpolate_demo(mano_poses, mano_joint_coords, object_poses, object_joints, factor):
    """Interpolate all demo arrays from source FPS to source*factor FPS."""
    N = mano_poses.shape[0]
    old_times = np.arange(N)
    new_times = np.linspace(0, N - 1, (N - 1) * factor + 1)

    mano_poses = interpolate_poses(mano_poses, factor)
    object_poses = interpolate_poses(object_poses, factor)

    num_joints = mano_joint_coords.shape[1]
    new_joint_coords = np.zeros((len(new_times), num_joints, 3))
    for j in range(num_joints):
        for d in range(3):
            new_joint_coords[:, j, d] = np.interp(new_times, old_times, mano_joint_coords[:, j, d])

    if object_joints is None:
        new_obj_joints = None
    else:
        new_obj_joints = np.column_stack(
            [np.interp(new_times, old_times, object_joints[:, i]) for i in range(object_joints.shape[1])]
        )

    return mano_poses, new_joint_coords, object_poses, new_obj_joints


def load_object_keypoints(object_keypoints_paths: dict, scale: float = 1.0):
    keypoints = {
        key: np.asarray(np.load(value), dtype=float) * scale
        for key, value in object_keypoints_paths.items()
    }
    for name, points in keypoints.items():
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"object keypoints {name!r} must have shape (N, 3), got {points.shape}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"object keypoints {name!r} contain NaN/Inf")
    return keypoints


def _compute_robot_keypoints(retargeter, retargeted_motions, mano_to_robot_mapping):
    """World-frame robot semantic points corresponding to mapped MANO joints."""
    mano_keypoint_names = list(mano_to_robot_mapping.keys())
    T = retargeted_motions.shape[0]
    robot_keypoints = np.full((T, len(mano_keypoint_names), 3), np.nan)
    for t in range(T):
        if not np.isfinite(retargeted_motions[t]).all():
            continue
        robot_keypoints[t] = retargeter._get_robot_link_positions(
            retargeted_motions[t], mano_to_robot_mapping
        )
    robot_keypoint_links = [mano_to_robot_mapping[k] for k in mano_keypoint_names]
    return robot_keypoints, mano_keypoint_names, robot_keypoint_links


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_retargeted(out_path: str, data: dict):
    """Write a flat-dataset trajectory as HDF5 or compressed NPZ."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if str(out_path).lower().endswith(".npz"):
        np.savez_compressed(
            out_path,
            **{key: np.asarray(value) for key, value in data.items() if value is not None},
        )
        print(f"Saved retargeted trajectory to {out_path}")
        return
    if not str(out_path).lower().endswith((".h5", ".hdf5")):
        raise ValueError("--out must end in .h5, .hdf5, or .npz")
    with h5py.File(out_path, "w") as f:
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, str):
                f.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding="utf-8"))
                continue
            array = np.asarray(value)
            if array.dtype.kind in ("U", "O"):
                f.create_dataset(
                    key,
                    data=array.astype(object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            else:
                f.create_dataset(key, data=array)
    print(f"Saved retargeted trajectory to {out_path}")


def _default_out_path(robot_name: str, object_name: str, scale_suffix: str) -> str:
    data_dir = os.environ["REGRIND_DATA_DIR"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        data_dir, "retargeted_traj", robot_name, f"{object_name}{scale_suffix}", f"retarget_{timestamp}.h5"
    )


def _resolve_object_config(args, robot_constants) -> dict:
    """Merge packaged object presets with explicit DexYCB path overrides."""
    try:
        cfg = robot_constants.get_object_config(args.object)
    except ValueError:
        if not (args.demo and args.object_model and args.object_keypoints):
            raise
        cfg = {
            "obj_scale": 1.0,
            "obj_scale_suffix": "",
            "table_height": None,
        }

    cfg = dict(cfg)
    if args.demo:
        cfg["demo_file"] = os.path.abspath(args.demo)
    if args.demo_type:
        cfg["demo_data_type"] = args.demo_type
    if args.object_model:
        cfg["object_urdf_file"] = os.path.abspath(args.object_model)
    if args.object_keypoints:
        cfg["object_keypoints_paths"] = {
            "bottom": os.path.abspath(args.object_keypoints)
        }
    if args.object_keypoints_top:
        cfg.setdefault("object_keypoints_paths", {})["top"] = os.path.abspath(
            args.object_keypoints_top
        )
    if args.object_scale is not None:
        cfg["obj_scale"] = args.object_scale
    if args.table_height is not None:
        cfg["table_height"] = args.table_height
    if args.object_body_name:
        cfg["object_body_name"] = args.object_body_name

    required = (
        "demo_file",
        "demo_data_type",
        "object_urdf_file",
        "object_keypoints_paths",
        "obj_scale",
    )
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(f"object configuration is missing fields: {missing}")
    cfg.setdefault("obj_scale_suffix", "")
    cfg.setdefault("table_height", None)
    cfg.setdefault("object_body_name", None)
    return cfg


def _print_single_frame_diagnostics(
    retargeter,
    q,
    objective,
    robot_keypoints,
    human_keypoints,
    object_keypoints,
):
    diagnostics = retargeter.frame_diagnostics[0]
    print("\n[single-frame validation]")
    print(f"  solver success:             {diagnostics['solver_success']}")
    print(f"  objective value:            {diagnostics['objective_value']:.9g}")
    print(f"  joint limit violation:      {diagnostics['joint_limit_violation']}")
    print(
        "  max joint limit violation:  "
        f"{diagnostics['max_joint_limit_violation']:.9g} rad"
    )
    print(f"  Revo2 keypoint shape:       {diagnostics['keypoint_shape']}")
    print(f"  Revo2 keypoints finite:     {diagnostics['keypoints_finite']}")
    print(f"  human keypoint shape:       {human_keypoints.shape}")
    print(f"  object keypoint shape:      {object_keypoints.shape}")
    if not diagnostics["solver_success"]:
        print(f"  failure:                    {diagnostics['error']}")
        return
    correspondence_error = np.linalg.norm(robot_keypoints - human_keypoints, axis=1)
    print(f"  mean correspondence error:  {correspondence_error.mean():.9g} m")
    print(f"  max correspondence error:   {correspondence_error.max():.9g} m")


def _diagnostic_arrays(retargeter):
    diagnostics = retargeter.frame_diagnostics
    return {
        "solver_success": np.asarray(
            [item["solver_success"] for item in diagnostics], dtype=bool
        ),
        "objective_value": np.asarray(
            [item["objective_value"] for item in diagnostics], dtype=float
        ),
        "joint_limit_violation": np.asarray(
            [item["joint_limit_violation"] for item in diagnostics], dtype=bool
        ),
        "max_joint_limit_violation": np.asarray(
            [item["max_joint_limit_violation"] for item in diagnostics], dtype=float
        ),
        "warm_start_frame": np.asarray(
            [
                -1 if item["warm_start_frame"] is None else item["warm_start_frame"]
                for item in diagnostics
            ],
            dtype=int,
        ),
        "solver_error": np.asarray([item["error"] for item in diagnostics]),
    }


def _print_sequence_summary(retargeter, robot_joints, robot_keypoints, joint_names):
    arrays = _diagnostic_arrays(retargeter)
    success = arrays["solver_success"]
    objectives = arrays["objective_value"][success]
    objectives = objectives[np.isfinite(objectives)]
    failures = np.flatnonzero(~success)

    print("\n[sequence retargeting summary]")
    print(f"  frames:                     {len(success)}")
    print(f"  success rate:               {100.0 * success.mean():.2f}%")
    print(f"  successful / failed:        {success.sum()} / {(~success).sum()}")
    print(
        "  mean / max objective:       "
        + (
            f"{objectives.mean():.9g} / {objectives.max():.9g}"
            if len(objectives)
            else "n/a / n/a"
        )
    )
    print(f"  failure frame indices:      {failures.tolist()}")
    print(
        "  any joint limit violation:  "
        f"{bool(np.any(arrays['joint_limit_violation'][success]))}"
    )
    print(f"  robot keypoint shape:       {robot_keypoints.shape}")
    print(
        "  successful keypoints finite: "
        f"{bool(np.isfinite(robot_keypoints[success]).all())}"
    )
    print("  optimized joint range [rad]:")
    if success.any():
        minimum = np.nanmin(robot_joints[success], axis=0)
        maximum = np.nanmax(robot_joints[success], axis=0)
        for name, lo, hi in zip(joint_names, minimum, maximum):
            print(f"    {name:<28} {lo: .6f} .. {hi: .6f}")
    else:
        print("    n/a (no successful frames)")
    return arrays


def main():
    parser = argparse.ArgumentParser(description="Hand-object interaction-mesh retargeting (1-stage).")
    parser.add_argument(
        "--robot", type=str, default="leaphand",
        choices=("leaphand", "wujihand", "revo2"),
    )
    parser.add_argument("--object", type=str, default="scissors")
    parser.add_argument("--robot-model", type=str, help="Override the robot URDF path.")
    parser.add_argument("--demo", type=str, help="Override the input demo HDF5/NPZ path.")
    parser.add_argument(
        "--demo-type", choices=("arctic", "zed_mocap", "dexycb"),
        help="Input demo layout; required for a custom DexYCB demo.",
    )
    parser.add_argument("--object-model", type=str, help="Override the object URDF/SDF path.")
    parser.add_argument(
        "--object-keypoints", type=str,
        help="Object surface keypoints .npy file with shape (N, 3).",
    )
    parser.add_argument("--object-keypoints-top", type=str)
    parser.add_argument("--object-body-name", type=str)
    parser.add_argument("--object-scale", type=float)
    parser.add_argument("--table-height", type=float)
    parser.add_argument(
        "--input-quat-convention", choices=("wxyz", "xyzw"), default="wxyz"
    )
    parser.add_argument(
        "--input-pose-layout", choices=("pos_quat", "quat_pos"), default="pos_quat"
    )
    parser.add_argument(
        "--wrist-init",
        choices=("auto", "demo", "keypoints"),
        default="auto",
        help=(
            "Initial floating-wrist pose. 'auto' uses semantic-keypoint rigid "
            "alignment for Revo2 DexYCB and the stored demo pose otherwise."
        ),
    )
    parser.add_argument(
        "--initial-wrist-local-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Fixed intrinsic local XYZ correction applied after keypoint wrist "
            "initialization. Use '0 0 180' to flip the palm while preserving "
            "the Revo2 finger-forward local Z direction."
        ),
    )
    parser.add_argument(
        "--temporal-smooth-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of the existing frame-to-frame configuration smoothness "
            "cost (default: 0.5). Increase it to preserve a selected wrist "
            "orientation branch after a 180-degree initial correction."
        ),
    )
    parser.add_argument(
        "--penetration-tolerance",
        type=float,
        default=None,
        help=(
            "Override the robot/object non-penetration tolerance in metres. "
            "Defaults to the existing robot/object-specific value."
        ),
    )
    parser.add_argument(
        "--single-frame", type=int,
        help="Optimize only this source-demo frame (before interpolation).",
    )
    parser.add_argument(
        "--interpolation-factor", type=int, default=None,
        help=(
            "Temporal interpolation factor; defaults to 1 for DexYCB and 4 for "
            "legacy ARCTIC/ZED demos. Ignored with --single-frame."
        ),
    )
    parser.add_argument(
        "--source-fps", type=float, default=30.0,
        help="Source demo frame rate, stored in the output metadata (default: 30).",
    )
    parser.add_argument(
        "--solver", choices=("mosek", "clarabel"), default="mosek",
        help="SQP subproblem solver; Clarabel does not require a license.",
    )
    parser.add_argument("--num-timesteps", type=int, default=None, help="Only optimize the first K timesteps.")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output .h5/.hdf5/.npz path (default under data/retargeted_traj/...).",
    )
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable Meshcat visualization.")
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be > 0")
    if args.temporal_smooth_weight < 0 or not np.isfinite(
        args.temporal_smooth_weight
    ):
        raise ValueError("--temporal-smooth-weight must be finite and >= 0")
    if args.penetration_tolerance is not None and (
        args.penetration_tolerance < 0
        or not np.isfinite(args.penetration_tolerance)
    ):
        raise ValueError("--penetration-tolerance must be finite and >= 0")

    rc, mano_to_robot = _load_robot_constants_module(args.robot)
    obj_cfg = _resolve_object_config(args, rc)

    robot_dof = rc.ROBOT_DOF
    mano_joints = rc.MANO_JOINTS

    # Load and interpolate the demo (30 -> 120 fps).
    mano_poses, mano_joint_coords, object_poses, object_joints = load_demo_data(
        obj_cfg["demo_file"],
        obj_cfg["demo_data_type"],
        input_quat_convention=args.input_quat_convention,
        input_pose_layout=args.input_pose_layout,
    )
    if args.single_frame is not None:
        frame = args.single_frame
        if frame < 0 or frame >= len(mano_joint_coords):
            raise IndexError(
                f"--single-frame {frame} is outside [0, {len(mano_joint_coords) - 1}]"
            )
        selection = slice(frame, frame + 1)
        mano_poses = mano_poses[selection]
        mano_joint_coords = mano_joint_coords[selection]
        object_poses = object_poses[selection]
        object_joints = object_joints[selection] if object_joints is not None else None
    else:
        interpolation_factor = args.interpolation_factor
        if interpolation_factor is None:
            interpolation_factor = 1 if obj_cfg["demo_data_type"] == "dexycb" else 4
        if interpolation_factor < 1:
            raise ValueError("--interpolation-factor must be >= 1")
        if interpolation_factor > 1:
            mano_poses, mano_joint_coords, object_poses, object_joints = interpolate_demo(
                mano_poses,
                mano_joint_coords,
                object_poses,
                object_joints,
                factor=interpolation_factor,
            )
    output_fps = args.source_fps * (
        1 if args.single_frame is not None else interpolation_factor
    )

    object_keypoints = load_object_keypoints(obj_cfg["object_keypoints_paths"], scale=obj_cfg["obj_scale"])

    hand_keypoint_weight = None
    penetration_tolerance = 0.001
    if args.robot == "wujihand" and args.object == "scissors":
        # De-emphasize the index finger for the Wuji scissors demo.
        hand_keypoint_weight = {k: (0.1 if k in ("1", "2", "3", "17") else 1.0) for k in mano_joints}
        penetration_tolerance = 0.002
    if args.penetration_tolerance is not None:
        penetration_tolerance = args.penetration_tolerance

    retargeter = HandInteractionMeshOneStageRetargeter(
        robot_model_path=args.robot_model or rc.ROBOT_URDF_FILE,
        robot_name=rc.ROBOT_NAME,
        object_model_path=obj_cfg["object_urdf_file"],
        object_name=args.object,
        table_height=obj_cfg["table_height"],
        object_body_name=obj_cfg["object_body_name"],
        q_a_init_idx=-7,
        activate_joint_limits=True,
        activate_obj_non_penetration=False,
        demo_joints=mano_joints,
        laplacian_match_links=mano_to_robot,
        hand_keypoint_weight=hand_keypoint_weight,
        penetration_tolerance=penetration_tolerance,
        step_size=0.2,
        solver=MosekSolver() if args.solver == "mosek" else ClarabelSolver(),
        visualize=args.visualize,
        debug=False,
        w_nominal_tracking_init=5.0,
        semantic_keypoints=getattr(rc, "SEMANTIC_KEYPOINTS", None),
        independent_joint_names=getattr(rc, "ACTUATED_JOINT_NAMES", None),
        mimic_joints=getattr(rc, "MIMIC_JOINTS", None),
        robot_model_dof=getattr(rc, "ROBOT_MODEL_DOF", None),
    )
    retargeter.smooth_weight = args.temporal_smooth_weight

    default_joints = np.asarray(
        getattr(rc, "DEFAULT_JOINT_POSITIONS", np.zeros(robot_dof)), dtype=float
    )
    wrist_init = args.wrist_init
    if wrist_init == "auto":
        wrist_init = (
            "keypoints"
            if args.robot == "revo2" and obj_cfg["demo_data_type"] == "dexycb"
            else "demo"
        )
    if wrist_init == "keypoints":
        human_initial = mano_joint_coords[
            0, retargeter.mano_mapped_joint_indices
        ]
        q_a_init = _keypoint_aligned_wrist_initialization(
            retargeter,
            human_initial,
            object_poses[0],
            default_joints,
            mano_to_robot,
            args.initial_wrist_local_rpy_deg,
        )
    else:
        q_a_init = np.concatenate([mano_poses[0], default_joints])
        print("[wrist initialization] using pose stored in the demo")

    K = args.num_timesteps
    if K is not None:
        if K < 1:
            raise ValueError("--num-timesteps must be >= 1")
        mano_poses = mano_poses[:K]
        mano_joint_coords = mano_joint_coords[:K]
        object_poses = object_poses[:K]
        object_joints = object_joints[:K] if object_joints is not None else None

    top_kp = object_keypoints.get("top")

    retargeter.penetration_tolerance = penetration_tolerance
    retargeter.activate_obj_non_penetration = True
    (retargeted_motions, *_) = retargeter.retarget_motion(
        human_joint_motions=mano_joint_coords,
        object_poses=object_poses,
        object_poses_augmented=object_poses.copy(),
        object_points_local_demo=object_keypoints["bottom"],
        object_points_local=object_keypoints["bottom"],
        object_points_local_demo_2=top_kp,
        object_points_local_2=top_kp,
        object_joints=object_joints,
        q_a_init=q_a_init,
        q_nominal_list=None,
        original=True,
    )

    if args.visualize:
        retargeter.remove_all_keypoints()
        retargeter.draw_q_knots(retargeted_motions, 1 / output_fps)

    # Slice the packed WXYZ Drake solution into the XYZW trajectory schema.
    # Revo2's packed plant state includes five dependent mimic positions, but
    # get_actuated_joint_positions() exports only the six independent joints.
    robot_keypoints, mano_kp_names, robot_kp_links = _compute_robot_keypoints(
        retargeter, retargeted_motions, mano_to_robot
    )
    robot_joints = retargeter.get_actuated_joint_positions(retargeted_motions)
    diagnostic_arrays = _print_sequence_summary(
        retargeter,
        robot_joints,
        robot_keypoints,
        getattr(rc, "ACTUATED_JOINT_NAMES", tuple(range(robot_dof))),
    )
    failure_indices = np.flatnonzero(~diagnostic_arrays["solver_success"])
    semantic_keypoints = getattr(rc, "SEMANTIC_KEYPOINTS", {})
    robot_keypoint_names = np.asarray(
        [
            semantic_keypoints.get(name, {}).get("name", robot_kp_links[index])
            for index, name in enumerate(mano_kp_names)
        ]
    )
    out_data = {
        "quat_convention": "xyzw",
        "wrist_initialization": wrist_init,
        "initial_wrist_quat_wxyz": q_a_init[:4],
        "initial_wrist_pos": q_a_init[4:7],
        "initial_wrist_local_rpy_correction_deg": np.asarray(
            args.initial_wrist_local_rpy_deg, dtype=float
        ),
        "temporal_smooth_weight": np.asarray(args.temporal_smooth_weight),
        "penetration_tolerance_m": np.asarray(penetration_tolerance),
        "fps": output_fps,
        "frame_index": np.arange(len(retargeted_motions), dtype=int),
        "robot_pos": retargeted_motions[:, 4:7],
        "robot_quat": _quat_wxyz_to_xyzw(retargeted_motions[:, :4]),
        "robot_joints": robot_joints,
        "actuated_joint_names": np.asarray(
            getattr(rc, "ACTUATED_JOINT_NAMES", tuple(range(robot_dof)))
        ).astype(str),
        # The object trajectory is observed input, not an optimization result;
        # retain it even when a robot solve fails.
        "object_pos": object_poses[:, 4:7],
        "object_quat": _quat_wxyz_to_xyzw(object_poses[:, :4]),
        "object_joint": object_joints[:, 0] if object_joints is not None else None,
        "object_points_local": object_keypoints["bottom"],
        "robot_keypoints": robot_keypoints,
        "robot_keypoint_names": robot_keypoint_names,
        "mano_joint_coords": mano_joint_coords,
        "failure_indices": failure_indices,
        **diagnostic_arrays,
    }

    if args.robot == "revo2":
        if robot_keypoints.shape[1:] != (21, 3):
            raise RuntimeError(
                f"Revo2 FK must produce (T, 21, 3), got {robot_keypoints.shape}"
            )
        success = diagnostic_arrays["solver_success"]
        if not np.isfinite(robot_keypoints[success]).all():
            raise RuntimeError("Successful Revo2 frames contain NaN/Inf keypoints")
        if failure_indices.size and not np.isnan(robot_keypoints[~success]).all():
            raise RuntimeError("Failed Revo2 frames must be marked with NaN keypoints")

    if args.single_frame is not None:
        human_mapped = mano_joint_coords[0, retargeter.mano_mapped_joint_indices]
        _print_single_frame_diagnostics(
            retargeter,
            retargeted_motions[0],
            retargeter.last_objective,
            robot_keypoints[0],
            human_mapped,
            object_keypoints["bottom"],
        )

    out_path = args.out or _default_out_path(rc.ROBOT_NAME, args.object, obj_cfg["obj_scale_suffix"])
    save_retargeted(out_path, out_data)


if __name__ == "__main__":
    main()
