"""Hand-object interaction-mesh retargeting, producing regrind-format trajectories.

Maps a MANO hand-joint demo onto the LeapHand / WujiHand for scissors / screwdriver using the
Drake-based :class:`~regrind.retargeting.HandInteractionMeshOneStageRetargeter`, then writes a
regrind-consumable ``.h5`` (flat datasets ``robot_pos``/``robot_quat``/``robot_joints``/
``object_pos``/``object_quat``/``object_joint``/``robot_keypoints``/``mano_joint_coords``) that
``regrind.data.arctic.load_retargeted_traj`` / ``MotionLoader`` read directly.

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

from regrind.data.utils import read_h5_to_dict
from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter


def _load_robot_constants_module(robot_name: str):
    """Load the per-hand constants module and its MANO->robot link mapping."""
    if robot_name == "leaphand":
        from regrind.retargeting import leaphand_constants as rc

        return rc, rc.MANO_TO_LEAP_MAPPING
    if robot_name == "wujihand":
        from regrind.retargeting import wujihand_constants as rc

        return rc, rc.MANO_TO_WUJI_MAPPING
    raise ValueError(f"Unknown robot {robot_name!r}; expected 'leaphand' or 'wujihand'.")


# ---------------------------------------------------------------------------
# Demo loading
# ---------------------------------------------------------------------------
def load_demo_data(demo_path: str, data_type: Literal["arctic", "zed_mocap"]):
    if data_type == "arctic":
        return load_arctic_demo(demo_path)
    if data_type == "zed_mocap":
        return load_custom_zed_mocap_demo(demo_path)
    raise ValueError(f"Invalid data type: {data_type}")


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
    object_quat_list = demo_data["obj_quat"]  # (T, 4)
    object_joints = demo_data["obj_joint"] if "obj_joint" in demo_data else None  # (T, 1)

    object_poses = np.concatenate([object_quat_list, object_pos_list], axis=1)  # (T, 7)
    object_joints = np.array(object_joints) if object_joints is not None else None

    root_positions = demo_data["wrist_pos"]  # (T, 3)
    root_rotations = demo_data["wrist_quat"]  # (T, 4)
    root_rotations = R.from_quat(root_rotations).as_quat(scalar_first=True)
    mano_poses = np.concatenate([root_rotations, root_positions], axis=1)  # (T, 7)

    return mano_poses, mano_joint_coords, object_poses, object_joints


# ---------------------------------------------------------------------------
# Interpolation (30 -> 120 fps) and keypoint loading
# ---------------------------------------------------------------------------
def interpolate_poses(poses, factor):
    """Interpolate (N, 7) pose arrays [quat(4), pos(3)] by ``factor`` using slerp + linear."""
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
    return {key: np.load(value) * scale for key, value in object_keypoints_paths.items()}


def _compute_robot_keypoints(retargeter, retargeted_motions, mano_to_robot_mapping):
    """World-frame robot link positions corresponding to each mapped MANO joint."""
    mano_keypoint_names = list(mano_to_robot_mapping.keys())
    robot_keypoint_link_names = [mano_to_robot_mapping[k] for k in mano_keypoint_names]
    T = retargeted_motions.shape[0]
    robot_keypoints = np.zeros((T, len(robot_keypoint_link_names), 3))
    for t in range(T):
        robot_keypoints[t] = retargeter._get_robot_link_positions(
            retargeted_motions[t], robot_keypoint_link_names
        )
    return robot_keypoints, mano_keypoint_names, robot_keypoint_link_names


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_retargeted_h5(out_path: str, data: dict):
    """Write a flat-dataset regrind retargeted-trajectory ``.h5``."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for key, value in data.items():
            if value is None:
                continue
            f.create_dataset(key, data=np.asarray(value))
    print(f"Saved retargeted trajectory to {out_path}")


def _default_out_path(robot_name: str, object_name: str, scale_suffix: str) -> str:
    data_dir = os.environ["REGRIND_DATA_DIR"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        data_dir, "retargeted_traj", robot_name, f"{object_name}{scale_suffix}", f"retarget_{timestamp}.h5"
    )


def main():
    parser = argparse.ArgumentParser(description="Hand-object interaction-mesh retargeting (1-stage).")
    parser.add_argument("--robot", type=str, default="leaphand", choices=("leaphand", "wujihand"))
    parser.add_argument("--object", type=str, default="scissors", choices=("scissors", "screwdriver"))
    parser.add_argument("--num-timesteps", type=int, default=None, help="Only optimize the first K timesteps.")
    parser.add_argument(
        "--out", type=str, default=None, help="Output .h5 path (default under data/retargeted_traj/...)."
    )
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable Meshcat visualization.")
    parser.set_defaults(visualize=True)
    args = parser.parse_args()

    rc, mano_to_robot = _load_robot_constants_module(args.robot)
    obj_cfg = rc.get_object_config(args.object)

    robot_dof = rc.ROBOT_DOF
    mano_joints = rc.MANO_JOINTS

    # Load and interpolate the demo (30 -> 120 fps).
    mano_poses, mano_joint_coords, object_poses, object_joints = load_demo_data(
        obj_cfg["demo_file"], obj_cfg["demo_data_type"]
    )
    mano_poses, mano_joint_coords, object_poses, object_joints = interpolate_demo(
        mano_poses, mano_joint_coords, object_poses, object_joints, factor=4
    )

    object_keypoints = load_object_keypoints(obj_cfg["object_keypoints_paths"], scale=obj_cfg["obj_scale"])

    hand_keypoint_weight = None
    penetration_tolerance = 0.001
    if args.robot == "wujihand" and args.object == "scissors":
        # De-emphasize the index finger for the Wuji scissors demo.
        hand_keypoint_weight = {k: (0.1 if k in ("1", "2", "3", "17") else 1.0) for k in mano_joints}
        penetration_tolerance = 0.002

    retargeter = HandInteractionMeshOneStageRetargeter(
        robot_model_path=rc.ROBOT_URDF_FILE,
        robot_name=rc.ROBOT_NAME,
        object_model_path=obj_cfg["object_urdf_file"],
        object_name=args.object,
        table_height=obj_cfg["table_height"],
        q_a_init_idx=-7,
        activate_joint_limits=True,
        activate_obj_non_penetration=False,
        demo_joints=mano_joints,
        laplacian_match_links=mano_to_robot,
        hand_keypoint_weight=hand_keypoint_weight,
        penetration_tolerance=penetration_tolerance,
        step_size=0.2,
        visualize=args.visualize,
        debug=False,
        w_nominal_tracking_init=5.0,
    )

    q_a_init = np.concatenate([mano_poses[0], np.zeros(robot_dof)])

    K = args.num_timesteps
    if K is not None:
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
        retargeter.draw_q_knots(retargeted_motions, 1 / 120)

    # Slice the packed solution into the regrind retargeted-trajectory schema.
    # retargeted_motions[t]: [0:4] robot quat (wxyz), [4:7] robot pos, [7:7+DOF] robot joints,
    #   [obj:obj+4] object quat (wxyz), [obj+4:obj+7] object pos, [-1] object joint (if articulated).
    obj_start = 7 + robot_dof
    robot_keypoints, mano_kp_names, robot_kp_links = _compute_robot_keypoints(
        retargeter, retargeted_motions, mano_to_robot
    )
    out_data = {
        "robot_pos": retargeted_motions[:, 4:7],
        "robot_quat": retargeted_motions[:, :4],
        "robot_joints": retargeted_motions[:, 7 : 7 + robot_dof],
        "object_pos": retargeted_motions[:, obj_start + 4 : obj_start + 7],
        "object_quat": retargeted_motions[:, obj_start : obj_start + 4],
        "object_joint": retargeted_motions[:, -1] if object_joints is not None else None,
        "robot_keypoints": robot_keypoints,
        "mano_joint_coords": mano_joint_coords,
    }

    out_path = args.out or _default_out_path(rc.ROBOT_NAME, args.object, obj_cfg["obj_scale_suffix"])
    save_retargeted_h5(out_path, out_data)


if __name__ == "__main__":
    main()
