"""Search upright object yaw angles for strict RB3 wrist-pose IK reachability.

The first object's XY position and bottom-on-table placement remain fixed.  Only
the common camera-to-world rotation about world +Z changes between candidates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLD_TOOL_DIR = PROJECT_ROOT / "tools" / "dexycb_world_transform"
for directory in (PROJECT_ROOT, WORLD_TOOL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from transform_trajectory import (  # noqa: E402
    _decode_scalar,
    _first,
    _load_data,
    _load_mesh,
    _to_wxyz,
    transform_trajectory,
)

from rb3_kinematics import RB3730Kinematics  # noqa: E402


def evaluate_yaw(
    source: dict[str, np.ndarray],
    mesh_vertices: np.ndarray,
    yaw_deg: float,
    desired_xy: tuple[float, float],
    object_model_frame_rpy_deg: tuple[float, float, float],
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    max_nfev: int,
    target_wrist_local_rpy_deg: tuple[float, float, float],
    camera_frame_convention: str = "object_upright",
) -> dict[str, object]:
    order = _decode_scalar(
        source.get("quat_convention", source.get("quaternion_order")), "wxyz"
    ).lower()
    object_pos = _first(source, ("object_pos", "obj_pos"), "object position")
    object_quat = _to_wxyz(
        _first(source, ("object_quat", "obj_quat"), "object quaternion"),
        order,
        "object_quat",
    )
    wrist_pos = _first(source, ("wrist_pos", "robot_pos"), "wrist position")
    wrist_quat = _to_wxyz(
        _first(source, ("wrist_quat", "robot_quat"), "wrist quaternion"),
        order,
        "wrist_quat",
    )
    mano = _first(
        source,
        ("mano_joint_coords", "human_hand_keypoints", "mano_joint_coord"),
        "MANO joint coordinates",
    )
    half_yaw = np.deg2rad(yaw_deg) / 2.0
    desired_object_quat_wxyz = (
        float(np.cos(half_yaw)),
        0.0,
        0.0,
        float(np.sin(half_yaw)),
    )
    world = transform_trajectory(
        object_pos,
        object_quat,
        wrist_pos,
        wrist_quat,
        mano,
        mesh_vertices,
        desired_xy=desired_xy,
        desired_object_quat_wxyz=desired_object_quat_wxyz,
        object_model_frame_rpy_deg=object_model_frame_rpy_deg,
        camera_frame_convention=camera_frame_convention,
        world_yaw_deg=yaw_deg,
    )

    target_pos = world["wrist_pos_world"]
    target_quat_xyzw = world["wrist_quat_world"][:, (1, 2, 3, 0)]
    target_quat_xyzw = (
        Rotation.from_quat(target_quat_xyzw)
        * Rotation.from_euler("XYZ", target_wrist_local_rpy_deg, degrees=True)
    ).as_quat()
    kinematics = RB3730Kinematics()
    neutral_q = np.zeros(6)
    warm_q = neutral_q.copy()
    solutions = np.full((len(target_pos), 6), np.nan)
    success = np.zeros(len(target_pos), dtype=bool)
    position_error = np.full(len(target_pos), np.nan)
    orientation_error = np.full(len(target_pos), np.nan)

    for frame in range(len(target_pos)):
        result = kinematics.inverse(
            target_pos[frame],
            target_quat_xyzw[frame],
            initial_q=warm_q,
            neutral_q=neutral_q,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            max_nfev=max_nfev,
        )
        solutions[frame] = result.q
        success[frame] = result.success
        position_error[frame] = result.position_error_m
        orientation_error[frame] = result.orientation_error_rad
        if result.finite:
            warm_q = result.q.copy()

    step_norm = np.linalg.norm(np.diff(solutions, axis=0), axis=1)
    wrist_radius = np.linalg.norm(target_pos[:, :2], axis=1)
    return {
        "yaw_deg": float(yaw_deg),
        "success_count": int(success.sum()),
        "frame_count": int(len(success)),
        "failed_indices": np.flatnonzero(~success).tolist(),
        "max_position_error_m": float(np.nanmax(position_error)),
        "max_orientation_error_rad": float(np.nanmax(orientation_error)),
        "max_joint_step_rad": float(np.nanmax(step_norm, initial=0.0)),
        "wrist_radius_min_m": float(wrist_radius.min()),
        "wrist_radius_max_m": float(wrist_radius.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Camera-frame retargeting .h5/.npz")
    parser.add_argument("--mesh", required=True, help="YCB object mesh")
    parser.add_argument(
        "--yaw-deg",
        nargs="+",
        type=float,
        default=list(np.arange(-180.0, 180.0, 30.0)),
    )
    parser.add_argument("--desired-x", type=float, default=0.4)
    parser.add_argument("--desired-y", type=float, default=0.0)
    parser.add_argument(
        "--camera-frame-convention",
        choices=("object_upright", "dexycb_y_down"),
        default="object_upright",
    )
    parser.add_argument(
        "--object-model-frame-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--orientation-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-nfev", type=int, default=800)
    parser.add_argument(
        "--target-wrist-local-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
    )
    args = parser.parse_args()

    source = _load_data(args.input)
    mesh = _load_mesh(args.mesh, args.mesh_scale)
    results = []
    for yaw_deg in args.yaw_deg:
        result = evaluate_yaw(
            source,
            np.asarray(mesh.vertices),
            yaw_deg,
            (args.desired_x, args.desired_y),
            tuple(args.object_model_frame_rpy_deg),
            args.position_tolerance,
            args.orientation_tolerance,
            args.max_nfev,
            tuple(args.target_wrist_local_rpy_deg),
            args.camera_frame_convention,
        )
        results.append(result)
        print(
            f"yaw={yaw_deg:+7.1f} deg  "
            f"success={result['success_count']:3d}/{result['frame_count']}  "
            f"max_step={result['max_joint_step_rad']:.4f} rad  "
            f"failed={result['failed_indices']}"
        )

    best = min(
        results,
        key=lambda item: (
            -item["success_count"],
            item["max_joint_step_rad"],
            item["max_position_error_m"],
        ),
    )
    print("\nBest candidate")
    for key, value in best.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
