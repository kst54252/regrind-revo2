"""Transform a DexYCB/REGRIND trajectory from camera to Isaac world coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _load_data(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as h5_file:
            return {key: h5_file[key][()] for key in h5_file}
    raise ValueError(f"unsupported trajectory format: {path.suffix}")


def _write_data(path: str | Path, data: dict) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".npz":
        np.savez_compressed(output, **{key: np.asarray(value) for key, value in data.items()})
    elif output.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(output, "w") as h5_file:
            for key, value in data.items():
                array = np.asarray(value)
                if array.dtype.kind in ("U", "O"):
                    h5_file.create_dataset(
                        key,
                        data=array.astype(object),
                        dtype=h5py.string_dtype("utf-8"),
                    )
                else:
                    h5_file.create_dataset(key, data=array)
    else:
        raise ValueError("output must end in .npz, .h5, or .hdf5")
    return output


def _first(data: dict, names: tuple[str, ...], description: str) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"input has no {description}; tried {names}")


def _decode_scalar(value, default: str) -> str:
    if value is None:
        return default
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def _load_mesh(path: str | Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(path).expanduser().resolve(), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"mesh scene contains no geometry: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices):
        raise ValueError(f"failed to load a non-empty triangle mesh: {path}")
    mesh = loaded.copy()
    mesh.apply_scale(scale)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh vertices contain NaN/Inf")
    return mesh


def _normalize_quaternions(quaternions: np.ndarray, name: str) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"{name} must have shape (T,4), got {quaternions.shape}")
    if not np.isfinite(quaternions).all():
        raise ValueError(f"{name} contains NaN/Inf")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1.0e-12):
        raise ValueError(f"{name} contains a zero-length quaternion")
    return quaternions / norms[:, None]


def _wxyz_to_xyzw(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray(quaternions)[..., (1, 2, 3, 0)]


def _xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray(quaternions)[..., (3, 0, 1, 2)]


def _to_wxyz(quaternions: np.ndarray, order: str, name: str) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    if order == "xyzw":
        quaternions = _xyzw_to_wxyz(quaternions)
    elif order != "wxyz":
        raise ValueError(f"unsupported quaternion order: {order}")
    return _normalize_quaternions(quaternions, name)


def _poses(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    count = len(position)
    transforms = np.repeat(np.eye(4)[None], count, axis=0)
    transforms[:, :3, :3] = Rotation.from_quat(
        _wxyz_to_xyzw(quaternion_wxyz)
    ).as_matrix()
    transforms[:, :3, 3] = position
    return transforms


def _poses_from_transforms(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = transforms[:, :3, 3].copy()
    quaternion_wxyz = _xyzw_to_wxyz(
        Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    )
    return position, quaternion_wxyz


def transform_trajectory(
    object_pos: np.ndarray,
    object_quat_wxyz: np.ndarray,
    wrist_pos: np.ndarray,
    wrist_quat_wxyz: np.ndarray,
    mano_joint_coords: np.ndarray,
    mesh_vertices: np.ndarray,
    desired_xy=(0.50, 0.00),
    desired_object_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    object_model_frame_rpy_deg=(0.0, 0.0, 0.0),
    camera_frame_convention="object_upright",
    world_yaw_deg=0.0,
) -> dict[str, np.ndarray]:
    """Apply one camera-to-world rigid transform to every trajectory frame."""
    object_pos = np.asarray(object_pos, dtype=float)
    wrist_pos = np.asarray(wrist_pos, dtype=float)
    mano_joint_coords = np.asarray(mano_joint_coords, dtype=float)
    mesh_vertices = np.asarray(mesh_vertices, dtype=float)
    T = len(object_pos)
    expected = {
        "object_pos": (T, 3),
        "wrist_pos": (T, 3),
        "mano_joint_coords": (T, 21, 3),
    }
    actual = {
        "object_pos": object_pos.shape,
        "wrist_pos": wrist_pos.shape,
        "mano_joint_coords": mano_joint_coords.shape,
    }
    invalid = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if T == 0:
        raise ValueError("trajectory has no frames")
    if invalid:
        raise ValueError(f"invalid input shapes (actual, expected): {invalid}")
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3 or not len(mesh_vertices):
        raise ValueError(f"mesh_vertices must have shape (N,3), got {mesh_vertices.shape}")
    if not all(
        np.isfinite(array).all()
        for array in (object_pos, wrist_pos, mano_joint_coords, mesh_vertices)
    ):
        raise ValueError("input trajectory or mesh contains NaN/Inf")

    object_quat_wxyz = _normalize_quaternions(object_quat_wxyz, "object_quat")
    wrist_quat_wxyz = _normalize_quaternions(wrist_quat_wxyz, "wrist_quat")
    if object_quat_wxyz.shape[0] != T or wrist_quat_wxyz.shape[0] != T:
        raise ValueError("all pose trajectories must have the same frame count")

    mesh_z_min = float(mesh_vertices[:, 2].min())
    object_model_frame_rpy_deg = np.asarray(
        object_model_frame_rpy_deg, dtype=float
    )
    if (
        object_model_frame_rpy_deg.shape != (3,)
        or not np.isfinite(object_model_frame_rpy_deg).all()
    ):
        raise ValueError("object_model_frame_rpy_deg must contain three finite values")
    object_model_frame_rotation = Rotation.from_euler(
        "XYZ", object_model_frame_rpy_deg, degrees=True
    ).as_matrix()

    T_camera_object = _poses(object_pos, object_quat_wxyz)
    # Fixed model-frame calibration, right-composed with every observed object
    # pose. For the tuna-can sequence, +90 deg about local Y maps the annotated
    # motion-up axis (+X) to the mesh cylinder axis (+Z).
    T_camera_object[:, :3, :3] = (
        T_camera_object[:, :3, :3] @ object_model_frame_rotation
    )
    T_camera_wrist = _poses(wrist_pos, wrist_quat_wxyz)

    if camera_frame_convention == "object_upright":
        desired_quaternion = _normalize_quaternions(
            np.asarray(desired_object_quat_wxyz, dtype=float).reshape(1, 4),
            "desired_object_quat_wxyz",
        )[0]
        desired_rotation = Rotation.from_quat(
            _wxyz_to_xyzw(desired_quaternion)
        ).as_matrix()
        if not np.allclose(desired_rotation[:, 2], (0.0, 0.0, 1.0), atol=1.0e-8):
            raise ValueError(
                "desired object orientation must be upright: local +Z must map to world +Z"
            )
        desired_position = np.array(
            (float(desired_xy[0]), float(desired_xy[1]), -mesh_z_min), dtype=float
        )
        T_world_object_desired = np.eye(4)
        T_world_object_desired[:3, :3] = desired_rotation
        T_world_object_desired[:3, 3] = desired_position
        T_world_camera = T_world_object_desired @ np.linalg.inv(T_camera_object[0])
    elif camera_frame_convention == "dexycb_y_down":
        # DexYCB uses an optical camera frame: +X right, +Y down, +Z forward.
        # Keep gravity independent of the grasped object's local axes:
        # camera -Y is Isaac world +Z.  The yaw only rotates the table plane.
        base_rotation = np.array(
            ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
            dtype=float,
        )
        yaw_rotation = Rotation.from_euler(
            "Z", float(world_yaw_deg), degrees=True
        ).as_matrix()
        world_camera_rotation = yaw_rotation @ base_rotation
        rotated_object_position = world_camera_rotation @ T_camera_object[0, :3, 3]
        first_object_rotation = world_camera_rotation @ T_camera_object[0, :3, :3]
        first_mesh_relative = mesh_vertices @ first_object_rotation.T
        desired_position = np.array(
            (
                float(desired_xy[0]),
                float(desired_xy[1]),
                -float(first_mesh_relative[:, 2].min()),
            ),
            dtype=float,
        )
        T_world_camera = np.eye(4)
        T_world_camera[:3, :3] = world_camera_rotation
        T_world_camera[:3, 3] = desired_position - rotated_object_position
        T_world_object_desired = T_world_camera @ T_camera_object[0]
    else:
        raise ValueError(
            "camera_frame_convention must be 'object_upright' or 'dexycb_y_down', "
            f"got {camera_frame_convention!r}"
        )

    T_world_object = T_world_camera[None] @ T_camera_object
    T_world_wrist = T_world_camera[None] @ T_camera_wrist
    object_pos_world, object_quat_world = _poses_from_transforms(T_world_object)
    wrist_pos_world, wrist_quat_world = _poses_from_transforms(T_world_wrist)
    R_world_camera = T_world_camera[:3, :3]
    t_world_camera = T_world_camera[:3, 3]
    mano_joint_world = (
        mano_joint_coords @ R_world_camera.T + t_world_camera[None, None, :]
    )

    mesh_world_first = mesh_vertices @ T_world_object[0, :3, :3].T + T_world_object[0, :3, 3]
    min_world_z = float(mesh_world_first[:, 2].min())
    determinant = float(np.linalg.det(R_world_camera))
    arrays = (
        object_pos_world,
        object_quat_world,
        wrist_pos_world,
        wrist_quat_world,
        mano_joint_world,
        T_world_camera,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("world trajectory contains NaN/Inf after transformation")

    return {
        "object_pos_world": object_pos_world,
        "object_quat_world": object_quat_world,
        "wrist_pos_world": wrist_pos_world,
        "wrist_quat_world": wrist_quat_world,
        "mano_joint_world": mano_joint_world,
        "T_world_camera": T_world_camera,
        "T_world_object_desired": T_world_object_desired,
        "mesh_local_bounds": np.stack((mesh_vertices.min(axis=0), mesh_vertices.max(axis=0))),
        "mesh_z_min": np.asarray(mesh_z_min),
        "first_mesh_world_min_z": np.asarray(min_world_z),
        "R_world_camera_determinant": np.asarray(determinant),
        "object_model_frame_rpy_deg": object_model_frame_rpy_deg,
        "camera_frame_convention": np.asarray(camera_frame_convention),
        "world_yaw_deg": np.asarray(float(world_yaw_deg)),
        "object_model_frame_quat_wxyz": _xyzw_to_wxyz(
            Rotation.from_matrix(object_model_frame_rotation).as_quat()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Camera-frame trajectory .h5/.hdf5/.npz")
    parser.add_argument("--mesh", required=True, help="YCB textured_simple.obj")
    parser.add_argument("--out", required=True, help="New world-frame .h5/.hdf5/.npz")
    parser.add_argument(
        "--input-quat-order",
        choices=("auto", "wxyz", "xyzw"),
        default="auto",
        help="auto uses file metadata and falls back to wxyz.",
    )
    parser.add_argument("--desired-x", type=float, default=0.50)
    parser.add_argument("--desired-y", type=float, default=0.00)
    parser.add_argument(
        "--camera-frame-convention",
        choices=("object_upright", "dexycb_y_down"),
        default="object_upright",
        help=(
            "object_upright preserves the original object-relative alignment; "
            "dexycb_y_down uses the fixed optical-camera gravity convention "
            "camera -Y = Isaac world +Z."
        ),
    )
    parser.add_argument(
        "--world-yaw-deg",
        type=float,
        default=0.0,
        help="Table-plane yaw used with dexycb_y_down.",
    )
    parser.add_argument(
        "--desired-object-quat-wxyz",
        nargs=4,
        type=float,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Explicit upright first-frame object orientation.",
    )
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument(
        "--object-model-frame-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Fixed intrinsic XYZ rotation right-composed with every input object "
            "pose before world alignment. For this tuna-can pickup use '0 90 0' "
            "to map the observed local +X lift direction to mesh local +Z."
        ),
    )
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()
    if args.mesh_scale <= 0 or args.validation_tolerance <= 0:
        raise ValueError("mesh scale and validation tolerance must be > 0")

    source = _load_data(args.input)
    order = args.input_quat_order
    if order == "auto":
        order = _decode_scalar(
            source.get("quat_convention", source.get("quaternion_order")), "wxyz"
        ).lower()
    if order not in ("wxyz", "xyzw"):
        raise ValueError(f"input quaternion metadata must be wxyz or xyzw, got {order!r}")

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
    mesh = _load_mesh(args.mesh, args.mesh_scale)
    result = transform_trajectory(
        object_pos,
        object_quat,
        wrist_pos,
        wrist_quat,
        mano,
        np.asarray(mesh.vertices),
        desired_xy=(args.desired_x, args.desired_y),
        desired_object_quat_wxyz=args.desired_object_quat_wxyz,
        object_model_frame_rpy_deg=args.object_model_frame_rpy_deg,
        camera_frame_convention=args.camera_frame_convention,
        world_yaw_deg=args.world_yaw_deg,
    )

    tolerance = args.validation_tolerance
    first_position = result["object_pos_world"][0]
    checks = {
        "first_object_x": abs(first_position[0] - args.desired_x) <= tolerance,
        "first_object_y": abs(first_position[1] - args.desired_y) <= tolerance,
        "mesh_bottom_z": abs(float(result["first_mesh_world_min_z"])) <= tolerance,
        "proper_rotation": abs(float(result["R_world_camera_determinant"]) - 1.0) <= tolerance,
        "finite": all(
            np.isfinite(value).all()
            for value in result.values()
            if np.asarray(value).dtype.kind not in ("U", "S", "O")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"world transform validation failed: {checks}")

    result.update(
        {
            "quaternion_order": "wxyz",
            "input_quaternion_order": order,
            "source_trajectory_file": str(Path(args.input).expanduser().resolve()),
            "source_mesh_file": str(Path(args.mesh).expanduser().resolve()),
            "mesh_scale": args.mesh_scale,
            "frame_index": np.asarray(source.get("frame_index", np.arange(len(object_pos)))),
            "fps": np.asarray(source.get("fps", 30.0)).item(),
        }
    )
    # Preserve the original sequential MANO21 topology separately from the
    # Revo2-semantic MANO order used by the optimizer.
    if "mano_joint_coords_mano21" in source:
        mano21 = np.asarray(source["mano_joint_coords_mano21"], dtype=float)
        if mano21.shape != mano.shape:
            raise ValueError(
                "mano_joint_coords_mano21 must have shape "
                f"{mano.shape}, got {mano21.shape}"
            )
        if not np.isfinite(mano21).all():
            raise ValueError("mano_joint_coords_mano21 contains NaN/Inf")
        result["mano_joint_world_mano21"] = (
            mano21 @ result["T_world_camera"][:3, :3].T
            + result["T_world_camera"][:3, 3][None, None, :]
        )
    # Joint angles and local object samples are frame-invariant under this
    # coordinate conversion and are useful to the later IK/replay stages.
    for output_name, aliases in {
        "revo2_joints": ("revo2_joints", "robot_joints"),
        "object_points_local": ("object_points_local",),
    }.items():
        for alias in aliases:
            if alias in source:
                result[output_name] = np.asarray(source[alias])
                break
    for name in ("source_frame_indices", "source_camera_serial"):
        if name in source:
            result[name] = np.asarray(source[name])

    output = _write_data(args.out, result)
    print("[DexYCB camera -> Isaac/RB3 world]")
    print(f"  input quaternion order:       {order}")
    print(f"  output quaternion order:      wxyz")
    print(f"  object model-frame RPY:       {result['object_model_frame_rpy_deg']} deg")
    print(f"  mesh local bounds [m]:\n{result['mesh_local_bounds']}")
    print(f"  mesh z_min:                   {float(result['mesh_z_min']):.9g} m")
    print(f"  first object position:        {first_position}")
    print(f"  first mesh world min Z:       {float(result['first_mesh_world_min_z']):.9g} m")
    print(f"  det(R_world_camera):          {float(result['R_world_camera_determinant']):.12g}")
    print(f"  finite values:                {checks['finite']}")
    print(f"  T_world_camera:\n{result['T_world_camera']}")
    print(f"Saved world-frame trajectory to {output}")


if __name__ == "__main__":
    main()
