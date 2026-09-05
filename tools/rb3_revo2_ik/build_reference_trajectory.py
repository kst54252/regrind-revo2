"""Convert a REGRIND Revo2 wrist trajectory into an RB3+Revo2 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .rb3_kinematics import RB3730Kinematics
    from .reference_trajectory import DEFAULT_REVO2_JOINT_NAMES
except ImportError:  # Support direct execution.
    from rb3_kinematics import RB3730Kinematics
    from reference_trajectory import DEFAULT_REVO2_JOINT_NAMES


WORKCELL_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "workcell"
    / "rb3_revo2_table.json"
)
DEFAULT_OBJECT_MESH = (
    Path(__file__).resolve().parents[2]
    / "007_tuna_fish_can"
    / "textured_simple.obj"
)


def _default_rb3_base_position() -> tuple[float, float, float]:
    with WORKCELL_CONFIG.open("r", encoding="utf-8") as config_file:
        layout = json.load(config_file)
    return tuple(float(value) for value in layout["robot_mount"]["position"])


def _load(path: str) -> dict[str, np.ndarray]:
    if path.lower().endswith(".npz"):
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}
    with h5py.File(path, "r") as h5_file:
        return {name: h5_file[name][()] for name in h5_file}


def _first(data: dict, names: tuple[str, ...], description: str):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"input has no {description}; tried {names}")


def _decode_scalar(value, default: str) -> str:
    if value is None:
        return default
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def _to_xyzw(quaternion: np.ndarray, convention: str) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    if convention == "xyzw":
        return quaternion
    if convention == "wxyz":
        return quaternion[..., [1, 2, 3, 0]]
    raise ValueError(f"unsupported quaternion convention: {convention}")


def _align_trajectory_to_object_start(
    wrist_pos: np.ndarray,
    wrist_quat: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    desired_position: np.ndarray,
    desired_quaternion_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Rotation, np.ndarray]:
    """Apply one rigid transform so frame-zero object reaches a desired pose."""

    source_rotation = Rotation.from_quat(object_quat[0])
    desired_rotation = Rotation.from_quat(desired_quaternion_xyzw)
    world_delta_rotation = desired_rotation * source_rotation.inv()
    world_delta_translation = desired_position - world_delta_rotation.apply(object_pos[0])

    aligned_wrist_pos = world_delta_rotation.apply(wrist_pos) + world_delta_translation
    aligned_object_pos = world_delta_rotation.apply(object_pos) + world_delta_translation
    aligned_wrist_quat = (
        world_delta_rotation * Rotation.from_quat(wrist_quat)
    ).as_quat()
    aligned_object_quat = (
        world_delta_rotation * Rotation.from_quat(object_quat)
    ).as_quat()
    return (
        aligned_wrist_pos,
        aligned_wrist_quat,
        aligned_object_pos,
        aligned_object_quat,
        world_delta_rotation,
        world_delta_translation,
    )


def _stable_upright_pose_on_table(
    object_position: np.ndarray,
    object_quaternion_xyzw: np.ndarray,
    mesh_vertices: np.ndarray,
    *,
    desired_xy: np.ndarray | None = None,
    table_height: float = 0.0,
    clearance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the closest exactly-upright first pose with its mesh above a table.

    Some DexYCB tuna poses are almost vertical but lean by several degrees. A
    dynamic cylinder cannot remain in that pose once gravity starts, so PhysX
    immediately topples it. Preserve the current horizontal heading and the
    sign of the mesh's local Z axis, remove only roll/pitch, and compute the
    origin height from the *rotated mesh* rather than an axis-aligned bound.
    """

    position = np.asarray(object_position, dtype=float)
    quaternion = np.asarray(object_quaternion_xyzw, dtype=float)
    vertices = np.asarray(mesh_vertices, dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("object position/quaternion must have shape (3,)/(4,)")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"mesh_vertices must have shape (N,3), got {vertices.shape}")
    if not all(np.isfinite(value).all() for value in (position, quaternion, vertices)):
        raise ValueError("object pose and mesh vertices must be finite")
    if not np.isfinite(table_height) or not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("table height must be finite and clearance must be >= 0")

    source_rotation = Rotation.from_quat(quaternion).as_matrix()
    z_sign = 1.0 if source_rotation[2, 2] >= 0.0 else -1.0
    z_axis = np.asarray([0.0, 0.0, z_sign])
    x_axis = source_rotation[:, 0] - z_axis * np.dot(source_rotation[:, 0], z_axis)
    if np.linalg.norm(x_axis) < 1.0e-10:
        y_axis = source_rotation[:, 1] - z_axis * np.dot(source_rotation[:, 1], z_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
    else:
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
    target_rotation = np.column_stack((x_axis, y_axis, z_axis))
    if not np.isclose(np.linalg.det(target_rotation), 1.0, atol=1.0e-10):
        raise RuntimeError("failed to construct a proper upright object rotation")

    rotated_vertices = vertices @ target_rotation.T
    mesh_min_relative_z = float(rotated_vertices[:, 2].min())
    target_position = position.copy()
    if desired_xy is not None:
        desired_xy = np.asarray(desired_xy, dtype=float)
        if desired_xy.shape != (2,) or not np.isfinite(desired_xy).all():
            raise ValueError("desired_xy must contain two finite values")
        target_position[:2] = desired_xy
    target_position[2] = float(table_height) + float(clearance) - mesh_min_relative_z
    initial_tilt_rad = float(
        np.arccos(np.clip(abs(source_rotation[2, 2]), 0.0, 1.0))
    )
    return target_position, Rotation.from_matrix(target_rotation).as_quat(), initial_tilt_rad


def _write(path: str, data: dict):
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
        raise ValueError("output must end in .h5, .hdf5, or .npz")
    print(f"Saved RB3+Revo2 reference trajectory to {output}")


def _default_output(input_path: str) -> str:
    path = Path(input_path).expanduser().resolve()
    return str(path.with_name(path.stem + "_rb3_revo2_reference.h5"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="REGRIND retargeting .h5/.hdf5/.npz")
    parser.add_argument("--out", help="Output .h5/.hdf5/.npz")
    parser.add_argument(
        "--input-quat-convention",
        choices=("auto", "xyzw", "wxyz"),
        default="auto",
    )
    parser.add_argument(
        "--initial-q", nargs=6, type=float, default=np.zeros(6), metavar="RAD",
        help="Neutral/current RB3 configuration used for frame 0.",
    )
    parser.add_argument(
        "--base-position", nargs=3, type=float, default=_default_rb3_base_position(), metavar="M",
        help=(
            "RB3 root position in the table-relative REGRIND world frame. "
            "Default comes from config/workcell/rb3_revo2_table.json."
        ),
    )
    parser.add_argument(
        "--base-quat-xyzw", nargs=4, type=float, default=(0.0, 0.0, 0.0, 1.0),
        help="RB3 root orientation in the REGRIND world frame.",
    )
    parser.add_argument(
        "--target-wrist-local-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Fixed intrinsic local XYZ frame correction right-composed with "
            "every target wrist orientation before IK. Use '0 0 180' when "
            "the mounted Revo2 palm frame is upside down relative to REGRIND."
        ),
    )
    parser.add_argument(
        "--object-start-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Rigidly move/rotate the complete floating rollout so its frame-zero "
            "object origin reaches this RB3-world position."
        ),
    )
    parser.add_argument(
        "--object-start-quat-xyzw",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "W"),
        help=(
            "Desired frame-zero object orientation. If omitted while position is "
            "specified, the rollout's initial object orientation is preserved."
        ),
    )
    parser.add_argument(
        "--level-object-on-table",
        action="store_true",
        help=(
            "Remove frame-zero can roll/pitch, preserve its heading/Z-axis sign, "
            "and rigidly transform the complete hand/object trajectory before IK."
        ),
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=DEFAULT_OBJECT_MESH,
        help="Mesh used to place the leveled object bottom on the table.",
    )
    parser.add_argument("--table-height", type=float, default=0.0)
    parser.add_argument(
        "--object-clearance",
        type=float,
        default=0.0,
        help="Initial clearance above the table in meters (default: exact contact).",
    )
    parser.add_argument(
        "--drop-leading-frames",
        type=int,
        default=0,
        help=(
            "Drop this many leading rollout frames before alignment and IK. "
            "Useful for physics-reset settling frames."
        ),
    )
    parser.add_argument("--position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--orientation-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--position-weight", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=800)
    parser.add_argument(
        "--verbose-frames", action="store_true",
        help="Print target and FK wrist poses for every frame.",
    )
    args = parser.parse_args()
    if args.position_tolerance <= 0 or args.orientation_tolerance <= 0:
        raise ValueError("IK tolerances must be > 0")

    source = _load(args.input)
    wrist_pos = _first(
        source, ("wrist_pos_world", "wrist_pos", "robot_pos"), "wrist position"
    )
    wrist_quat = _first(
        source,
        ("wrist_quat_world", "wrist_quat", "robot_quat"),
        "wrist quaternion",
    )
    revo2_joints = _first(
        source, ("revo2_joints", "robot_joints"), "Revo2 joint trajectory"
    )
    revo2_follower_joints = source.get("revo2_follower_joints")
    revo2_joint_drive_target = source.get("revo2_joint_drive_target")
    object_pos = _first(
        source, ("object_pos_world", "object_pos", "obj_pos"), "object position"
    )
    object_quat = _first(
        source,
        ("object_quat_world", "object_quat", "obj_quat"),
        "object quaternion",
    )
    convention = args.input_quat_convention
    if convention == "auto":
        convention = _decode_scalar(
            source.get("quat_convention", source.get("quaternion_order")), "xyzw"
        )
    wrist_quat = _to_xyzw(wrist_quat, convention)
    object_quat = _to_xyzw(object_quat, convention)
    source_frame_count = len(wrist_pos)
    if revo2_follower_joints is not None:
        revo2_follower_joints = np.asarray(revo2_follower_joints, dtype=float)
        if revo2_follower_joints.shape != (source_frame_count, 5):
            raise ValueError(
                "revo2_follower_joints must have shape "
                f"{(source_frame_count, 5)}, got {revo2_follower_joints.shape}"
            )
    if revo2_joint_drive_target is not None:
        revo2_joint_drive_target = np.asarray(revo2_joint_drive_target, dtype=float)
        if revo2_joint_drive_target.shape != (source_frame_count, 6):
            raise ValueError(
                "revo2_joint_drive_target must have shape "
                f"{(source_frame_count, 6)}, got {revo2_joint_drive_target.shape}"
            )
    if any(
        value is not None and not np.isfinite(value).all()
        for value in (revo2_follower_joints, revo2_joint_drive_target)
    ):
        raise ValueError("Revo2 follower state/drive target contains NaN/Inf")
    if args.drop_leading_frames < 0 or args.drop_leading_frames >= source_frame_count:
        raise ValueError(
            "--drop-leading-frames must be in "
            f"[0,{source_frame_count - 1}], got {args.drop_leading_frames}"
        )
    source_start_frame = int(args.drop_leading_frames)
    if source_start_frame:
        source_slice = slice(source_start_frame, None)
        wrist_pos = wrist_pos[source_slice]
        wrist_quat = wrist_quat[source_slice]
        revo2_joints = revo2_joints[source_slice]
        if revo2_follower_joints is not None:
            revo2_follower_joints = revo2_follower_joints[source_slice]
        if revo2_joint_drive_target is not None:
            revo2_joint_drive_target = revo2_joint_drive_target[source_slice]
        object_pos = object_pos[source_slice]
        object_quat = object_quat[source_slice]
        print(
            f"[input trim] dropped {source_start_frame} physics-settling frame(s); "
            f"{source_frame_count} -> {len(wrist_pos)}"
        )
    if args.level_object_on_table and args.object_start_quat_xyzw is not None:
        raise ValueError(
            "--level-object-on-table and --object-start-quat-xyzw are mutually exclusive"
        )
    if args.object_clearance < 0.0 or not np.isfinite(args.object_clearance):
        raise ValueError("--object-clearance must be finite and >= 0")
    if not np.isfinite(args.table_height):
        raise ValueError("--table-height must be finite")

    leveled_initial_tilt_rad = 0.0
    leveled_mesh_min_z = np.nan
    leveled_mesh_path = ""
    leveled_position = None
    leveled_quaternion = None
    if args.level_object_on_table:
        import trimesh

        mesh_path = args.object_mesh.expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"object mesh not found: {mesh_path}")
        loaded_mesh = trimesh.load(mesh_path, force="scene", process=False)
        if isinstance(loaded_mesh, trimesh.Scene):
            geometries = tuple(loaded_mesh.geometry.values())
            if not geometries:
                raise ValueError(f"object mesh contains no geometry: {mesh_path}")
            loaded_mesh = trimesh.util.concatenate(geometries)
        desired_xy = (
            np.asarray(args.object_start_position, dtype=float)[:2]
            if args.object_start_position is not None
            else object_pos[0, :2]
        )
        leveled_position, leveled_quaternion, leveled_initial_tilt_rad = (
            _stable_upright_pose_on_table(
                object_pos[0],
                object_quat[0],
                np.asarray(loaded_mesh.vertices, dtype=float),
                desired_xy=desired_xy,
                table_height=args.table_height,
                clearance=args.object_clearance,
            )
        )
        leveled_mesh_min_z = float(
            (
                Rotation.from_quat(leveled_quaternion).apply(
                    np.asarray(loaded_mesh.vertices, dtype=float)
                )
                + leveled_position
            )[:, 2].min()
        )
        leveled_mesh_path = str(mesh_path)

    alignment_rotation = Rotation.identity()
    alignment_translation = np.zeros(3, dtype=float)
    alignment_applied = (
        args.level_object_on_table
        or args.object_start_position is not None
        or args.object_start_quat_xyzw is not None
    )
    if alignment_applied:
        desired_object_position = np.asarray(
            leveled_position
            if leveled_position is not None
            else (
                args.object_start_position
                if args.object_start_position is not None
                else object_pos[0]
            ),
            dtype=float,
        )
        desired_object_quaternion = np.asarray(
            leveled_quaternion
            if leveled_quaternion is not None
            else (
                args.object_start_quat_xyzw
                if args.object_start_quat_xyzw is not None
                else object_quat[0]
            ),
            dtype=float,
        )
        if not np.isfinite(desired_object_position).all() or desired_object_position.shape != (3,):
            raise ValueError("--object-start-position must contain three finite values")
        if not np.isfinite(desired_object_quaternion).all() or desired_object_quaternion.shape != (4,):
            raise ValueError("--object-start-quat-xyzw must contain four finite values")
        quaternion_norm = np.linalg.norm(desired_object_quaternion)
        if quaternion_norm < 1.0e-12:
            raise ValueError("--object-start-quat-xyzw cannot be a zero quaternion")
        desired_object_quaternion /= quaternion_norm
        (
            wrist_pos,
            wrist_quat,
            object_pos,
            object_quat,
            alignment_rotation,
            alignment_translation,
        ) = _align_trajectory_to_object_start(
            wrist_pos,
            wrist_quat,
            object_pos,
            object_quat,
            desired_object_position,
            desired_object_quaternion,
        )
        print("[floating rollout alignment]")
        print(f"  desired object frame 0 position: {desired_object_position}")
        print(f"  actual object frame 0 position:  {object_pos[0]}")
        print(f"  actual object frame 0 quat xyzw: {object_quat[0]}")
        if args.level_object_on_table:
            print(
                f"  removed initial can tilt:         "
                f"{np.degrees(leveled_initial_tilt_rad):.6g} deg"
            )
            print(
                f"  leveled mesh minimum Z:           "
                f"{leveled_mesh_min_z:.9g} m"
            )
    wrist_frame_correction_rpy_deg = np.asarray(
        args.target_wrist_local_rpy_deg, dtype=float
    )
    if (
        wrist_frame_correction_rpy_deg.shape != (3,)
        or not np.isfinite(wrist_frame_correction_rpy_deg).all()
    ):
        raise ValueError("--target-wrist-local-rpy-deg must contain three finite values")
    wrist_frame_correction = Rotation.from_euler(
        "XYZ", wrist_frame_correction_rpy_deg, degrees=True
    )
    wrist_quat = (
        Rotation.from_quat(wrist_quat) * wrist_frame_correction
    ).as_quat()

    T = len(wrist_pos)
    if T == 0:
        raise ValueError("input trajectory has no frames")
    expected = {
        "wrist_pos": (T, 3),
        "wrist_quat": (T, 4),
        "revo2_joints": (T, 6),
        "object_pos": (T, 3),
        "object_quat": (T, 4),
    }
    actual = {
        "wrist_pos": wrist_pos.shape,
        "wrist_quat": wrist_quat.shape,
        "revo2_joints": revo2_joints.shape,
        "object_pos": object_pos.shape,
        "object_quat": object_quat.shape,
    }
    bad = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if bad:
        raise ValueError(f"invalid input trajectory shapes (actual, expected): {bad}")
    if not all(
        np.isfinite(value).all()
        for value in (
            wrist_pos,
            wrist_quat,
            revo2_joints,
            object_pos,
            object_quat,
        )
    ):
        raise ValueError("input wrist/Revo2/object trajectory contains NaN/Inf")

    kinematics = RB3730Kinematics(
        base_position=args.base_position,
        base_quaternion_xyzw=args.base_quat_xyzw,
    )
    initial_q = np.asarray(args.initial_q, dtype=float)
    lower, upper = kinematics.get_joint_limits()
    if initial_q.shape != (6,) or not np.isfinite(initial_q).all():
        raise ValueError("--initial-q must contain six finite values")
    if np.any(initial_q < lower) or np.any(initial_q > upper):
        raise ValueError("--initial-q is outside RB3 joint limits")

    rb3_joints = np.full((T, 6), np.nan)
    ik_success = np.zeros(T, dtype=bool)
    optimizer_success = np.zeros(T, dtype=bool)
    position_error = np.full(T, np.nan)
    orientation_error = np.full(T, np.nan)
    fk_wrist_pos = np.full((T, 3), np.nan)
    fk_wrist_quat = np.full((T, 4), np.nan)
    joint_limit_violation = np.ones(T, dtype=bool)
    max_joint_limit_violation = np.full(T, np.nan)
    finite_solution = np.zeros(T, dtype=bool)
    solver_cost = np.full(T, np.nan)
    solver_nfev = np.zeros(T, dtype=int)
    solver_message = []
    warm_start_frame = np.full(T, -1, dtype=int)

    warm_q = initial_q.copy()
    last_finite_frame = -1
    for frame in range(T):
        warm_start_frame[frame] = last_finite_frame
        try:
            result = kinematics.inverse(
                wrist_pos[frame],
                wrist_quat[frame],
                initial_q=warm_q,
                neutral_q=initial_q,
                position_tolerance_m=args.position_tolerance,
                orientation_tolerance_rad=args.orientation_tolerance,
                position_weight=args.position_weight,
                max_nfev=args.max_nfev,
            )
            rb3_joints[frame] = result.q
            ik_success[frame] = result.success
            optimizer_success[frame] = result.optimizer_success
            position_error[frame] = result.position_error_m
            orientation_error[frame] = result.orientation_error_rad
            fk_wrist_pos[frame] = result.fk_position
            fk_wrist_quat[frame] = result.fk_quaternion_xyzw
            joint_limit_violation[frame] = result.joint_limit_violation
            max_joint_limit_violation[frame] = result.max_joint_limit_violation_rad
            finite_solution[frame] = result.finite
            solver_cost[frame] = result.cost
            solver_nfev[frame] = result.nfev
            solver_message.append(result.message)
            if result.finite:
                warm_q = result.q.copy()
                last_finite_frame = frame
        except Exception as error:
            solver_message.append(f"{type(error).__name__}: {error}")

        status = "OK" if ik_success[frame] else "FAIL"
        print(
            f"frame {frame:04d} {status} "
            f"pos_err={position_error[frame]:.6g} m "
            f"ori_err={orientation_error[frame]:.6g} rad "
            f"limit={joint_limit_violation[frame]} finite={finite_solution[frame]}"
        )
        if args.verbose_frames:
            print(f"  target pos:  {wrist_pos[frame]}")
            print(f"  target quat: {wrist_quat[frame]} (xyzw)")
            print(f"  FK pos:      {fk_wrist_pos[frame]}")
            print(f"  FK quat:     {fk_wrist_quat[frame]} (xyzw)")

    reference_joints = np.concatenate((rb3_joints, revo2_joints), axis=1)
    rb3_joint_step_norm = np.concatenate(
        ([0.0], np.linalg.norm(np.diff(rb3_joints, axis=0), axis=1))
    )
    failed_indices = np.flatnonzero(~ik_success)
    finite_error = np.isfinite(position_error) & np.isfinite(orientation_error)
    print("\n[RB3 IK sequence summary]")
    print(f"  frames:                     {T}")
    print(f"  IK success rate:            {100.0 * ik_success.mean():.2f}%")
    print(f"  failed frame indices:       {failed_indices.tolist()}")
    if finite_error.any():
        print(
            "  mean / max position error:  "
            f"{position_error[finite_error].mean():.9g} / "
            f"{position_error[finite_error].max():.9g} m"
        )
        print(
            "  mean / max orientation err: "
            f"{orientation_error[finite_error].mean():.9g} / "
            f"{orientation_error[finite_error].max():.9g} rad"
        )
    print("  RB3 joint ranges [rad]:")
    finite_q = np.isfinite(rb3_joints).all(axis=1)
    for index, name in enumerate(kinematics.joint_names):
        if finite_q.any():
            print(
                f"    {name:<12} {rb3_joints[finite_q, index].min(): .6f} .. "
                f"{rb3_joints[finite_q, index].max(): .6f}"
            )
        else:
            print(f"    {name:<12} n/a")
    print(f"  max RB3 joint step norm:   {np.nanmax(rb3_joint_step_norm):.9g} rad")

    output_data = {
        "quat_convention": "xyzw",
        "fps": np.asarray(source.get("fps", 30.0)).item(),
        "frame_index": np.asarray(
            source.get("frame_index", np.arange(source_frame_count))
        )[source_start_frame:],
        "source_start_frame": source_start_frame,
        "rb3_joint_names": np.asarray(kinematics.joint_names),
        "revo2_joint_names": np.asarray(
            source.get(
                "actuated_joint_names",
                DEFAULT_REVO2_JOINT_NAMES,
            )
        ).astype(str),
        "reference_joint_order": "rb3_first_then_revo2",
        "rb3_joints": rb3_joints,
        "revo2_joints": revo2_joints,
        "reference_joints": reference_joints,
        "rb3_joint_step_norm_rad": rb3_joint_step_norm,
        "wrist_pos": wrist_pos,
        "wrist_quat": wrist_quat,
        "object_pos": object_pos,
        "object_quat": object_quat,
        "target_wrist_pos": wrist_pos,
        "target_wrist_quat": wrist_quat,
        "fk_wrist_pos": fk_wrist_pos,
        "fk_wrist_quat": fk_wrist_quat,
        "ik_success": ik_success,
        "optimizer_success": optimizer_success,
        "failed_frame_indices": failed_indices,
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "joint_limit_violation": joint_limit_violation,
        "max_joint_limit_violation_rad": max_joint_limit_violation,
        "finite_solution": finite_solution,
        "solver_cost": solver_cost,
        "solver_nfev": solver_nfev,
        "solver_message": np.asarray(solver_message),
        "warm_start_frame": warm_start_frame,
        "rb3_joint_lower_rad": lower,
        "rb3_joint_upper_rad": upper,
        "rb3_base_position": np.asarray(args.base_position),
        "rb3_base_quat_xyzw": np.asarray(args.base_quat_xyzw),
        "ik_position_tolerance_m": args.position_tolerance,
        "ik_orientation_tolerance_rad": args.orientation_tolerance,
        "source_retargeting_file": str(Path(args.input).expanduser().resolve()),
        "source_robot_usd": str(kinematics.source_usd_path),
        "mounted_wrist_frame": kinematics.mounted_wrist_frame,
        "link6_to_wrist_xyz_m": kinematics.link6_to_wrist_position,
        "target_wrist_local_rpy_correction_deg": wrist_frame_correction_rpy_deg,
        "target_wrist_local_quat_xyzw_correction": wrist_frame_correction.as_quat(),
        "floating_alignment_applied": alignment_applied,
        "floating_alignment_rotation": alignment_rotation.as_matrix(),
        "floating_alignment_translation": alignment_translation,
        "object_leveled_on_table": args.level_object_on_table,
        "object_leveling_initial_tilt_rad": leveled_initial_tilt_rad,
        "object_leveling_table_height_m": args.table_height,
        "object_leveling_clearance_m": args.object_clearance,
        "object_leveling_mesh_world_min_z_m": leveled_mesh_min_z,
        "object_leveling_mesh_path": leveled_mesh_path,
    }
    if revo2_follower_joints is not None:
        output_data["revo2_follower_joints"] = revo2_follower_joints
        output_data["revo2_follower_joint_names"] = np.asarray(
            source.get("revo2_follower_joint_names", ()), dtype=str
        )
    if revo2_joint_drive_target is not None:
        output_data["revo2_joint_drive_target"] = revo2_joint_drive_target
    if "revo2_fingertip_pos" in source:
        fingertips = np.asarray(source["revo2_fingertip_pos"], dtype=float)
        if fingertips.shape != (source_frame_count, 5, 3) or not np.isfinite(fingertips).all():
            raise ValueError("revo2_fingertip_pos must be finite and have shape (T,5,3)")
        fingertips = fingertips[source_start_frame:]
        output_data["revo2_fingertip_pos"] = (
            alignment_rotation.apply(fingertips.reshape(-1, 3)) + alignment_translation
        ).reshape(T, 5, 3)
    mano_joint_world = next(
        (
            np.asarray(source[name])
            for name in (
                "mano_joint_world_mano21",
                "mano_joint_world",
                "mano_joint_coords_world",
                "mano_joint_coords",
                "human_hand_keypoints",
            )
            if name in source
        ),
        None,
    )
    if mano_joint_world is not None:
        if mano_joint_world.shape == (source_frame_count, 21, 3):
            mano_joint_world = mano_joint_world[source_start_frame:]
        if mano_joint_world.shape != (T, 21, 3):
            raise ValueError(
                "MANO skeleton must have shape "
                f"{(T, 21, 3)}, got {mano_joint_world.shape}"
            )
        if not np.isfinite(mano_joint_world).all():
            raise ValueError("MANO skeleton contains NaN/Inf")
        if alignment_applied:
            mano_joint_world = (
                alignment_rotation.apply(mano_joint_world.reshape(-1, 3))
                + alignment_translation
            ).reshape(T, 21, 3)
        output_data["mano_joint_world"] = mano_joint_world
        output_data["mano_joint_order"] = _decode_scalar(
            source.get("mano_joint_order"),
            "mano21_sequential_thumb_index_middle_ring_little",
        )
    # Keep the configured mount quaternion verbatim; FK stores its rotation matrix.
    output_data["link6_to_wrist_quat_xyzw"] = np.asarray(
        kinematics.config["link6_to_mounted_wrist_quat_xyzw"], dtype=float
    )
    _write(args.out or _default_output(args.input), output_data)


if __name__ == "__main__":
    main()
