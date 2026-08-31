"""Prepend a slow minimum-jerk approach to the first strict-IK configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .rb3_kinematics import RB3730Kinematics
except ImportError:
    from rb3_kinematics import RB3730Kinematics


def _load(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as h5_file:
            return {key: h5_file[key][()] for key in h5_file}
    raise ValueError(f"unsupported trajectory format: {path.suffix}")


def _write(path: str | Path, data: dict[str, object]) -> Path:
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


def _smoothstep5(value: np.ndarray) -> np.ndarray:
    """Minimum-jerk blend with zero endpoint velocity and acceleration."""
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _repeat_first_then_append(array: np.ndarray, approach_frames: int) -> np.ndarray:
    array = np.asarray(array)
    prefix = np.repeat(array[0:1], approach_frames, axis=0)
    return np.concatenate((prefix, array[1:]), axis=0)


def _unwrap_to_nearest_branch(
    trajectory: np.ndarray,
    initial_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Choose equivalent 2-pi branches closest to the preceding configuration."""
    result = np.asarray(trajectory, dtype=float).copy()
    previous = np.asarray(initial_q, dtype=float)
    for frame in range(len(result)):
        for joint in range(result.shape[1]):
            candidates = result[frame, joint] + 2.0 * np.pi * np.arange(-3, 4)
            candidates = candidates[
                (candidates >= lower[joint] - 1.0e-12)
                & (candidates <= upper[joint] + 1.0e-12)
            ]
            if len(candidates):
                result[frame, joint] = candidates[
                    np.argmin(np.abs(candidates - previous[joint]))
                ]
        previous = result[frame]
    return result


def build_approach_reference(
    source: dict[str, np.ndarray],
    duration_s: float,
    extended_rb3_q: np.ndarray,
    open_revo2_q: np.ndarray,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> dict[str, object]:
    target_pos_source = np.asarray(source["target_wrist_pos"], dtype=float)
    target_quat_source = np.asarray(source["target_wrist_quat"], dtype=float)
    rb3_source = np.asarray(source["rb3_joints"], dtype=float)
    revo2_source = np.asarray(source["revo2_joints"], dtype=float)
    source_frames = len(target_pos_source)
    expected = {
        "target_wrist_pos": (source_frames, 3),
        "target_wrist_quat": (source_frames, 4),
        "rb3_joints": (source_frames, 6),
        "revo2_joints": (source_frames, 6),
    }
    actual = {
        "target_wrist_pos": target_pos_source.shape,
        "target_wrist_quat": target_quat_source.shape,
        "rb3_joints": rb3_source.shape,
        "revo2_joints": revo2_source.shape,
    }
    bad = {key: (actual[key], shape) for key, shape in expected.items() if actual[key] != shape}
    if source_frames == 0 or bad:
        raise ValueError(f"invalid source trajectory: frames={source_frames}, shapes={bad}")
    if not all(
        np.isfinite(array).all()
        for array in (target_pos_source, target_quat_source, rb3_source, revo2_source)
    ):
        raise ValueError("source target or Revo2 trajectory contains NaN/Inf")

    fps = float(np.asarray(source.get("fps", 30.0)).item())
    if not np.isfinite(fps) or fps <= 0 or duration_s <= 0:
        raise ValueError("fps and approach duration must be finite and > 0")
    approach_frames = int(round(duration_s * fps)) + 1
    if approach_frames < 2:
        raise ValueError("approach duration must produce at least two frames")

    kinematics = RB3730Kinematics()
    lower, upper = kinematics.get_joint_limits()
    extended_rb3_q = np.asarray(extended_rb3_q, dtype=float)
    open_revo2_q = np.asarray(open_revo2_q, dtype=float)
    if extended_rb3_q.shape != (6,) or not np.isfinite(extended_rb3_q).all():
        raise ValueError("extended RB3 configuration must be finite shape (6,)")
    if open_revo2_q.shape != (6,) or not np.isfinite(open_revo2_q).all():
        raise ValueError("open Revo2 configuration must be finite shape (6,)")
    if np.any(extended_rb3_q < lower) or np.any(extended_rb3_q > upper):
        raise ValueError("extended RB3 configuration is outside joint limits")

    # The source frame 0 is already a strict full-pose IK solution. Resolve
    # equivalent revolute-joint branches relative to the extended posture so
    # the approach never makes an unnecessary full turn.
    rb3_source = _unwrap_to_nearest_branch(
        rb3_source, extended_rb3_q, lower, upper
    )
    start_pos, start_quat = kinematics.forward(extended_rb3_q)
    unit_time = np.linspace(0.0, 1.0, approach_frames)
    blend = _smoothstep5(unit_time)
    rb3_approach = (
        extended_rb3_q[None] * (1.0 - blend[:, None])
        + rb3_source[0][None] * blend[:, None]
    )
    approach_pos, approach_quat = kinematics.forward_batch(rb3_approach)
    rb3_joints = np.concatenate((rb3_approach, rb3_source[1:]), axis=0)
    target_pos = np.concatenate((approach_pos, target_pos_source[1:]), axis=0)
    target_quat = np.concatenate((approach_quat, target_quat_source[1:]), axis=0)
    total_frames = len(target_pos)
    fk_pos, fk_quat = kinematics.forward_batch(rb3_joints)
    position_error = np.linalg.norm(fk_pos - target_pos, axis=1)
    orientation_error = (
        Rotation.from_quat(target_quat).inv() * Rotation.from_quat(fk_quat)
    ).magnitude()
    finite_solution = np.isfinite(rb3_joints).all(axis=1)
    joint_limit_violation = np.any(
        (rb3_joints < lower[None] - 1.0e-9)
        | (rb3_joints > upper[None] + 1.0e-9),
        axis=1,
    )
    max_limit_violation = np.maximum(
        np.maximum(lower[None] - rb3_joints, rb3_joints - upper[None]), 0.0
    ).max(axis=1)
    success = (
        finite_solution
        & ~joint_limit_violation
        & (position_error <= position_tolerance_m)
        & (orientation_error <= orientation_tolerance_rad)
    )
    optimizer_success = success.copy()
    solver_cost = np.zeros(total_frames)
    solver_nfev = np.zeros(total_frames, dtype=int)
    solver_message = np.full(
        total_frames,
        "minimum-jerk joint approach to strict-IK frame; FK validated",
        dtype=object,
    )
    warm_start_frame = np.arange(-1, total_frames - 1, dtype=int)

    revo2_approach = (
        open_revo2_q[None] * (1.0 - blend[:, None])
        + revo2_source[0][None] * blend[:, None]
    )
    revo2_joints = np.concatenate((revo2_approach, revo2_source[1:]), axis=0)
    reference_joints = np.concatenate((rb3_joints, revo2_joints), axis=1)
    step_norm = np.concatenate(
        ((0.0,), np.linalg.norm(np.diff(reference_joints, axis=0), axis=1))
    )
    failed = np.flatnonzero(~success)

    output: dict[str, object] = {}
    time_series_replacements = {
        "rb3_joints",
        "revo2_joints",
        "reference_joints",
        "rb3_joint_step_norm_rad",
        "wrist_pos",
        "wrist_quat",
        "target_wrist_pos",
        "target_wrist_quat",
        "fk_wrist_pos",
        "fk_wrist_quat",
        "ik_success",
        "optimizer_success",
        "failed_frame_indices",
        "position_error_m",
        "orientation_error_rad",
        "joint_limit_violation",
        "max_joint_limit_violation_rad",
        "finite_solution",
        "solver_cost",
        "solver_nfev",
        "solver_message",
        "warm_start_frame",
        "frame_index",
        "object_pos",
        "object_quat",
        "mano_joint_world",
    }
    for key, value in source.items():
        if key not in time_series_replacements:
            output[key] = value

    output.update(
        {
            "fps": fps,
            "rb3_joints": rb3_joints,
            "revo2_joints": revo2_joints,
            "reference_joints": reference_joints,
            "rb3_joint_step_norm_rad": step_norm,
            "wrist_pos": target_pos,
            "wrist_quat": target_quat,
            "target_wrist_pos": target_pos,
            "target_wrist_quat": target_quat,
            "fk_wrist_pos": fk_pos,
            "fk_wrist_quat": fk_quat,
            "ik_success": success,
            "optimizer_success": optimizer_success,
            "failed_frame_indices": failed,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "joint_limit_violation": joint_limit_violation,
            "max_joint_limit_violation_rad": max_limit_violation,
            "finite_solution": finite_solution,
            "solver_cost": solver_cost,
            "solver_nfev": solver_nfev,
            "solver_message": np.asarray(solver_message),
            "warm_start_frame": warm_start_frame,
            "frame_index": np.arange(total_frames),
            "source_frame_index": np.concatenate(
                (
                    np.full(approach_frames - 1, -1, dtype=int),
                    np.asarray(source.get("frame_index", np.arange(source_frames))),
                )
            ),
            "approach_phase": np.arange(total_frames) < approach_frames,
            "approach_frame_count": approach_frames,
            "approach_duration_s": duration_s,
            "approach_interpolation": "minimum_jerk_joint_space_to_strict_ik_frame0",
            "approach_start_rb3_joints": extended_rb3_q,
            "approach_start_revo2_joints": open_revo2_q,
            "approach_start_wrist_pos": start_pos,
            "approach_start_wrist_quat_xyzw": start_quat,
            "source_reference_frame_count": source_frames,
        }
    )
    for key in ("object_pos", "object_quat", "mano_joint_world"):
        if key in source:
            output[key] = _repeat_first_then_append(source[key], approach_frames)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Strict RB3+Revo2 reference .h5/.npz")
    parser.add_argument("--out", required=True, help="Output .h5/.npz")
    parser.add_argument("--duration", type=float, default=5.0, help="Approach seconds")
    parser.add_argument(
        "--extended-rb3-q", nargs=6, type=float, default=np.zeros(6), metavar="RAD"
    )
    parser.add_argument(
        "--open-revo2-q", nargs=6, type=float, default=np.zeros(6), metavar="RAD"
    )
    parser.add_argument("--position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--orientation-tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()

    source = _load(args.input)
    result = build_approach_reference(
        source,
        args.duration,
        np.asarray(args.extended_rb3_q, dtype=float),
        np.asarray(args.open_revo2_q, dtype=float),
        args.position_tolerance,
        args.orientation_tolerance,
    )
    output = _write(args.out, result)
    success = np.asarray(result["ik_success"], dtype=bool)
    pos_error = np.asarray(result["position_error_m"], dtype=float)
    ori_error = np.asarray(result["orientation_error_rad"], dtype=float)
    reference = np.asarray(result["reference_joints"], dtype=float)
    fps = float(result["fps"])
    velocity = np.diff(reference, axis=0) / (1.0 / fps)
    acceleration = np.diff(velocity, axis=0) / (1.0 / fps)
    approach_frames = int(result["approach_frame_count"])
    approach_steps = np.linalg.norm(
        np.diff(reference[:approach_frames], axis=0), axis=1
    )
    approach_velocity = velocity[: approach_frames - 1]
    approach_acceleration = acceleration[: approach_frames - 2]
    print("[Extended-pose minimum-jerk approach to strict IK]")
    print(f"  output:                    {output}")
    print(f"  approach duration/frames:  {args.duration:.3f} s / {result['approach_frame_count']}")
    print(f"  source / total frames:     {result['source_reference_frame_count']} / {len(reference)}")
    print(f"  IK success:                {success.sum()}/{len(success)} ({100.0 * success.mean():.2f}%)")
    print(f"  failed frames:             {np.flatnonzero(~success).tolist()}")
    print(f"  max position error:        {np.nanmax(pos_error):.9g} m")
    print(f"  max orientation error:     {np.nanmax(ori_error):.9g} rad")
    print(f"  max 12-DoF joint step:     {np.nanmax(np.linalg.norm(np.diff(reference, axis=0), axis=1)):.9g} rad")
    print(f"  max abs joint velocity:    {np.nanmax(np.abs(velocity)):.9g} rad/s")
    print(f"  max abs joint acceleration:{np.nanmax(np.abs(acceleration)):.9g} rad/s^2")
    print(f"  approach max joint step:   {np.nanmax(approach_steps):.9g} rad")
    print(f"  approach max abs velocity: {np.nanmax(np.abs(approach_velocity)):.9g} rad/s")
    print(f"  approach max abs accel:    {np.nanmax(np.abs(approach_acceleration)):.9g} rad/s^2")


if __name__ == "__main__":
    main()
