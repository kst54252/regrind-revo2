"""Diagnose RB3 full-pose IK failures with workspace and position-only IK."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    from .rb3_kinematics import RB3730Kinematics
except ImportError:
    from rb3_kinematics import RB3730Kinematics


def _load(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    with h5py.File(path, "r") as h5_file:
        return {key: h5_file[key][()] for key in h5_file}


def _first(data: dict, names: tuple[str, ...], description: str) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"input has no {description}; tried {names}")


def _failure_segments(success: np.ndarray) -> list[tuple[int, int]]:
    failed = np.flatnonzero(~success)
    if not len(failed):
        return []
    breaks = np.flatnonzero(np.diff(failed) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(failed) - 1]))
    return [(int(failed[start]), int(failed[end])) for start, end in zip(starts, ends)]


def _joint_margin(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(q - lower[None], upper[None] - q).min(axis=1)


def _equal_axes(ax, points: np.ndarray) -> None:
    finite = points[np.isfinite(points).all(axis=1)]
    lower, upper = finite.min(axis=0), finite.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) * 0.53, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _draw_workspace_scene(
    ax,
    workspace_points: np.ndarray,
    target: np.ndarray,
    success: np.ndarray,
    object_initial: np.ndarray,
    segments: list[tuple[int, int]],
    title: str,
) -> None:
    ax.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        workspace_points[:, 2],
        s=0.3,
        c="#999999",
        alpha=0.045,
        rasterized=True,
        label="Sampled RB3 workspace",
    )
    ax.plot(
        target[:, 0], target[:, 1], target[:, 2], color="#377eb8", linewidth=1.3,
        alpha=0.75, label="Target wrist trajectory",
    )
    ax.scatter(
        *target[success].T, s=25, c="#2ca02c", edgecolors="white", linewidths=0.3,
        label="Full-pose IK success",
    )
    ax.scatter(
        *target[~success].T, s=38, c="#d62728", marker="x", linewidths=1.5,
        label="Full-pose IK failure",
    )
    ax.scatter(
        *object_initial, s=180, c="#ffbf00", marker="*", edgecolors="#805500",
        linewidths=0.8, label="Initial object position",
    )
    for segment_index, (start, end) in enumerate(segments):
        points = target[start : end + 1]
        ax.plot(
            points[:, 0], points[:, 1], points[:, 2], color="#ff7f0e", linewidth=3.0,
            label="Consecutive failure segment" if segment_index == 0 else None,
        )
        ax.text(*points[0], f" {start}", color="#a83f00", fontsize=8)
        ax.text(*points[-1], f" {end}", color="#a83f00", fontsize=8)
    visible = np.concatenate((workspace_points, target, object_initial[None]), axis=0)
    _equal_axes(ax, visible)
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.set_zlabel("World Z [m]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)


def _save_diagnostic_h5(path: Path, data: dict) -> None:
    with h5py.File(path, "w") as h5_file:
        for key, value in data.items():
            array = np.asarray(value)
            if array.dtype.kind in ("U", "O"):
                h5_file.create_dataset(
                    key, data=array.astype(object), dtype=h5py.string_dtype("utf-8")
                )
            else:
                compression = "gzip" if array.ndim > 0 and array.size > 1000 else None
                h5_file.create_dataset(key, data=array, compression=compression)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_trajectory", help="World-frame trajectory HDF5/NPZ")
    parser.add_argument("full_pose_result", help="Existing full-pose IK HDF5/NPZ")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workspace-samples", type=int, default=100_000)
    parser.add_argument("--plot-workspace-points", type=int, default=45_000)
    parser.add_argument("--nearest-seeds", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=730)
    parser.add_argument("--position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=1000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.workspace_samples < 1000:
        raise ValueError("--workspace-samples must be at least 1000")
    if not 1 <= args.nearest_seeds <= args.workspace_samples:
        raise ValueError("invalid --nearest-seeds")
    if args.position_tolerance <= 0 or args.joint_limit_margin < 0:
        raise ValueError("invalid position tolerance or joint-limit margin")

    world = _load(args.world_trajectory)
    full = _load(args.full_pose_result)
    target = _first(
        world, ("wrist_pos_world", "wrist_pos", "robot_pos"), "world wrist position"
    ).astype(float)
    object_pos = _first(
        world, ("object_pos_world", "object_pos"), "world object position"
    ).astype(float)
    full_q = np.asarray(full["rb3_joints"], dtype=float)
    full_success = np.asarray(full["ik_success"], dtype=bool)
    full_position_error = np.asarray(full["position_error_m"], dtype=float)
    full_orientation_error = np.asarray(full["orientation_error_rad"], dtype=float)
    T = len(target)
    expected = {
        "target": (T, 3),
        "object_pos": (T, 3),
        "rb3_joints": (T, 6),
        "ik_success": (T,),
        "position_error": (T,),
        "orientation_error": (T,),
    }
    actual = {
        "target": target.shape,
        "object_pos": object_pos.shape,
        "rb3_joints": full_q.shape,
        "ik_success": full_success.shape,
        "position_error": full_position_error.shape,
        "orientation_error": full_orientation_error.shape,
    }
    bad = {key: (actual[key], shape) for key, shape in expected.items() if actual[key] != shape}
    if T == 0 or bad:
        raise ValueError(f"invalid trajectory/result shapes: {bad}")
    if not all(
        np.isfinite(value).all()
        for value in (target, object_pos, full_q, full_position_error, full_orientation_error)
    ):
        raise ValueError("input trajectory or IK result contains NaN/Inf")

    base_position = np.asarray(full.get("rb3_base_position", np.zeros(3)), dtype=float)
    base_quaternion = np.asarray(
        full.get("rb3_base_quat_xyzw", (0.0, 0.0, 0.0, 1.0)), dtype=float
    )
    robot = RB3730Kinematics(
        base_position=base_position, base_quaternion_xyzw=base_quaternion
    )
    lower, upper = robot.get_joint_limits()
    full_margin = _joint_margin(full_q, lower, upper)
    base_distance = np.linalg.norm(target - base_position[None], axis=1)

    rng = np.random.default_rng(args.random_seed)
    workspace_q = rng.uniform(lower, upper, size=(args.workspace_samples, 6))
    workspace_points, _ = robot.forward_batch(workspace_q)
    if not np.isfinite(workspace_points).all():
        raise RuntimeError("sampled workspace contains NaN/Inf")

    position_q = full_q.copy()
    position_fk = np.full((T, 3), np.nan)
    position_error = full_position_error.copy()
    position_success = full_success.copy()
    position_optimizer_success = full_success.copy()
    position_margin = full_margin.copy()
    nearest_workspace_distance = np.full(T, np.nan)
    classification = np.full(T, "full-pose-success", dtype=object)
    failed_indices = np.flatnonzero(~full_success)
    for frame in failed_indices:
        distances_squared = np.sum((workspace_points - target[frame]) ** 2, axis=1)
        seed_indices = np.argpartition(distances_squared, args.nearest_seeds - 1)[
            : args.nearest_seeds
        ]
        seed_indices = seed_indices[np.argsort(distances_squared[seed_indices])]
        nearest_workspace_distance[frame] = float(
            np.sqrt(distances_squared[seed_indices[0]])
        )
        result = robot.inverse_position(
            target[frame],
            initial_q=full_q[frame],
            neutral_q=np.zeros(6),
            additional_seeds=workspace_q[seed_indices],
            position_tolerance_m=args.position_tolerance,
            max_nfev=args.max_nfev,
        )
        position_q[frame] = result.q
        position_fk[frame] = result.fk_position
        position_error[frame] = result.position_error_m
        position_success[frame] = result.success
        position_optimizer_success[frame] = result.optimizer_success
        position_margin[frame] = result.min_joint_limit_margin_rad
        classification[frame] = (
            "orientation-related" if result.success else "position/workspace-related"
        )

    position_fk[full_success] = robot.forward_batch(full_q[full_success])[0]
    combined_margin = np.minimum(full_margin, position_margin)
    joint_limit_near = combined_margin <= args.joint_limit_margin
    orientation_related = np.flatnonzero(
        (~full_success) & position_success
    )
    position_related = np.flatnonzero(
        (~full_success) & (~position_success)
    )
    joint_limit_near_indices = np.flatnonzero(joint_limit_near)
    segments = _failure_segments(full_success)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "failed_frames.csv"
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = (
            "frame", "target_x", "target_y", "target_z", "base_distance_m",
            "full_position_error_m", "full_orientation_error_rad",
            "full_joint_limit_margin_rad", "position_only_success",
            "position_only_error_m", "position_only_joint_limit_margin_rad",
            "nearest_workspace_sample_m", "classification", "joint_limit_near",
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in failed_indices:
            writer.writerow(
                {
                    "frame": int(frame),
                    "target_x": target[frame, 0],
                    "target_y": target[frame, 1],
                    "target_z": target[frame, 2],
                    "base_distance_m": base_distance[frame],
                    "full_position_error_m": full_position_error[frame],
                    "full_orientation_error_rad": full_orientation_error[frame],
                    "full_joint_limit_margin_rad": full_margin[frame],
                    "position_only_success": bool(position_success[frame]),
                    "position_only_error_m": position_error[frame],
                    "position_only_joint_limit_margin_rad": position_margin[frame],
                    "nearest_workspace_sample_m": nearest_workspace_distance[frame],
                    "classification": classification[frame],
                    "joint_limit_near": bool(joint_limit_near[frame]),
                }
            )

    diagnostic_path = out_dir / "rb3_ik_diagnostics.h5"
    _save_diagnostic_h5(
        diagnostic_path,
        {
            "workspace_q": workspace_q,
            "workspace_points": workspace_points,
            "target_wrist_pos": target,
            "object_initial_pos": object_pos[0],
            "full_pose_success": full_success,
            "full_pose_position_error_m": full_position_error,
            "full_pose_orientation_error_rad": full_orientation_error,
            "full_pose_joint_limit_margin_rad": full_margin,
            "position_only_q": position_q,
            "position_only_fk_position": position_fk,
            "position_only_success": position_success,
            "position_only_optimizer_success": position_optimizer_success,
            "position_only_error_m": position_error,
            "position_only_joint_limit_margin_rad": position_margin,
            "nearest_workspace_sample_m": nearest_workspace_distance,
            "base_distance_m": base_distance,
            "classification": classification,
            "joint_limit_near": joint_limit_near,
            "orientation_related_frames": orientation_related,
            "position_workspace_related_frames": position_related,
            "joint_limit_near_frames": joint_limit_near_indices,
            "failure_segments": np.asarray(segments, dtype=int).reshape(-1, 2),
            "rb3_joint_lower_rad": lower,
            "rb3_joint_upper_rad": upper,
            "position_tolerance_m": args.position_tolerance,
            "joint_limit_margin_threshold_rad": args.joint_limit_margin,
            "random_seed": args.random_seed,
            "source_world_trajectory": str(Path(args.world_trajectory).resolve()),
            "source_full_pose_result": str(Path(args.full_pose_result).resolve()),
        },
    )

    plot_count = min(args.plot_workspace_points, len(workspace_points))
    plot_indices = rng.choice(len(workspace_points), size=plot_count, replace=False)
    plot_workspace = workspace_points[plot_indices]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    _draw_workspace_scene(
        ax, plot_workspace, target, full_success, object_pos[0], segments,
        "RB3 workspace and world-frame wrist targets",
    )
    fig.tight_layout()
    main_plot_path = out_dir / "workspace_trajectory_diagnostic.png"
    fig.savefig(main_plot_path, dpi=180, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    segment_paths = []
    for start, end in segments:
        segment_target = target[start : end + 1]
        focus_points = np.concatenate((segment_target, object_pos[0][None]), axis=0)
        focus_lower = focus_points.min(axis=0) - 0.10
        focus_upper = focus_points.max(axis=0) + 0.10
        local_mask = np.all(
            (plot_workspace >= focus_lower[None])
            & (plot_workspace <= focus_upper[None]),
            axis=1,
        )
        local_workspace = plot_workspace[local_mask]
        if len(local_workspace) < 500:
            local_workspace = plot_workspace
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            *local_workspace.T, s=0.5, c="#999999", alpha=0.07,
            rasterized=True, label="Local sampled RB3 workspace",
        )
        ax.plot(
            target[:, 0], target[:, 1], target[:, 2], color="#377eb8",
            alpha=0.18, linewidth=0.8, label="Full target trajectory",
        )
        ax.plot(
            *segment_target.T, color="#ff7f0e", linewidth=3.2,
            label=f"Failure segment {start}-{end}",
        )
        ax.scatter(
            *segment_target.T, s=42, c="#d62728", marker="x", linewidths=1.5,
            label="Full-pose IK failure",
        )
        ax.scatter(
            *object_pos[0], s=180, c="#ffbf00", marker="*", edgecolors="#805500",
            linewidths=0.8, label="Initial object position",
        )
        ax.text(*segment_target[0], f" {start}", color="#a83f00", fontsize=9)
        ax.text(*segment_target[-1], f" {end}", color="#a83f00", fontsize=9)
        _equal_axes(
            ax,
            np.concatenate((local_workspace, segment_target, object_pos[0][None]), axis=0),
        )
        ax.set_xlabel("World X [m]")
        ax.set_ylabel("World Y [m]")
        ax.set_zlabel("World Z [m]")
        ax.set_title(f"Consecutive full-pose IK failure segment: frames {start}-{end}")
        ax.legend(loc="upper left", fontsize=8)
        segment_path = out_dir / f"failure_segment_{start:04d}_{end:04d}.png"
        fig.tight_layout()
        fig.savefig(segment_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        segment_paths.append(segment_path)

    print("[RB3 IK failure diagnosis]")
    print(f"  frames:                         {T}")
    print(f"  workspace samples:              {len(workspace_points)}")
    print(f"  wrist XYZ min:                  {target.min(axis=0)}")
    print(f"  wrist XYZ max:                  {target.max(axis=0)}")
    print(f"  full-pose IK success rate:      {100.0 * full_success.mean():.2f}%")
    print(f"  position-only IK success rate:  {100.0 * position_success.mean():.2f}%")
    print(f"  orientation-related frames:     {orientation_related.tolist()}")
    print(f"  position/workspace frames:      {position_related.tolist()}")
    print(f"  joint-limit-near frames:        {joint_limit_near_indices.tolist()}")
    print(f"  consecutive failure segments:   {segments}")
    print("\n[failed frame details]")
    for frame in failed_indices:
        print(
            f"  frame {frame:04d} target={np.array2string(target[frame], precision=6)} "
            f"base_dist={base_distance[frame]:.6f} m "
            f"full_pos={full_position_error[frame]:.6f} m "
            f"full_ori={full_orientation_error[frame]:.6f} rad "
            f"full_margin={full_margin[frame]:.6f} rad "
            f"pos_only={'OK' if position_success[frame] else 'FAIL'} "
            f"pos_err={position_error[frame]:.6f} m "
            f"pos_margin={position_margin[frame]:.6f} rad "
            f"class={classification[frame]}"
        )
    print(f"\nSaved diagnostic data: {diagnostic_path}")
    print(f"Saved failure CSV:      {report_path}")
    print(f"Saved workspace plot:   {main_plot_path}")
    for path in segment_paths:
        print(f"Saved segment plot:     {path}")


if __name__ == "__main__":
    main()
