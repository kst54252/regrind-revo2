"""Animate MANO, retargeted Revo2, and object keypoints in one 3D view.

The input is the HDF5/NPZ file produced by ``retarget_hand_object.py``.  Failed
retargeting frames remain visible as MANO/object observations while the Revo2
points are omitted and the frame title is marked ``FAILURE``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation


FINGER_CHAINS = (
    (0, 1, 2, 3, 17),    # index
    (0, 4, 5, 6, 18),    # middle
    (0, 10, 11, 12, 19), # ring
    (0, 7, 8, 9, 20),    # little
    (0, 13, 14, 15, 16), # thumb
)


def _load(path: str) -> dict[str, np.ndarray]:
    if path.lower().endswith(".npz"):
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}
    with h5py.File(path, "r") as h5_file:
        return {name: h5_file[name][()] for name in h5_file}


def _require_shape(name: str, value: np.ndarray, trailing_shape: tuple[int, ...]):
    if value.ndim != len(trailing_shape) + 1 or value.shape[1:] != trailing_shape:
        raise ValueError(
            f"{name} must have shape (T, {', '.join(map(str, trailing_shape))}), "
            f"got {value.shape}"
        )


def _object_points_world(
    local_points: np.ndarray, positions: np.ndarray, quaternions_xyzw: np.ndarray
) -> np.ndarray:
    result = np.empty((len(positions), len(local_points), 3), dtype=float)
    for frame in range(len(positions)):
        result[frame] = (
            Rotation.from_quat(quaternions_xyzw[frame]).apply(local_points)
            + positions[frame]
        )
    return result


def _set_scatter(scatter, points: np.ndarray):
    if points.size == 0 or not np.isfinite(points).all():
        scatter._offsets3d = ([], [], [])
    else:
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])


def _set_chain_lines(lines, points: np.ndarray):
    valid = points.shape == (21, 3) and np.isfinite(points).all()
    for line, chain in zip(lines, FINGER_CHAINS):
        if valid:
            segment = points[np.asarray(chain)]
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
        else:
            line.set_data_3d([], [], [])


def main():
    parser = argparse.ArgumentParser(
        description="Animate MANO, Revo2, and object keypoints from a retargeted sequence."
    )
    parser.add_argument("result", help="Retargeted .h5/.hdf5/.npz result.")
    parser.add_argument(
        "--object-keypoints",
        help="Optional (N,3) .npy override for older results without object_points_local.",
    )
    parser.add_argument("--fps", type=float, help="Playback FPS override.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, help="Exclusive end frame.")
    parser.add_argument("--save", help="Optional .gif or .mp4 animation path.")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    data = _load(args.result)
    human = np.asarray(data["mano_joint_coords"], dtype=float)
    robot = np.asarray(data["robot_keypoints"], dtype=float)
    object_pos = np.asarray(data["object_pos"], dtype=float)
    object_quat = np.asarray(data["object_quat"], dtype=float)
    _require_shape("mano_joint_coords", human, (21, 3))
    _require_shape("robot_keypoints", robot, (21, 3))
    _require_shape("object_pos", object_pos, (3,))
    _require_shape("object_quat", object_quat, (4,))
    if len({len(human), len(robot), len(object_pos), len(object_quat)}) != 1:
        raise ValueError("Trajectory arrays have different frame counts")

    if args.object_keypoints:
        object_local = np.asarray(np.load(args.object_keypoints), dtype=float)
    elif "object_points_local" in data:
        object_local = np.asarray(data["object_points_local"], dtype=float)
    else:
        raise KeyError(
            "Result has no object_points_local; pass --object-keypoints points.npy"
        )
    if object_local.ndim != 2 or object_local.shape[1] != 3:
        raise ValueError(f"object keypoints must have shape (N,3), got {object_local.shape}")
    object_world = _object_points_world(object_local, object_pos, object_quat)

    T = len(human)
    start = args.start
    end = T if args.end is None else args.end
    if not 0 <= start < end <= T:
        raise ValueError(f"Require 0 <= start < end <= {T}, got {start}, {end}")
    frame_indices = np.arange(start, end)
    success = np.asarray(data.get("solver_success", np.isfinite(robot).all((1, 2))), dtype=bool)
    objective = np.asarray(data.get("objective_value", np.full(T, np.nan)), dtype=float)
    fps = float(args.fps if args.fps is not None else np.asarray(data.get("fps", 30.0)))
    if fps <= 0:
        raise ValueError("FPS must be > 0")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    human_scatter = ax.scatter([], [], [], s=25, c="#1f77b4", label="Human MANO")
    robot_scatter = ax.scatter([], [], [], s=25, c="#d62728", label="Revo2")
    object_scatter = ax.scatter([], [], [], s=13, c="#ffbf00", alpha=0.8, label="Object")
    human_lines = [ax.plot([], [], [], color="#1f77b4", linestyle="--", alpha=0.75)[0]
                   for _ in FINGER_CHAINS]
    robot_lines = [ax.plot([], [], [], color="#d62728", linewidth=1.8)[0]
                   for _ in FINGER_CHAINS]
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper right")
    ax.set_box_aspect((1, 1, 1))

    visible = np.concatenate(
        (human[frame_indices].reshape(-1, 3), object_world[frame_indices].reshape(-1, 3),
         robot[frame_indices].reshape(-1, 3)),
        axis=0,
    )
    visible = visible[np.isfinite(visible).all(axis=1)]
    if not len(visible):
        raise ValueError("No finite points in selected frame range")
    lower = visible.min(axis=0)
    upper = visible.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) * 0.55, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    def update(animation_frame):
        frame = int(frame_indices[animation_frame])
        _set_scatter(human_scatter, human[frame])
        _set_scatter(object_scatter, object_world[frame])
        _set_chain_lines(human_lines, human[frame])
        if success[frame]:
            _set_scatter(robot_scatter, robot[frame])
            _set_chain_lines(robot_lines, robot[frame])
            status = "success"
        else:
            _set_scatter(robot_scatter, np.empty((0, 3)))
            _set_chain_lines(robot_lines, np.empty((0, 3)))
            status = "FAILURE"
        objective_text = (
            f"{objective[frame]:.6g}" if np.isfinite(objective[frame]) else "n/a"
        )
        ax.set_title(
            f"Frame {frame}/{T - 1} | {status} | objective={objective_text}"
        )
        return (
            human_scatter,
            robot_scatter,
            object_scatter,
            *human_lines,
            *robot_lines,
        )

    update(0)
    fig.tight_layout()

    animation = None
    if args.save or not args.no_show:
        animation = FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=1000.0 / fps,
            blit=False,
            repeat=True,
        )
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = "pillow" if save_path.suffix.lower() == ".gif" else "ffmpeg"
        animation.save(save_path, writer=writer, fps=fps)
        print(f"Saved animation to {save_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
