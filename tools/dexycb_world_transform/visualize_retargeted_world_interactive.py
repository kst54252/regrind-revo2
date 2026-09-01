#!/usr/bin/env python3
"""Visualize world-frame MANO, Revo2 retargeting, and tuna keypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


MANO21_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
REVO2_CHAINS = (
    (0, 1, 2, 3, 17),
    (0, 4, 5, 6, 18),
    (0, 10, 11, 12, 19),
    (0, 7, 8, 9, 20),
    (0, 13, 14, 15, 16),
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        return {key: h5_file[key][()] for key in h5_file}


def _hand_trace(points, chains, name, color, labels, visible=True):
    if not visible or not np.isfinite(points).all():
        return go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", name=name)
    x, y, z, text = [], [], [], []
    for chain in chains:
        for index in chain:
            x.append(points[index, 0])
            y.append(points[index, 1])
            z.append(points[index, 2])
            text.append(labels[index])
        x.append(None); y.append(None); z.append(None); text.append("")
    return go.Scatter3d(
        x=x, y=y, z=z, text=text,
        hovertemplate=(
            "%{text}<br>World X=%{x:.4f}<br>World Y=%{y:.4f}"
            "<br>World Z=%{z:.4f}<extra></extra>"
        ),
        mode="lines+markers", name=name,
        line={"color": color, "width": 6},
        marker={"color": color, "size": 4},
    )


def _object_trace(points):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        text=[f"tuna_{index:02d}" for index in range(len(points))],
        hovertemplate=(
            "%{text}<br>World X=%{x:.4f}<br>World Y=%{y:.4f}"
            "<br>World Z=%{z:.4f}<extra></extra>"
        ),
        mode="markers", name="Tuna can (50 points)",
        marker={"color": "#f9a825", "size": 4, "opacity": 0.9},
    )


def _path(points, name, color, dash):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="lines", name=name, hoverinfo="skip", opacity=0.55,
        line={"color": color, "width": 4, "dash": dash},
    )


def make_html(retargeted_path: Path, world_path: Path, output_path: Path) -> dict:
    retargeted = _load(retargeted_path)
    world = _load(world_path)
    human = np.asarray(world["mano_joint_world_mano21"], dtype=float)
    robot_camera = np.asarray(retargeted["robot_keypoints"], dtype=float)
    transform = np.asarray(world["T_world_camera"], dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    robot = robot_camera @ rotation.T + translation[None, None, :]
    object_pos = np.asarray(world["object_pos_world"], dtype=float)
    object_quat = np.asarray(world["object_quat_world"], dtype=float)
    object_local = np.asarray(world["object_points_local"], dtype=float)
    wrist = np.asarray(world["wrist_pos_world"], dtype=float)
    success = np.asarray(retargeted["solver_success"], dtype=bool)
    objective = np.asarray(retargeted["objective_value"], dtype=float)
    source_indices = np.asarray(world["source_frame_indices"], dtype=int)
    frames_count = len(human)
    if not (
        human.shape == robot.shape == (frames_count, 21, 3)
        and object_pos.shape == wrist.shape == (frames_count, 3)
        and object_quat.shape == (frames_count, 4)
        and success.shape == objective.shape == (frames_count,)
    ):
        raise ValueError(
            f"frame/shape mismatch: human={human.shape}, robot={robot.shape}, "
            f"object={object_pos.shape}, wrist={wrist.shape}, success={success.shape}"
        )
    object_rotations = Rotation.from_quat(object_quat[:, [1, 2, 3, 0]]).as_matrix()
    object_world = (
        np.einsum("tij,kj->tki", object_rotations, object_local)
        + object_pos[:, None, :]
    )
    finite = np.isfinite(human).all() and np.isfinite(object_world).all()
    finite = finite and np.isfinite(robot[success]).all()
    if not finite:
        raise ValueError("visualization input contains unexpected NaN/Inf")

    robot_names = [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in np.asarray(retargeted["robot_keypoint_names"]).reshape(-1)
    ]
    human_names = [f"MANO_{index:02d}" for index in range(21)]
    static = [
        _path(wrist, "Wrist path", "#8e0000", "dash"),
        _path(object_pos, "Object center path", "#ef6c00", "dot"),
    ]

    def traces(frame):
        return [
            _hand_trace(human[frame], MANO21_CHAINS, "Original MANO21", "#1976d2", human_names),
            _hand_trace(robot[frame], REVO2_CHAINS, "Retargeted Revo2", "#d32f2f", robot_names, success[frame]),
            _object_trace(object_world[frame]),
            *static,
        ]

    def title(frame):
        status = "SUCCESS" if success[frame] else "FAILURE"
        cost = f"{objective[frame]:.6g}" if np.isfinite(objective[frame]) else "n/a"
        return (
            f"20200709_143747_left | world-frame retargeting | frame {frame}/{frames_count - 1} "
            f"(source {source_indices[frame]})<br>"
            f"<sup>{status} | objective={cost} | +Z is up</sup>"
        )

    points = np.concatenate((human.reshape(-1, 3), robot[success].reshape(-1, 3), object_world.reshape(-1, 3)))
    points = points[np.isfinite(points).all(axis=1)]
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.58, 0.12)
    fps = float(np.asarray(world.get("fps", 30.0)).item())
    duration = 1000.0 / fps
    frames = [go.Frame(data=traces(i), name=str(i), layout={"title": title(i)}) for i in range(frames_count)]
    figure = go.Figure(data=traces(0), frames=frames)
    figure.update_layout(
        title=title(0), template="plotly_white", height=880,
        margin={"l": 0, "r": 0, "b": 0, "t": 88},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.75)"},
        scene={
            "xaxis": {"title": "World X [m]", "range": [center[0]-radius, center[0]+radius]},
            "yaxis": {"title": "World Y [m]", "range": [center[1]-radius, center[1]+radius]},
            "zaxis": {"title": "World Z [m]", "range": [min(0.0, center[2]-radius), center[2]+radius]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": -1.6, "z": 1.15}},
        },
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.02, "y": 0.04,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": duration, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "Frame: "}, "pad": {"t": 44},
            "steps": [
                {"args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(i), "method": "animate"}
                for i in range(frames_count)
            ],
        }],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return {
        "frames": frames_count,
        "success": int(success.sum()),
        "object_start": object_pos[0],
        "object_end": object_pos[-1],
        "object_delta": object_pos[-1] - object_pos[0],
        "output": output_path.resolve(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retargeted", type=Path)
    parser.add_argument("world", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = make_html(
        args.retargeted.expanduser().resolve(),
        args.world.expanduser().resolve(),
        args.out.expanduser().resolve(),
    )
    print(f"Saved world retargeting HTML: {summary['output']}")
    print(f"  solver: {summary['success']}/{summary['frames']}")
    print(f"  object start: {summary['object_start']}")
    print(f"  object end:   {summary['object_end']}")
    print(f"  object delta: {summary['object_delta']}")


if __name__ == "__main__":
    main()
