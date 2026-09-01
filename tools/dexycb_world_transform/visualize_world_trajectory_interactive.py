#!/usr/bin/env python3
"""Create a standalone interactive HTML for a transformed DexYCB trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


# Standard sequential MANO21 topology: wrist, then four joints per finger.
MANO21_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


def _load(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    with h5py.File(path, "r") as h5_file:
        return {key: h5_file[key][()] for key in h5_file}


def _mano_trace(points: np.ndarray) -> go.Scatter3d:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    labels: list[str] = []
    for chain in MANO21_CHAINS:
        for index in chain:
            x.append(float(points[index, 0]))
            y.append(float(points[index, 1]))
            z.append(float(points[index, 2]))
            labels.append(f"MANO_{index:02d}")
        x.append(None)
        y.append(None)
        z.append(None)
        labels.append("")
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        text=labels,
        hovertemplate=(
            "%{text}<br>World X=%{x:.4f} m<br>World Y=%{y:.4f} m"
            "<br>World Z=%{z:.4f} m<extra></extra>"
        ),
        mode="lines+markers",
        name="Original MANO21 skeleton",
        line={"color": "#1976d2", "width": 7},
        marker={"color": "#42a5f5", "size": 4},
    )


def _object_trace(points: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        text=[f"tuna_point_{index:02d}" for index in range(len(points))],
        hovertemplate=(
            "%{text}<br>World X=%{x:.4f} m<br>World Y=%{y:.4f} m"
            "<br>World Z=%{z:.4f} m<extra></extra>"
        ),
        mode="markers",
        name="Tuna can (50 keypoints)",
        marker={"color": "#f9a825", "size": 4, "opacity": 0.9},
    )


def _wrist_trace(point: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        text=["wrist pose origin"],
        hovertemplate=(
            "%{text}<br>World X=%{x:.4f} m<br>World Y=%{y:.4f} m"
            "<br>World Z=%{z:.4f} m<extra></extra>"
        ),
        mode="markers",
        name="Wrist",
        marker={"color": "#d32f2f", "size": 8, "symbol": "diamond"},
    )


def _path_trace(points: np.ndarray, name: str, color: str, dash: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        name=name,
        hoverinfo="skip",
        line={"color": color, "width": 4, "dash": dash},
        opacity=0.62,
    )


def _world_axes(origin: np.ndarray, length: float) -> list[go.Scatter3d]:
    result = []
    for axis, color, index in (("+X", "#d32f2f", 0), ("+Y", "#388e3c", 1), ("+Z", "#1976d2", 2)):
        end = origin.copy()
        end[index] += length
        result.append(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                text=["World origin", axis],
                mode="lines+text",
                textposition="top center",
                name=f"World {axis}",
                hoverinfo="text",
                line={"color": color, "width": 6},
                showlegend=False,
            )
        )
    return result


def make_html(trajectory_path: Path, output_path: Path, fps_override: float | None) -> dict:
    data = _load(trajectory_path)
    required = (
        "mano_joint_world_mano21",
        "object_pos_world",
        "object_quat_world",
        "object_points_local",
        "wrist_pos_world",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"trajectory is missing required fields: {missing}")

    mano = np.asarray(data["mano_joint_world_mano21"], dtype=float)
    object_pos = np.asarray(data["object_pos_world"], dtype=float)
    object_quat = np.asarray(data["object_quat_world"], dtype=float)
    object_local = np.asarray(data["object_points_local"], dtype=float)
    wrist = np.asarray(data["wrist_pos_world"], dtype=float)
    frames_count = len(object_pos)
    expected_shapes = {
        "mano_joint_world_mano21": ((frames_count, 21, 3), mano.shape),
        "object_quat_world": ((frames_count, 4), object_quat.shape),
        "wrist_pos_world": ((frames_count, 3), wrist.shape),
    }
    invalid = {name: actual for name, (expected, actual) in expected_shapes.items() if actual != expected}
    if object_local.ndim != 2 or object_local.shape[1] != 3:
        invalid["object_points_local"] = object_local.shape
    if invalid:
        raise ValueError(f"invalid trajectory shapes: {invalid}")
    if not all(np.isfinite(value).all() for value in (mano, object_pos, object_quat, object_local, wrist)):
        raise ValueError("trajectory contains NaN/Inf")

    rotations = Rotation.from_quat(object_quat[:, [1, 2, 3, 0]]).as_matrix()
    object_world = np.einsum("tij,kj->tki", rotations, object_local) + object_pos[:, None, :]
    source_indices = np.asarray(data.get("source_frame_indices", np.arange(frames_count)), dtype=int)
    fps = float(fps_override if fps_override is not None else np.asarray(data.get("fps", 30.0)).item())
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    dynamic_points = np.concatenate((mano.reshape(-1, 3), object_world.reshape(-1, 3), wrist), axis=0)
    lower = dynamic_points.min(axis=0)
    upper = dynamic_points.max(axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.58, 0.12)
    origin = np.array([center[0] - radius * 0.86, center[1] - radius * 0.86, 0.0])
    axis_length = radius * 0.30

    static_traces = [
        _path_trace(wrist, "Full wrist path", "#c62828", "dash"),
        _path_trace(object_pos, "Full object-center path", "#ef6c00", "dot"),
        *_world_axes(origin, axis_length),
    ]

    def traces(frame: int) -> list[go.Scatter3d]:
        return [
            _mano_trace(mano[frame]),
            _object_trace(object_world[frame]),
            _wrist_trace(wrist[frame]),
            *static_traces,
        ]

    object_start = object_pos[0]
    object_delta = object_pos[-1] - object_start
    wrist_delta = wrist[-1] - wrist[0]

    def title(frame: int) -> str:
        return (
            f"+90° rotated DexYCB world trajectory | frame {frame}/{frames_count - 1} "
            f"(source {source_indices[frame]})<br>"
            f"<sup>object start=({object_start[0]:.3f}, {object_start[1]:.3f}, {object_start[2]:.3f}) m "
            f"| drag: rotate · wheel: zoom · slider: seek</sup>"
        )

    frame_duration = 1000.0 / fps
    figure_frames = [
        go.Frame(data=traces(frame), name=str(frame), layout={"title": title(frame)})
        for frame in range(frames_count)
    ]
    figure = go.Figure(data=traces(0), frames=figure_frames)
    figure.update_layout(
        title=title(0),
        template="plotly_white",
        height=880,
        margin={"l": 0, "r": 0, "b": 0, "t": 88},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.72)"},
        scene={
            "xaxis": {"title": "World X [m]", "range": [center[0] - radius, center[0] + radius]},
            "yaxis": {"title": "World Y [m]", "range": [center[1] - radius, center[1] + radius]},
            "zaxis": {"title": "World Z [m]", "range": [min(0.0, center[2] - radius), center[2] + radius]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": -1.65, "z": 1.15}},
        },
        annotations=[
            {
                "x": 0.99,
                "y": 0.02,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "right",
                "yanchor": "bottom",
                "align": "left",
                "showarrow": False,
                "bgcolor": "rgba(255,255,255,0.78)",
                "bordercolor": "#aaaaaa",
                "text": (
                    f"Object Δ: [{object_delta[0]:+.3f}, {object_delta[1]:+.3f}, {object_delta[2]:+.3f}] m<br>"
                    f"Wrist Δ: [{wrist_delta[0]:+.3f}, {wrist_delta[1]:+.3f}, {wrist_delta[2]:+.3f}] m"
                ),
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.04,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": frame_duration, "redraw": True},
                                        "transition": {"duration": 0}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"t": 44},
                "steps": [
                    {
                        "args": [[str(frame)], {"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate"}],
                        "label": str(frame),
                        "method": "animate",
                    }
                    for frame in range(frames_count)
                ],
            }
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return {
        "frames": frames_count,
        "fps": fps,
        "output": output_path.resolve(),
        "object_start": object_start,
        "object_delta": object_delta,
        "wrist_delta": wrist_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="Transformed world trajectory (.h5/.npz)")
    parser.add_argument("--out", required=True, type=Path, help="Standalone output HTML")
    parser.add_argument("--fps", type=float, help="Override playback FPS")
    args = parser.parse_args()
    summary = make_html(
        args.trajectory.expanduser().resolve(),
        args.out.expanduser().resolve(),
        args.fps,
    )
    print(f"Saved standalone HTML: {summary['output']}")
    print(f"  frames/fps:  {summary['frames']} / {summary['fps']:.3f}")
    print(f"  object start: {summary['object_start']}")
    print(f"  object delta: {summary['object_delta']}")
    print(f"  wrist delta:  {summary['wrist_delta']}")


if __name__ == "__main__":
    main()
