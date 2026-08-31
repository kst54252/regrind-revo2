#!/usr/bin/env python3
"""Generate standalone interactive Plotly HTML for preprocessed DexYCB data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


def _text(value) -> str:
    value = np.asarray(value).item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _hand_trace(points: np.ndarray, name: str, color: str, opacity: float = 1.0):
    x, y, z, labels = [], [], [], []
    for chain in FINGER_CHAINS:
        for index in chain:
            x.append(points[index, 0])
            y.append(points[index, 1])
            z.append(points[index, 2])
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
        hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        mode="lines+markers",
        name=name,
        opacity=opacity,
        line={"color": color, "width": 6},
        marker={"color": color, "size": 4},
    )


def _object_trace(points: np.ndarray, name: str):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        text=[f"object_{index:02d}" for index in range(len(points))],
        hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        mode="markers",
        name=name,
        marker={"color": "#ffbf00", "size": 4, "opacity": 0.9},
    )


def make_html(input_path: Path, output_path: Path) -> dict:
    with np.load(input_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    mano = np.asarray(data["mano_joint_coords_right_mano21"], dtype=float)
    original = np.asarray(data["mano_joint_coords_original"], dtype=float)
    object_pos = np.asarray(data["object_pos"], dtype=float)
    object_quat = np.asarray(data["object_quat"], dtype=float)
    object_local = np.asarray(data["object_points_local"], dtype=float)
    source_indices = np.asarray(data["source_frame_indices"], dtype=int)
    sequence = _text(data["sequence_name"])
    source_side = _text(data["source_hand_side"])
    object_name = _text(data["grasped_object_name"])
    fps = float(np.asarray(data["fps"]))
    frames_count = len(mano)
    if mano.shape != (frames_count, 21, 3) or original.shape != mano.shape:
        raise ValueError(f"invalid MANO shapes: {mano.shape}, {original.shape}")
    rotations = Rotation.from_quat(object_quat[:, [1, 2, 3, 0]]).as_matrix()
    object_world = (
        np.einsum("tij,kj->tki", rotations, object_local)
        + object_pos[:, None, :]
    )

    show_original = source_side == "left"

    def traces(frame: int):
        result = [
            _hand_trace(mano[frame], "MANO (Revo2 right-hand input)", "#1f77b4"),
            _object_trace(object_world[frame], object_name),
        ]
        if show_original:
            result.append(
                _hand_trace(original[frame], "Original left MANO", "#7f7f7f", 0.45)
            )
        return result

    def title(frame: int) -> str:
        return (
            f"DexYCB {sequence} | valid {frame}/{frames_count - 1} | "
            f"source frame {source_indices[frame]} | {source_side}→right"
        )

    all_points = np.concatenate(
        (mano.reshape(-1, 3), original.reshape(-1, 3), object_world.reshape(-1, 3)),
        axis=0,
    )
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) * 0.55, 0.06)
    frame_duration = 1000.0 / fps
    figure_frames = [
        go.Frame(data=traces(frame), name=str(frame), layout={"title": title(frame)})
        for frame in range(frames_count)
    ]
    figure = go.Figure(data=traces(0), frames=figure_frames)
    figure.update_layout(
        title=title(0),
        template="plotly_white",
        height=850,
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
        legend={"x": 0.01, "y": 0.99},
        scene={
            "xaxis": {"title": "Camera X [m]", "range": [center[0] - radius, center[0] + radius]},
            "yaxis": {"title": "Camera Y [m]", "range": [center[1] - radius, center[1] + radius]},
            "zaxis": {"title": "Camera Z [m]", "range": [center[2] - radius, center[2] + radius]},
            "aspectmode": "cube",
        },
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
                "currentvalue": {"prefix": "Valid frame: "},
                "pad": {"t": 40},
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
        "sequence": sequence,
        "frames": frames_count,
        "source_side": source_side,
        "object": object_name,
        "html": str(output_path.resolve()),
        "input": str(input_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = make_html(args.input.expanduser().resolve(), args.out.expanduser().resolve())
    print(
        f"Saved {summary['sequence']} ({summary['frames']} frames) interactive HTML: "
        f"{summary['html']}"
    )


if __name__ == "__main__":
    main()
