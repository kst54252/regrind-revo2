"""Create an interactive Plotly view of a retargeted hand-object sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


FINGER_CHAINS = (
    (0, 1, 2, 3, 17),
    (0, 4, 5, 6, 18),
    (0, 10, 11, 12, 19),
    (0, 7, 8, 9, 20),
    (0, 13, 14, 15, 16),
)


def _load(path: str) -> dict[str, np.ndarray]:
    if path.lower().endswith(".npz"):
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}
    with h5py.File(path, "r") as h5_file:
        return {name: h5_file[name][()] for name in h5_file}


def _decode_names(value, count: int, prefix: str) -> list[str]:
    if value is None:
        return [f"{prefix}_{index:02d}" for index in range(count)]
    names = np.asarray(value).reshape(-1)
    return [name.decode() if isinstance(name, bytes) else str(name) for name in names]


def _hand_trace(points, name, color, labels, visible=True):
    if not visible or not np.isfinite(points).all():
        return go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", name=name)
    x, y, z, text = [], [], [], []
    for chain in FINGER_CHAINS:
        for index in chain:
            x.append(points[index, 0])
            y.append(points[index, 1])
            z.append(points[index, 2])
            text.append(labels[index])
        x.append(None)
        y.append(None)
        z.append(None)
        text.append("")
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        text=text,
        hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 5},
        marker={"color": color, "size": 4},
    )


def _object_trace(points):
    labels = [f"object_{index:02d}" for index in range(len(points))]
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        text=labels,
        hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        mode="markers",
        name="Object",
        marker={"color": "#ffbf00", "size": 3, "opacity": 0.85},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a standalone interactive 3D retargeting visualization."
    )
    parser.add_argument("result", help="Retargeted .h5/.hdf5/.npz file")
    parser.add_argument("--out", help="Output HTML path")
    parser.add_argument("--fps", type=float, help="Playback FPS override")
    args = parser.parse_args()

    data = _load(args.result)
    human = np.asarray(data["mano_joint_coords"], dtype=float)
    robot = np.asarray(data["robot_keypoints"], dtype=float)
    object_pos = np.asarray(data["object_pos"], dtype=float)
    object_quat = np.asarray(data["object_quat"], dtype=float)
    object_local = np.asarray(data["object_points_local"], dtype=float)
    T = len(human)
    if human.shape != (T, 21, 3) or robot.shape != (T, 21, 3):
        raise ValueError(
            f"Expected MANO/Revo2 shapes (T,21,3), got {human.shape}, {robot.shape}"
        )
    if object_pos.shape != (T, 3) or object_quat.shape != (T, 4):
        raise ValueError("Expected object_pos (T,3) and object_quat (T,4)")

    object_world = np.empty((T, len(object_local), 3), dtype=float)
    for frame in range(T):
        object_world[frame] = (
            Rotation.from_quat(object_quat[frame]).apply(object_local)
            + object_pos[frame]
        )
    success = np.asarray(data.get("solver_success", np.isfinite(robot).all((1, 2))), dtype=bool)
    objective = np.asarray(data.get("objective_value", np.full(T, np.nan)), dtype=float)
    fps = float(args.fps if args.fps is not None else np.asarray(data.get("fps", 30.0)))
    if fps <= 0:
        raise ValueError("FPS must be > 0")

    robot_labels = _decode_names(data.get("robot_keypoint_names"), 21, "kp")
    human_labels = [f"MANO_{index:02d}" for index in range(21)]

    def traces(frame):
        return [
            _hand_trace(human[frame], "Human MANO", "#1f77b4", human_labels),
            _hand_trace(
                robot[frame], "Revo2", "#d62728", robot_labels, success[frame]
            ),
            _object_trace(object_world[frame]),
        ]

    def title(frame):
        status = "success" if success[frame] else "FAILURE"
        cost = f"{objective[frame]:.6g}" if np.isfinite(objective[frame]) else "n/a"
        return f"Frame {frame}/{T - 1} | {status} | objective={cost}"

    frames = [
        go.Frame(data=traces(frame), name=str(frame), layout={"title": title(frame)})
        for frame in range(T)
    ]
    all_points = np.concatenate(
        (human.reshape(-1, 3), robot.reshape(-1, 3), object_world.reshape(-1, 3)),
        axis=0,
    )
    all_points = all_points[np.isfinite(all_points).all(axis=1)]
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) * 0.55, 0.05)

    steps = [
        {
            "args": [[str(frame)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(frame),
            "method": "animate",
        }
        for frame in range(T)
    ]
    frame_duration = 1000.0 / fps
    figure = go.Figure(data=traces(0), frames=frames)
    figure.update_layout(
        title=title(0),
        template="plotly_white",
        height=850,
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
        legend={"x": 0.01, "y": 0.99},
        scene={
            "xaxis": {"title": "X [m]", "range": [center[0] - radius, center[0] + radius]},
            "yaxis": {"title": "Y [m]", "range": [center[1] - radius, center[1] + radius]},
            "zaxis": {"title": "Z [m]", "range": [center[2] - radius, center[2] + radius]},
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
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
    )

    output = Path(args.out) if args.out else Path(args.result).with_name(
        Path(args.result).stem + "_interactive.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"Saved interactive visualization to {output}")


if __name__ == "__main__":
    main()
