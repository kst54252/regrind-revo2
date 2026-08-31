"""Build a standalone interactive RB3 IK workspace and robot-pose viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go

try:
    from .rb3_kinematics import RB3730Kinematics
except ImportError:
    from rb3_kinematics import RB3730Kinematics


LINK_LABELS = ("RB3 base", "shoulder", "elbow", "wrist center", "Revo2 wrist")


def _load(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    with h5py.File(path, "r") as h5_file:
        return {key: h5_file[key][()] for key in h5_file}


def _decode_strings(values) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def _range(points: np.ndarray, padding_fraction=0.08, minimum_padding=0.05):
    finite = points[np.isfinite(points).all(axis=1)]
    lower, upper = finite.min(axis=0), finite.max(axis=0)
    padding = np.maximum((upper - lower) * padding_fraction, minimum_padding)
    return [[float(lower[i] - padding[i]), float(upper[i] + padding[i])] for i in range(3)]


def _robot_trace(chain: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=chain[:, 0], y=chain[:, 1], z=chain[:, 2],
        mode="lines+markers+text",
        text=LINK_LABELS,
        textposition="top center",
        name="RB3 + mounted wrist",
        line={"color": "#202020", "width": 9},
        marker={"color": ["#000000", "#9467bd", "#9467bd", "#9467bd", "#d62728"], "size": 7},
        hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    )


def _current_traces(
    frame: int,
    chains: np.ndarray,
    target: np.ndarray,
    fk_wrist: np.ndarray,
    object_pos: np.ndarray,
    full_success: np.ndarray,
) -> list[go.Scatter3d]:
    status_color = "#2ca02c" if full_success[frame] else "#d62728"
    return [
        _robot_trace(chains[frame]),
        go.Scatter3d(
            x=[target[frame, 0]], y=[target[frame, 1]], z=[target[frame, 2]],
            mode="markers", name="Current target wrist",
            marker={"color": status_color, "size": 10, "symbol": "diamond"},
            hovertemplate="Target wrist<br>x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>",
        ),
        go.Scatter3d(
            x=[fk_wrist[frame, 0]], y=[fk_wrist[frame, 1]], z=[fk_wrist[frame, 2]],
            mode="markers", name="Current FK wrist",
            marker={"color": "#111111", "size": 8, "symbol": "x"},
            hovertemplate="FK wrist<br>x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>",
        ),
        go.Scatter3d(
            x=[object_pos[frame, 0]], y=[object_pos[frame, 1]], z=[object_pos[frame, 2]],
            mode="markers", name="Current object origin",
            marker={"color": "#ffbf00", "size": 9, "symbol": "square"},
            hovertemplate="Object origin<br>x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_trajectory")
    parser.add_argument("full_pose_result")
    parser.add_argument("diagnostic_result")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workspace-points", type=int, default=20_000)
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()

    world = _load(args.world_trajectory)
    full = _load(args.full_pose_result)
    diagnostic = _load(args.diagnostic_result)
    target = np.asarray(diagnostic["target_wrist_pos"], dtype=float)
    object_pos = np.asarray(world["object_pos_world"], dtype=float)
    rb3_q = np.asarray(full["rb3_joints"], dtype=float)
    fk_wrist = np.asarray(full["fk_wrist_pos"], dtype=float)
    full_success = np.asarray(diagnostic["full_pose_success"], dtype=bool)
    position_success = np.asarray(diagnostic["position_only_success"], dtype=bool)
    full_position_error = np.asarray(
        diagnostic["full_pose_position_error_m"], dtype=float
    )
    full_orientation_error = np.asarray(
        diagnostic["full_pose_orientation_error_rad"], dtype=float
    )
    classifications = _decode_strings(diagnostic["classification"])
    failure_segments = np.asarray(diagnostic["failure_segments"], dtype=int).reshape(-1, 2)
    workspace = np.asarray(diagnostic["workspace_points"], dtype=float)
    T = len(target)
    if not (
        target.shape == (T, 3)
        and object_pos.shape == (T, 3)
        and rb3_q.shape == (T, 6)
        and fk_wrist.shape == (T, 3)
        and full_success.shape == (T,)
        and position_success.shape == (T,)
        and len(classifications) == T
    ):
        raise ValueError("trajectory and diagnostic shapes do not agree")
    if not all(
        np.isfinite(value).all()
        for value in (target, object_pos, rb3_q, fk_wrist, workspace)
    ):
        raise ValueError("input contains NaN/Inf")

    base_position = np.asarray(full.get("rb3_base_position", np.zeros(3)), dtype=float)
    base_quaternion = np.asarray(
        full.get("rb3_base_quat_xyzw", (0.0, 0.0, 0.0, 1.0)), dtype=float
    )
    robot = RB3730Kinematics(
        base_position=base_position, base_quaternion_xyzw=base_quaternion
    )
    chains = np.stack([robot.forward_chain_points(q) for q in rb3_q])
    if not np.allclose(chains[:, -1], fk_wrist, atol=1.0e-9):
        raise RuntimeError("robot skeleton endpoint does not match stored FK wrist")

    count = min(args.workspace_points, len(workspace))
    rng = np.random.default_rng(730)
    workspace = workspace[rng.choice(len(workspace), size=count, replace=False)]
    fps = float(args.fps if args.fps is not None else np.asarray(world.get("fps", 30.0)).item())
    if fps <= 0:
        raise ValueError("FPS must be > 0")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=workspace[:, 0], y=workspace[:, 1], z=workspace[:, 2],
            mode="markers", name=f"RB3 workspace ({count:,} samples)",
            marker={"color": "#999999", "size": 1.2, "opacity": 0.12},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=target[:, 0], y=target[:, 1], z=target[:, 2], mode="lines",
            name="Target wrist trajectory", line={"color": "#377eb8", "width": 4},
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=target[full_success, 0], y=target[full_success, 1], z=target[full_success, 2],
            mode="markers", name="Full-pose success",
            marker={"color": "#2ca02c", "size": 4},
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=target[~full_success, 0], y=target[~full_success, 1], z=target[~full_success, 2],
            mode="markers", name="Full-pose failure",
            marker={"color": "#d62728", "size": 5, "symbol": "x"},
        )
    )
    for start, end in failure_segments:
        segment = target[start : end + 1]
        figure.add_trace(
            go.Scatter3d(
                x=segment[:, 0], y=segment[:, 1], z=segment[:, 2], mode="lines",
                name=f"Failure segment {start}-{end}",
                line={"color": "#ff7f0e", "width": 8},
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=object_pos[:, 0], y=object_pos[:, 1], z=object_pos[:, 2], mode="lines",
            name="Object trajectory", line={"color": "#d4a000", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[object_pos[0, 0]], y=[object_pos[0, 1]], z=[object_pos[0, 2]],
            mode="markers", name="Initial object position",
            marker={"color": "#ffbf00", "size": 11, "symbol": "diamond"},
        )
    )

    axis_colors = ("#d62728", "#2ca02c", "#1f77b4")
    axis_names = ("RB3 base +X", "RB3 base +Y", "RB3 base +Z")
    for index, (color, name) in enumerate(zip(axis_colors, axis_names)):
        endpoint = base_position + 0.16 * robot.base_rotation[:, index]
        figure.add_trace(
            go.Scatter3d(
                x=[base_position[0], endpoint[0]],
                y=[base_position[1], endpoint[1]],
                z=[base_position[2], endpoint[2]],
                mode="lines", name=name, line={"color": color, "width": 6},
            )
        )

    dynamic_trace_indices = list(range(len(figure.data), len(figure.data) + 4))
    for trace in _current_traces(0, chains, target, fk_wrist, object_pos, full_success):
        figure.add_trace(trace)

    def frame_title(frame: int) -> str:
        full_status = "SUCCESS" if full_success[frame] else "FAIL"
        position_status = "OK" if position_success[frame] else "FAIL"
        return (
            f"Frame {frame}/{T - 1} | full-pose {full_status} | "
            f"position-only {position_status} | {classifications[frame]} | "
            f"pos err={full_position_error[frame]:.5f} m, "
            f"ori err={full_orientation_error[frame]:.5f} rad"
        )

    figure.frames = [
        go.Frame(
            name=str(frame),
            data=_current_traces(
                frame, chains, target, fk_wrist, object_pos, full_success
            ),
            traces=dynamic_trace_indices,
            layout={"title": frame_title(frame)},
        )
        for frame in range(T)
    ]

    focus_points = np.concatenate(
        (target, object_pos, chains.reshape(-1, 3), fk_wrist), axis=0
    )
    full_points = np.concatenate((workspace, focus_points), axis=0)
    focus_range = _range(focus_points, padding_fraction=0.12, minimum_padding=0.08)
    full_range = _range(full_points, padding_fraction=0.03, minimum_padding=0.05)
    frame_duration = 1000.0 / fps
    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"t": 55},
            "steps": [
                {
                    "label": str(frame),
                    "method": "animate",
                    "args": [
                        [str(frame)],
                        {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
                    ],
                }
                for frame in range(T)
            ],
        }
    ]
    figure.update_layout(
        title=frame_title(0),
        template="plotly_white",
        height=900,
        margin={"l": 0, "r": 0, "b": 20, "t": 90},
        legend={"x": 0.01, "y": 0.98, "bgcolor": "rgba(255,255,255,0.78)"},
        scene={
            "xaxis": {"title": "World X [m]", "range": focus_range[0]},
            "yaxis": {"title": "World Y [m]", "range": focus_range[1]},
            "zaxis": {"title": "World Z [m]", "range": focus_range[2]},
            "aspectmode": "data",
        },
        sliders=sliders,
        updatemenus=[
            {
                "type": "buttons", "direction": "left", "x": 0.0, "y": 1.10,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"fromcurrent": True, "frame": {"duration": frame_duration, "redraw": True}, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            },
            {
                "type": "buttons", "direction": "left", "x": 0.48, "y": 1.10,
                "buttons": [
                    {
                        "label": "Trajectory focus",
                        "method": "relayout",
                        "args": [{"scene.xaxis.range": focus_range[0], "scene.yaxis.range": focus_range[1], "scene.zaxis.range": focus_range[2]}],
                    },
                    {
                        "label": "Full workspace",
                        "method": "relayout",
                        "args": [{"scene.xaxis.range": full_range[0], "scene.yaxis.range": full_range[1], "scene.zaxis.range": full_range[2]}],
                    },
                ],
            },
        ],
    )
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )
    print(f"Saved standalone interactive diagnostic to {output}")
    print(f"frames={T}, workspace points shown={count}, size={output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
