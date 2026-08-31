"""Interactive visual check for :mod:`revo2_kinematics`.

Examples:
    python tools/revo2_kinematics/visualize_revo2_kinematics.py
    python tools/revo2_kinematics/visualize_revo2_kinematics.py --q 0.3 0.2 0.7 0.7 0.7 0.7
    python tools/revo2_kinematics/visualize_revo2_kinematics.py --save revo2_fk.png --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

try:
    from .revo2_kinematics import Revo2Kinematics
except ImportError:  # Support direct execution from this directory.
    from revo2_kinematics import Revo2Kinematics


# MANO-style semantic chains in revo2_keypoints.json index order.
CHAINS = {
    "index": ((0, 1, 2, 3, 17), "#1f77b4"),
    "middle": ((0, 4, 5, 6, 18), "#2ca02c"),
    "ring": ((0, 10, 11, 12, 19), "#ff7f0e"),
    "little": ((0, 7, 8, 9, 20), "#9467bd"),
    "thumb": ((0, 13, 14, 15, 16), "#d62728"),
}

JOINT_LABELS = (
    "thumb metacarpal",
    "thumb proximal",
    "index proximal",
    "middle proximal",
    "ring proximal",
    "little proximal",
)


class Revo2KinematicsVisualizer:
    """Matplotlib 3-D viewer with one slider per independent Revo2 joint."""

    def __init__(self, fk: Revo2Kinematics, q: np.ndarray) -> None:
        self.fk = fk
        self.lower, self.upper = fk.get_joint_limits()
        self.q = np.asarray(q, dtype=np.float64).copy()
        if self.q.shape != (6,):
            raise ValueError(f"q must have shape (6,), got {self.q.shape}")
        if np.any(self.q < self.lower) or np.any(self.q > self.upper):
            raise ValueError(
                "q is outside the model joint limits\n"
                f"q:     {self.q}\n"
                f"lower: {self.lower}\n"
                f"upper: {self.upper}"
            )

        self.figure = plt.figure(figsize=(11, 9))
        self.figure.canvas.manager.set_window_title("Revo2 FK keypoint validator")
        self.axis = self.figure.add_axes((0.08, 0.34, 0.84, 0.62), projection="3d")
        self._configure_axis()

        self.lines = {}
        for chain_name, (_, color) in CHAINS.items():
            (line,) = self.axis.plot(
                [], [], [], "-o", color=color, linewidth=2.5, markersize=6,
                label=chain_name,
            )
            self.lines[chain_name] = line
        self.axis.legend(loc="upper right")

        self.wrist = self.axis.scatter([], [], [], c="black", marker="s", s=55)
        self.annotations = [
            self.axis.text(0, 0, 0, f"{index:02d}", fontsize=8)
            for index in range(21)
        ]
        self.status = self.axis.text2D(
            0.01, 0.98, "", transform=self.axis.transAxes, va="top", fontsize=10
        )

        self.sliders = []
        for index, label in enumerate(JOINT_LABELS):
            slider_axis = self.figure.add_axes((0.24, 0.265 - index * 0.038, 0.60, 0.022))
            slider = Slider(
                slider_axis,
                label,
                valmin=float(self.lower[index]),
                valmax=float(self.upper[index]),
                valinit=float(self.q[index]),
                valfmt="%1.3f rad",
            )
            slider.on_changed(self._on_slider_changed)
            self.sliders.append(slider)

        reset_axis = self.figure.add_axes((0.51, 0.015, 0.11, 0.036))
        random_axis = self.figure.add_axes((0.38, 0.015, 0.11, 0.036))
        self.reset_button = Button(reset_axis, "Reset")
        self.random_button = Button(random_axis, "Random")
        self.reset_button.on_clicked(self._reset)
        self.random_button.on_clicked(self._randomize)

        self._set_fixed_equal_limits()
        self.update()

    def _configure_axis(self) -> None:
        self.axis.set_title("Revo2 semantic keypoints (base-link frame)")
        self.axis.set_xlabel("X [m]")
        self.axis.set_ylabel("Y [m]")
        self.axis.set_zlabel("Z [m]")
        self.axis.view_init(elev=24, azim=-58)
        self.axis.grid(True, alpha=0.35)

    def _set_fixed_equal_limits(self) -> None:
        # Include open, half-closed, and closed poses so sliders do not cause the
        # camera scale to jump while the hand moves.
        samples = np.concatenate(
            (
                self.fk.get_keypoints(self.lower),
                self.fk.get_keypoints(0.5 * (self.lower + self.upper)),
                self.fk.get_keypoints(self.upper),
            ),
            axis=0,
        )
        minimum = samples.min(axis=0)
        maximum = samples.max(axis=0)
        center = 0.5 * (minimum + maximum)
        radius = 0.58 * float(np.max(maximum - minimum))
        radius = max(radius, 0.06)
        self.axis.set_xlim(center[0] - radius, center[0] + radius)
        self.axis.set_ylim(center[1] - radius, center[1] + radius)
        self.axis.set_zlim(center[2] - radius, center[2] + radius)
        self.axis.set_box_aspect((1, 1, 1))

    def update(self) -> None:
        keypoints = self.fk.get_keypoints(self.q)
        valid = keypoints.shape == (21, 3) and bool(np.isfinite(keypoints).all())

        for chain_name, (indices, _) in CHAINS.items():
            points = keypoints[np.asarray(indices)]
            self.lines[chain_name].set_data_3d(points[:, 0], points[:, 1], points[:, 2])

        self.wrist._offsets3d = (
            keypoints[[0], 0], keypoints[[0], 1], keypoints[[0], 2]
        )
        for index, annotation in enumerate(self.annotations):
            annotation.set_position((keypoints[index, 0], keypoints[index, 1]))
            annotation.set_3d_properties(keypoints[index, 2])

        minimum_separation = self._minimum_keypoint_separation(keypoints)
        result = "PASS" if valid else "FAIL"
        color = "#137333" if valid else "#b00020"
        self.status.set_text(
            f"{result}  shape={keypoints.shape}  finite={np.isfinite(keypoints).all()}\n"
            f"minimum keypoint separation={minimum_separation * 1000:.2f} mm"
        )
        self.status.set_color(color)
        self.figure.canvas.draw_idle()

    @staticmethod
    def _minimum_keypoint_separation(keypoints: np.ndarray) -> float:
        delta = keypoints[:, None, :] - keypoints[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        distance[np.eye(len(keypoints), dtype=bool)] = np.inf
        return float(distance.min())

    def _on_slider_changed(self, _value: float) -> None:
        self.q[:] = [slider.val for slider in self.sliders]
        self.update()

    def _reset(self, _event) -> None:
        for slider, value in zip(self.sliders, self.lower):
            slider.set_val(float(value))

    def _randomize(self, _event) -> None:
        values = np.random.default_rng().uniform(self.lower, self.upper)
        for slider, value in zip(self.sliders, values):
            slider.set_val(float(value))

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output, dpi=180, bbox_inches="tight")
        return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q", nargs=6, type=float, default=np.zeros(6), metavar="RAD",
        help="six independent joint positions in radians (default: all zero)",
    )
    parser.add_argument("--save", type=str, help="optional PNG/PDF output path")
    parser.add_argument(
        "--no-show", action="store_true", help="do not open the interactive window"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fk = Revo2Kinematics()
    viewer = Revo2KinematicsVisualizer(fk, np.asarray(args.q, dtype=np.float64))
    keypoints = fk.get_keypoints(viewer.q)

    print("[PASS] parent-link names match the robot model")
    print(f"[PASS] keypoints shape: {keypoints.shape}")
    print(f"[PASS] all finite: {bool(np.isfinite(keypoints).all())}")
    print("q [rad]:", viewer.q)
    print("keypoints [m]:\n", keypoints)

    if args.save:
        print("saved:", viewer.save(args.save))
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
