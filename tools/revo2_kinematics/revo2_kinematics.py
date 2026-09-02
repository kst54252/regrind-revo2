"""GUI-independent forward kinematics for the six-DoF Revo2 right hand.

The implementation uses metres and radians. NumPy/list inputs return a NumPy
array. If PyTorch is installed, a torch.Tensor input returns a differentiable
torch.Tensor on the same device and with the same floating-point dtype.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

try:  # PyTorch is optional for lightweight FK use outside the optimizer.
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None


_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
DEFAULT_KEYPOINTS_PATH = _MODULE_DIR / "revo2_keypoints.json"
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "USD/revo2_right/payloads/base.usda"
DEFAULT_PHYSICS_PATH = (
    _PROJECT_ROOT / "USD/revo2_right/payloads/Physics/physics.usda"
)


def _quat_to_matrix(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    """Convert a USD-order (w, x, y, z) quaternion to a rotation matrix."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("zero-length quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose(
    xyz: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    return _quat_to_matrix(quaternion_wxyz), np.asarray(xyz, dtype=np.float64)


# Zero-pose link transforms copied from revo2_right/payloads/base.usda.
# Each entry is child pose in its parent link frame.
_ZERO_POSES = {
    "right_thumb_metacarpal_link": _pose(
        (0.005619, 0.019867, 0.027825),
        (0.9964795, 0.0, 0.0, -0.083836645),
    ),
    "right_thumb_proximal_link": _pose(
        (0.0, 0.014227, 0.0),
        (0.9865567, 0.07652065, 0.1248039, 0.07262531),
    ),
    "right_thumb_distal_link": _pose((0.0, 0.052, 0.0)),
    "right_thumb_touch_link": _pose(
        (0.0, 0.0137927522748974, 0.00767659940411662),
        (0.9994431, -0.033368584, 0.0, 0.0),
    ),
    "right_index_proximal_link": _pose(
        (-0.0021181, 0.029568, 0.080876),
        (0.9921471, -0.0522806, 0.07836512, -0.08227881),
    ),
    "right_index_distal_link": _pose((0.0, 0.0, 0.032)),
    "right_index_touch_link": _pose(
        (0.014497, 0.0, 0.023303),
        (0.63609993, 0.73476934, 0.15418185, -0.17809795),
    ),
    "right_middle_proximal_link": _pose(
        (-0.0045767, 0.010051, 0.084993),
        (0.99668145, -0.017523544, 0.0746071, -0.027437024),
    ),
    "right_middle_distal_link": _pose((0.0, 0.0, 0.037)),
    "right_middle_touch_link": _pose(
        (0.015975871997898, 0.0, 0.0265894268903603),
        (0.9718583, 0.0, 0.23556612, 0.0),
    ),
    "right_ring_proximal_link": _pose(
        (-0.0046709, -0.010037, 0.083982),
        (0.996722, 0.017538698, 0.07406351, 0.027427714),
    ),
    "right_ring_distal_link": _pose((0.0, 0.0, 0.035)),
    "right_ring_touch_link": _pose(
        (0.0154392324697935, 0.0, 0.0260561041381566),
        (0.9718583, 0.0, 0.23556612, 0.0),
    ),
    "right_pinky_proximal_link": _pose(
        (-0.0023566, -0.029366, 0.078694),
        (0.9922455, 0.05238695, 0.07710652, 0.08221356),
    ),
    "right_pinky_distal_link": _pose((0.0, 0.0, 0.029)),
    "right_pinky_touch_link": _pose(
        (0.01401, 0.0, 0.021124),
        (0.97185856, 0.0, 0.23556514, 0.0),
    ),
}

_SUPPORTED_LINKS = {"right_hand_base_link", *_ZERO_POSES.keys()}


class Revo2Kinematics:
    """Compute the 21 semantic keypoints of the Revo2 right hand.

    Joint order is available in ``joint_names`` and is fixed to:
    thumb metacarpal, thumb proximal, index, middle, ring, pinky proximal.
    Distal joints are generated from the robot's mimic constraints.
    """

    joint_names = (
        "right_thumb_metacarpal_joint",
        "right_thumb_proximal_joint",
        "right_index_proximal_joint",
        "right_middle_proximal_joint",
        "right_ring_proximal_joint",
        "right_pinky_proximal_joint",
    )

    def __init__(
        self,
        keypoints_path: str | Path = DEFAULT_KEYPOINTS_PATH,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        physics_path: str | Path = DEFAULT_PHYSICS_PATH,
        base_transform: Any | None = None,
    ) -> None:
        self.keypoints_path = Path(keypoints_path).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.physics_path = Path(physics_path).expanduser().resolve()

        self._keypoint_names, self._parent_links, self._local_xyz = (
            self._load_keypoints(self.keypoints_path)
        )
        self._validate_parent_links_against_model(self.model_path)
        self._joint_lower, self._joint_upper = self._read_joint_limits(
            self.physics_path
        )

        if base_transform is None:
            base_transform = np.eye(4, dtype=np.float64)
        self._base_transform = np.asarray(base_transform, dtype=np.float64)
        if self._base_transform.shape != (4, 4):
            raise ValueError(
                "base_transform must have shape (4, 4), got "
                f"{self._base_transform.shape}"
            )
        if not np.isfinite(self._base_transform).all():
            raise ValueError("base_transform contains a non-finite value")

    @property
    def keypoint_names(self) -> tuple[str, ...]:
        return self._keypoint_names

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return independent-joint lower and upper limits in radians."""
        return self._joint_lower.copy(), self._joint_upper.copy()

    def get_keypoints(self, q: Any) -> Any:
        """Return keypoints with shape (21, 3), expressed in the base frame.

        If ``base_transform`` was supplied, its world transform is applied last.
        A torch input preserves autograd; other array-like inputs return NumPy.
        """
        is_torch = torch is not None and isinstance(q, torch.Tensor)
        if is_torch:
            if q.shape != (6,):
                raise ValueError(f"q must have shape (6,), got {tuple(q.shape)}")
            if not q.is_floating_point():
                raise TypeError("a torch q must have a floating-point dtype")
            if not bool(torch.isfinite(q).all()):
                raise ValueError("q contains a non-finite value")
        else:
            q = np.asarray(q, dtype=np.float64)
            if q.shape != (6,):
                raise ValueError(f"q must have shape (6,), got {q.shape}")
            if not np.isfinite(q).all():
                raise ValueError("q contains a non-finite value")

        links = self._forward_links(q)
        points = []
        for parent_link, local_xyz in zip(self._parent_links, self._local_xyz):
            rotation, translation = links[parent_link]
            local = self._constant(local_xyz, q)
            points.append(rotation @ local + translation)

        return torch.stack(points) if is_torch else np.stack(points)

    def _forward_links(self, q: Any) -> dict[str, tuple[Any, Any]]:
        base = self._constant(self._base_transform, q)
        links = {"right_hand_base_link": (base[:3, :3], base[:3, 3])}

        # The thumb metacarpal joint's child anchor is rotated 180 deg about X,
        # so its USD Z joint axis is -Z in the zero-pose child coordinates.
        meta_rel = self._moving_pose("right_thumb_metacarpal_link", "Z", -q[0], q)
        links["right_thumb_metacarpal_link"] = self._compose(
            links["right_hand_base_link"], meta_rel
        )
        thumb_prox_rel = self._moving_pose(
            "right_thumb_proximal_link", "X", q[1], q
        )
        links["right_thumb_proximal_link"] = self._compose(
            links["right_thumb_metacarpal_link"], thumb_prox_rel
        )
        thumb_dist_rel = self._moving_pose(
            "right_thumb_distal_link", "X", q[1], q
        )
        links["right_thumb_distal_link"] = self._compose(
            links["right_thumb_proximal_link"], thumb_dist_rel
        )
        links["right_thumb_touch_link"] = self._compose(
            links["right_thumb_distal_link"],
            self._fixed_pose("right_thumb_touch_link", q),
        )

        fingers = (
            ("index", q[2]),
            ("middle", q[3]),
            ("ring", q[4]),
            ("pinky", q[5]),
        )
        for finger, proximal_q in fingers:
            proximal_link = f"right_{finger}_proximal_link"
            distal_link = f"right_{finger}_distal_link"
            touch_link = f"right_{finger}_touch_link"
            links[proximal_link] = self._compose(
                links["right_hand_base_link"],
                self._moving_pose(proximal_link, "Y", proximal_q, q),
            )
            links[distal_link] = self._compose(
                links[proximal_link],
                self._moving_pose(distal_link, "Y", 1.155 * proximal_q, q),
            )
            links[touch_link] = self._compose(
                links[distal_link], self._fixed_pose(touch_link, q)
            )

        return links

    def _moving_pose(
        self, link_name: str, axis: str, angle: Any, like: Any
    ) -> tuple[Any, Any]:
        zero_rotation, zero_translation = _ZERO_POSES[link_name]
        rotation = self._constant(zero_rotation, like) @ self._axis_rotation(
            axis, angle
        )
        return rotation, self._constant(zero_translation, like)

    def _fixed_pose(self, link_name: str, like: Any) -> tuple[Any, Any]:
        rotation, translation = _ZERO_POSES[link_name]
        return self._constant(rotation, like), self._constant(translation, like)

    @staticmethod
    def _compose(
        parent: tuple[Any, Any], child: tuple[Any, Any]
    ) -> tuple[Any, Any]:
        parent_rotation, parent_translation = parent
        child_rotation, child_translation = child
        return (
            parent_rotation @ child_rotation,
            parent_rotation @ child_translation + parent_translation,
        )

    @staticmethod
    def _constant(value: Any, like: Any) -> Any:
        if torch is not None and isinstance(like, torch.Tensor):
            return torch.as_tensor(value, dtype=like.dtype, device=like.device)
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _axis_rotation(axis: str, angle: Any) -> Any:
        if torch is not None and isinstance(angle, torch.Tensor):
            c, s = torch.cos(angle), torch.sin(angle)
            zero, one = angle.new_zeros(()), angle.new_ones(())
            if axis == "X":
                rows = ((one, zero, zero), (zero, c, -s), (zero, s, c))
            elif axis == "Y":
                rows = ((c, zero, s), (zero, one, zero), (-s, zero, c))
            elif axis == "Z":
                rows = ((c, -s, zero), (s, c, zero), (zero, zero, one))
            else:
                raise ValueError(f"unsupported rotation axis: {axis}")
            return torch.stack(tuple(torch.stack(row) for row in rows))

        c, s = np.cos(angle), np.sin(angle)
        if axis == "X":
            return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
        if axis == "Y":
            return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
        if axis == "Z":
            return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        raise ValueError(f"unsupported rotation axis: {axis}")

    @staticmethod
    def _load_keypoints(
        path: Path,
    ) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
        try:
            with path.open("r", encoding="utf-8") as json_file:
                data = json.load(json_file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"keypoint JSON not found: {path}") from error

        items = list(data.values()) if isinstance(data, dict) else list(data)
        name_pattern = re.compile(r"^kp_(\d+)(?:_|$)")

        def number(item: dict[str, Any]) -> int:
            match = name_pattern.match(str(item.get("name", "")))
            if match is None:
                raise ValueError(f"invalid keypoint name: {item.get('name')!r}")
            return int(match.group(1))

        items.sort(key=number)
        numbers = [number(item) for item in items]
        if numbers != list(range(21)):
            raise ValueError(
                "keypoint JSON must contain each number kp_00..kp_20 exactly once; "
                f"found {numbers}"
            )

        try:
            names = tuple(str(item["name"]) for item in items)
            parents = tuple(str(item["parent_link"]) for item in items)
            xyz = np.asarray([item["xyz"] for item in items], dtype=np.float64)
        except KeyError as error:
            raise ValueError(f"missing keypoint JSON field: {error.args[0]}") from error
        if xyz.shape != (21, 3) or not np.isfinite(xyz).all():
            raise ValueError("keypoint xyz values must be finite with shape (21, 3)")
        return names, parents, xyz

    def _validate_parent_links_against_model(self, model_path: Path) -> None:
        try:
            model_text = model_path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise FileNotFoundError(f"robot model not found: {model_path}") from error
        except UnicodeDecodeError as error:
            raise ValueError(
                "model_path must point to an ASCII USDA model for link-name "
                f"validation, got: {model_path}"
            ) from error

        model_links = set(
            re.findall(
                r'\b(?:def|over)\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)?'
                r'"([^"\n]*_link)"',
                model_text,
            )
        )
        json_links = set(self._parent_links)
        missing_from_model = sorted(json_links - model_links)
        unsupported_by_fk = sorted(json_links - _SUPPORTED_LINKS)
        if missing_from_model or unsupported_by_fk:
            messages = []
            if missing_from_model:
                messages.append(
                    "not present in robot model: " + ", ".join(missing_from_model)
                )
            if unsupported_by_fk:
                messages.append(
                    "not implemented by Revo2 FK: " + ", ".join(unsupported_by_fk)
                )
            raise ValueError("keypoint parent_link validation failed; " + "; ".join(messages))

    def _read_joint_limits(self, physics_path: Path) -> tuple[np.ndarray, np.ndarray]:
        try:
            physics_text = physics_path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise FileNotFoundError(f"robot physics model not found: {physics_path}") from error

        lower_deg, upper_deg = [], []
        for joint_name in self.joint_names:
            block_match = re.search(
                rf'def\s+PhysicsRevoluteJoint\s+"{re.escape(joint_name)}".*?\n\s*\}}',
                physics_text,
                flags=re.DOTALL,
            )
            if block_match is None:
                raise ValueError(f"joint not found in physics model: {joint_name}")
            block = block_match.group(0)
            lower_match = re.search(r"physics:lowerLimit\s*=\s*([-+0-9.eE]+)", block)
            upper_match = re.search(r"physics:upperLimit\s*=\s*([-+0-9.eE]+)", block)
            if lower_match is None or upper_match is None:
                raise ValueError(f"joint limits missing for: {joint_name}")
            lower_deg.append(float(lower_match.group(1)))
            upper_deg.append(float(upper_match.group(1)))

        return np.deg2rad(lower_deg), np.deg2rad(upper_deg)


if __name__ == "__main__":
    fk = Revo2Kinematics()
    lower, upper = fk.get_joint_limits()
    q_test = 0.5 * (lower + upper)
    keypoints = fk.get_keypoints(q_test)

    np.set_printoptions(precision=8, suppress=True)
    print("joint order:", fk.joint_names)
    print("q [rad]:", q_test)
    print("keypoints [m]:\n", keypoints)
    print("shape:", keypoints.shape)
    print("all finite:", bool(np.isfinite(keypoints).all()))

    assert keypoints.shape == (21, 3)
    assert np.isfinite(keypoints).all()
