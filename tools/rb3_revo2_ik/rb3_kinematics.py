"""GUI-independent FK and bounded numerical IK for the RB3-730 + Revo2 mount."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG = _MODULE_DIR / "rb3_model.json"
_AXES = {
    "X": np.array((1.0, 0.0, 0.0)),
    "Y": np.array((0.0, 1.0, 0.0)),
    "Z": np.array((0.0, 0.0, 1.0)),
}


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    optimizer_success: bool
    position_error_m: float
    orientation_error_rad: float
    fk_position: np.ndarray
    fk_quaternion_xyzw: np.ndarray
    joint_limit_violation: bool
    max_joint_limit_violation_rad: float
    finite: bool
    cost: float
    nfev: int
    message: str
    candidates_evaluated: int


@dataclass(frozen=True)
class PositionIKResult:
    q: np.ndarray
    success: bool
    optimizer_success: bool
    position_error_m: float
    fk_position: np.ndarray
    joint_limit_violation: bool
    min_joint_limit_margin_rad: float
    finite: bool
    cost: float
    nfev: int
    message: str
    candidates_evaluated: int


def _normalize_quaternion_xyzw(quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"quaternion must be finite shape (4,), got {quaternion}")
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-12:
        raise ValueError("zero-length quaternion")
    return quaternion / norm


class RB3730Kinematics:
    """FK/IK for the six RB3 joints and the mounted Revo2 wrist frame.

    The target frame is the actual ``right_hand_base_link`` in the composed
    ``USD/rb3_revo2.usd`` stage. Its link6-relative pose is loaded from the
    model snapshot; it is deliberately not approximated by the stock RB3 TCP.
    """

    def __init__(
        self,
        model_config: str | Path = DEFAULT_MODEL_CONFIG,
        base_position=(0.0, 0.0, 0.0),
        base_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        verify_model_hash: bool = True,
    ) -> None:
        self.model_config_path = Path(model_config).expanduser().resolve()
        with self.model_config_path.open("r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        self.joint_names = tuple(self.config["joint_names"])
        self.joint_axes = np.stack([_AXES[name] for name in self.config["joint_axes"]])
        self.joint_offsets = np.asarray(
            self.config["joint_offsets_xyz_m"], dtype=float
        )
        self.joint_lower = np.deg2rad(
            np.asarray(self.config["joint_lower_deg"], dtype=float)
        )
        self.joint_upper = np.deg2rad(
            np.asarray(self.config["joint_upper_deg"], dtype=float)
        )
        self.link6_to_wrist_position = np.asarray(
            self.config["link6_to_mounted_wrist_xyz_m"], dtype=float
        )
        self.link6_to_wrist_rotation = Rotation.from_quat(
            _normalize_quaternion_xyzw(
                self.config["link6_to_mounted_wrist_quat_xyzw"]
            )
        ).as_matrix()
        self.base_position = np.asarray(base_position, dtype=float)
        self.base_rotation = Rotation.from_quat(
            _normalize_quaternion_xyzw(base_quaternion_xyzw)
        ).as_matrix()
        if self.base_position.shape != (3,) or not np.isfinite(self.base_position).all():
            raise ValueError("base_position must be finite shape (3,)")
        if self.joint_offsets.shape != (6, 3):
            raise ValueError("model joint offsets must have shape (6,3)")
        if not np.all(self.joint_lower < self.joint_upper):
            raise ValueError("invalid RB3 joint limits")
        if verify_model_hash:
            self._verify_source_model_hash()

    @property
    def source_usd_path(self) -> Path:
        return (self.model_config_path.parent / self.config["source_usd"]).resolve()

    @property
    def mounted_wrist_frame(self) -> str:
        return str(self.config["mounted_wrist_frame"])

    def _verify_source_model_hash(self) -> None:
        path = self.source_usd_path
        if not path.is_file():
            raise FileNotFoundError(f"RB3/Revo2 source USD not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = self.config.get("source_usd_sha256")
        if expected and digest != expected:
            warnings.warn(
                "RB3/Revo2 USD changed after rb3_model.json was extracted: "
                f"expected {expected}, got {digest}. Re-extract axes, limits, and mount transform.",
                stacklevel=2,
            )

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.joint_lower.copy(), self.joint_upper.copy()

    def forward(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Return mounted wrist position and XYZW quaternion in world frame."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError(f"q must be finite shape (6,), got {q}")
        rotation = self.base_rotation.copy()
        position = self.base_position.copy()
        for offset, axis, angle in zip(self.joint_offsets, self.joint_axes, q):
            position = position + rotation @ offset
            rotation = rotation @ Rotation.from_rotvec(axis * angle).as_matrix()
        position = position + rotation @ self.link6_to_wrist_position
        rotation = rotation @ self.link6_to_wrist_rotation
        return position, Rotation.from_matrix(rotation).as_quat()

    def forward_batch(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized mounted-wrist FK for joint configurations shaped ``(N,6)``."""
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != 6 or not np.isfinite(q).all():
            raise ValueError(f"q must be finite shape (N,6), got {q.shape}")
        count = len(q)
        rotation = np.broadcast_to(self.base_rotation, (count, 3, 3)).copy()
        position = np.broadcast_to(self.base_position, (count, 3)).copy()
        for joint_index, (offset, axis) in enumerate(
            zip(self.joint_offsets, self.joint_axes)
        ):
            position += np.einsum("nij,j->ni", rotation, offset)
            joint_rotation = Rotation.from_rotvec(
                q[:, joint_index, None] * axis[None]
            ).as_matrix()
            rotation = rotation @ joint_rotation
        position += np.einsum(
            "nij,j->ni", rotation, self.link6_to_wrist_position
        )
        rotation = rotation @ self.link6_to_wrist_rotation
        return position, Rotation.from_matrix(rotation).as_quat()

    def forward_chain_points(self, q) -> np.ndarray:
        """Return base, shoulder, elbow, wrist-center, and mounted-wrist points."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError(f"q must be finite shape (6,), got {q}")
        rotation = self.base_rotation.copy()
        position = self.base_position.copy()
        points = [position.copy()]
        for joint_index, (offset, axis, angle) in enumerate(
            zip(self.joint_offsets, self.joint_axes, q)
        ):
            position = position + rotation @ offset
            if joint_index in (0, 2, 4):
                points.append(position.copy())
            rotation = rotation @ Rotation.from_rotvec(axis * angle).as_matrix()
        position = position + rotation @ self.link6_to_wrist_position
        points.append(position.copy())
        return np.stack(points)

    def pose_error(self, q, target_position, target_quaternion_xyzw):
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,) or not np.isfinite(target_position).all():
            raise ValueError("target_position must be finite shape (3,)")
        target_quaternion_xyzw = _normalize_quaternion_xyzw(target_quaternion_xyzw)
        fk_position, fk_quaternion = self.forward(q)
        target_rotation = Rotation.from_quat(target_quaternion_xyzw).as_matrix()
        fk_rotation = Rotation.from_quat(fk_quaternion).as_matrix()
        position_error = float(np.linalg.norm(fk_position - target_position))
        orientation_error = float(
            np.linalg.norm(
                Rotation.from_matrix(target_rotation.T @ fk_rotation).as_rotvec()
            )
        )
        return position_error, orientation_error, fk_position, fk_quaternion

    def _residual(
        self,
        q,
        target_position,
        target_rotation,
        position_weight,
    ) -> np.ndarray:
        fk_position, fk_quaternion = self.forward(q)
        fk_rotation = Rotation.from_quat(fk_quaternion).as_matrix()
        orientation_residual = Rotation.from_matrix(
            target_rotation.T @ fk_rotation
        ).as_rotvec()
        return np.concatenate(
            (position_weight * (fk_position - target_position), orientation_residual)
        )

    def _position_residual(self, q, target_position) -> np.ndarray:
        return self.forward(q)[0] - target_position

    def _candidate_seeds(self, warm_start: np.ndarray, neutral: np.ndarray):
        seeds = [warm_start, neutral]
        elbow_mirror = warm_start.copy()
        elbow_mirror[1:3] *= -1.0
        seeds.append(elbow_mirror)
        base_alternative = warm_start.copy()
        base_alternative[0] += np.pi
        seeds.append(base_alternative)
        result = []
        for seed in seeds:
            clipped = np.clip(seed, self.joint_lower + 1.0e-10, self.joint_upper - 1.0e-10)
            if not any(np.allclose(clipped, existing) for existing in result):
                result.append(clipped)
        return result

    def inverse(
        self,
        target_position,
        target_quaternion_xyzw,
        initial_q=None,
        neutral_q=None,
        position_tolerance_m: float = 1.0e-4,
        orientation_tolerance_rad: float = 1.0e-3,
        position_weight: float = 10.0,
        max_nfev: int = 800,
    ) -> IKResult:
        """Solve bounded pose IK and prefer the solution nearest ``initial_q``."""
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,) or not np.isfinite(target_position).all():
            raise ValueError("target_position must be finite shape (3,)")
        target_quaternion_xyzw = _normalize_quaternion_xyzw(target_quaternion_xyzw)
        target_rotation = Rotation.from_quat(target_quaternion_xyzw).as_matrix()
        warm_start = np.zeros(6) if initial_q is None else np.asarray(initial_q, dtype=float)
        neutral = np.zeros(6) if neutral_q is None else np.asarray(neutral_q, dtype=float)
        if warm_start.shape != (6,) or not np.isfinite(warm_start).all():
            raise ValueError("initial_q must be finite shape (6,)")
        if neutral.shape != (6,) or not np.isfinite(neutral).all():
            raise ValueError("neutral_q must be finite shape (6,)")

        candidates = []
        for seed in self._candidate_seeds(warm_start, neutral):
            solution = least_squares(
                self._residual,
                seed,
                args=(target_position, target_rotation, position_weight),
                bounds=(self.joint_lower, self.joint_upper),
                method="trf",
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=max_nfev,
            )
            q = np.asarray(solution.x, dtype=float)
            pos_error, ori_error, fk_position, fk_quaternion = self.pose_error(
                q, target_position, target_quaternion_xyzw
            )
            violation = np.maximum(self.joint_lower - q, 0.0) + np.maximum(
                q - self.joint_upper, 0.0
            )
            finite = bool(
                np.isfinite(q).all()
                and np.isfinite(fk_position).all()
                and np.isfinite(fk_quaternion).all()
            )
            accepted = bool(
                solution.success
                and finite
                and pos_error <= position_tolerance_m
                and ori_error <= orientation_tolerance_rad
                and not np.any(violation > 1.0e-9)
            )
            candidates.append(
                {
                    "q": q,
                    "accepted": accepted,
                    "solution": solution,
                    "position_error": pos_error,
                    "orientation_error": ori_error,
                    "fk_position": fk_position,
                    "fk_quaternion": fk_quaternion,
                    "violation": violation,
                    "finite": finite,
                    "distance": float(np.linalg.norm(q - warm_start)),
                    "pose_score": (
                        pos_error / position_tolerance_m
                        + ori_error / orientation_tolerance_rad
                    ),
                }
            )

        accepted = [candidate for candidate in candidates if candidate["accepted"]]
        if accepted:
            best = min(accepted, key=lambda candidate: candidate["distance"])
        else:
            # Unreachable targets commonly produce equivalent least-squares
            # minima on two periodic wrist branches. Numerical noise at the
            # 1e-6 score level must not cause a multi-radian branch jump.
            best_pose_score = min(candidate["pose_score"] for candidate in candidates)
            equivalent_threshold = best_pose_score + 1.0e-4 * max(
                1.0, best_pose_score
            )
            equivalent = [
                candidate
                for candidate in candidates
                if candidate["pose_score"] <= equivalent_threshold
            ]
            best = min(equivalent, key=lambda candidate: candidate["distance"])
        solution = best["solution"]
        violation = best["violation"]
        message = str(solution.message)
        if not best["accepted"]:
            message = (
                f"pose tolerance not met: position={best['position_error']:.6g} m, "
                f"orientation={best['orientation_error']:.6g} rad; {message}"
            )
        return IKResult(
            q=best["q"].copy(),
            success=bool(best["accepted"]),
            optimizer_success=bool(solution.success),
            position_error_m=float(best["position_error"]),
            orientation_error_rad=float(best["orientation_error"]),
            fk_position=best["fk_position"].copy(),
            fk_quaternion_xyzw=best["fk_quaternion"].copy(),
            joint_limit_violation=bool(np.any(violation > 1.0e-9)),
            max_joint_limit_violation_rad=float(np.max(violation, initial=0.0)),
            finite=bool(best["finite"]),
            cost=float(solution.cost),
            nfev=int(solution.nfev),
            message=message,
            candidates_evaluated=len(candidates),
        )

    def inverse_position(
        self,
        target_position,
        initial_q=None,
        neutral_q=None,
        additional_seeds=None,
        position_tolerance_m: float = 1.0e-4,
        max_nfev: int = 800,
    ) -> PositionIKResult:
        """Solve joint-limited position-only IK using the same least-squares backend."""
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,) or not np.isfinite(target_position).all():
            raise ValueError("target_position must be finite shape (3,)")
        if position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be > 0")
        warm_start = np.zeros(6) if initial_q is None else np.asarray(initial_q, dtype=float)
        neutral = np.zeros(6) if neutral_q is None else np.asarray(neutral_q, dtype=float)
        if warm_start.shape != (6,) or not np.isfinite(warm_start).all():
            raise ValueError("initial_q must be finite shape (6,)")
        if neutral.shape != (6,) or not np.isfinite(neutral).all():
            raise ValueError("neutral_q must be finite shape (6,)")

        seeds = self._candidate_seeds(warm_start, neutral)
        if additional_seeds is not None:
            additional_seeds = np.asarray(additional_seeds, dtype=float)
            if additional_seeds.ndim != 2 or additional_seeds.shape[1] != 6:
                raise ValueError("additional_seeds must have shape (N,6)")
            if not np.isfinite(additional_seeds).all():
                raise ValueError("additional_seeds contain NaN/Inf")
            for seed in additional_seeds:
                clipped = np.clip(
                    seed,
                    self.joint_lower + 1.0e-10,
                    self.joint_upper - 1.0e-10,
                )
                if not any(np.allclose(clipped, existing) for existing in seeds):
                    seeds.append(clipped)

        candidates = []
        for seed in seeds:
            solution = least_squares(
                self._position_residual,
                seed,
                args=(target_position,),
                bounds=(self.joint_lower, self.joint_upper),
                method="trf",
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=max_nfev,
            )
            q = np.asarray(solution.x, dtype=float)
            fk_position = self.forward(q)[0]
            position_error = float(np.linalg.norm(fk_position - target_position))
            lower_margin = q - self.joint_lower
            upper_margin = self.joint_upper - q
            min_margin = float(np.min(np.concatenate((lower_margin, upper_margin))))
            violation = bool(min_margin < -1.0e-9)
            finite = bool(np.isfinite(q).all() and np.isfinite(fk_position).all())
            accepted = bool(
                solution.success
                and finite
                and position_error <= position_tolerance_m
                and not violation
            )
            candidates.append(
                {
                    "q": q,
                    "solution": solution,
                    "fk_position": fk_position,
                    "position_error": position_error,
                    "margin": min_margin,
                    "violation": violation,
                    "finite": finite,
                    "accepted": accepted,
                    "distance": float(np.linalg.norm(q - warm_start)),
                }
            )

        accepted = [candidate for candidate in candidates if candidate["accepted"]]
        if accepted:
            best = min(accepted, key=lambda candidate: candidate["distance"])
        else:
            best_error = min(candidate["position_error"] for candidate in candidates)
            equivalent = [
                candidate
                for candidate in candidates
                if candidate["position_error"] <= best_error + 1.0e-7
            ]
            best = min(equivalent, key=lambda candidate: candidate["distance"])
        solution = best["solution"]
        message = str(solution.message)
        if not best["accepted"]:
            message = (
                f"position tolerance not met: {best['position_error']:.6g} m; "
                f"{message}"
            )
        return PositionIKResult(
            q=best["q"].copy(),
            success=bool(best["accepted"]),
            optimizer_success=bool(solution.success),
            position_error_m=float(best["position_error"]),
            fk_position=best["fk_position"].copy(),
            joint_limit_violation=bool(best["violation"]),
            min_joint_limit_margin_rad=float(best["margin"]),
            finite=bool(best["finite"]),
            cost=float(solution.cost),
            nfev=int(solution.nfev),
            message=message,
            candidates_evaluated=len(candidates),
        )
