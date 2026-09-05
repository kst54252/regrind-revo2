"""GUI-independent loading and continuity analysis for 12-DoF references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DEFAULT_RB3_JOINT_NAMES = (
    "base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"
)
DEFAULT_REVO2_JOINT_NAMES = (
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)
DEFAULT_REVO2_FOLLOWER_JOINT_NAMES = (
    "right_thumb_distal_joint",
    "right_index_distal_joint",
    "right_middle_distal_joint",
    "right_ring_distal_joint",
    "right_pinky_distal_joint",
)
LEGACY_REVO2_JOINT_NAMES = tuple(f"revo2_joint_{index}" for index in range(6))
MANO21_SEQUENTIAL_TO_REVO_SEMANTIC = np.asarray(
    (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20),
    dtype=np.int64,
)
REVO_SEMANTIC_TO_MANO21_SEQUENTIAL = np.argsort(
    MANO21_SEQUENTIAL_TO_REVO_SEMANTIC
)


def _decode_scalar(value, default: str) -> str:
    if value is None:
        return default
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def _decode_names(value, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return fallback
    result = tuple(
        item.decode() if isinstance(item, bytes) else str(item)
        for item in np.asarray(value).reshape(-1)
    )
    if len(result) != 6:
        raise ValueError(f"expected six joint names, got {result}")
    return result


def _decode_revo2_names(value) -> tuple[str, ...]:
    """Map the legacy positional labels to the real USD joint names.

    Early RB3+Revo2 reference files stored ``revo2_joint_0`` through
    ``revo2_joint_5``.  Their numeric order is the established six-axis Revo2
    order, so retaining those labels would make Isaac Sim look for DOFs that
    do not exist in the robot asset.
    """
    names = _decode_names(value, DEFAULT_REVO2_JOINT_NAMES)
    return DEFAULT_REVO2_JOINT_NAMES if names == LEGACY_REVO2_JOINT_NAMES else names


def _load(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as h5_file:
            return {key: h5_file[key][()] for key in h5_file}
    raise ValueError("trajectory must be .h5, .hdf5, or .npz")


def _first_optional(data: dict, names: tuple[str, ...]):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _to_xyzw(quaternion: np.ndarray, order: str) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    if order == "wxyz":
        quaternion = quaternion[..., (1, 2, 3, 0)]
    elif order != "xyzw":
        raise ValueError(f"unsupported quaternion order {order!r}")
    norms = np.linalg.norm(quaternion, axis=-1)
    if np.any(norms < 1.0e-12):
        raise ValueError("zero-length quaternion in trajectory")
    return quaternion / norms[..., None]


@dataclass(frozen=True)
class ReferenceTrajectory:
    source_path: Path
    rb3_joints: np.ndarray
    revo2_joints: np.ndarray
    revo2_follower_joints: np.ndarray | None
    revo2_joint_drive_target: np.ndarray | None
    revo2_fingertip_pos: np.ndarray | None
    reference_joints: np.ndarray
    rb3_joint_names: tuple[str, ...]
    revo2_joint_names: tuple[str, ...]
    revo2_follower_joint_names: tuple[str, ...]
    wrist_pos: np.ndarray | None
    wrist_quat_xyzw: np.ndarray | None
    object_pos: np.ndarray | None
    object_quat_xyzw: np.ndarray | None
    mano_joint_world: np.ndarray | None
    fps: float
    dt: float

    @property
    def frames(self) -> int:
        return len(self.reference_joints)


@dataclass(frozen=True)
class ContinuityAnalysis:
    dt: float
    joint_step: np.ndarray
    joint_step_norm: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    max_joint_step_norm: float
    max_abs_velocity_per_joint: np.ndarray
    max_abs_acceleration_per_joint: np.ndarray
    discontinuity_frames: np.ndarray
    nonfinite_frames: np.ndarray


def load_reference_trajectory(
    path: str | Path,
    dt_override: float | None = None,
) -> ReferenceTrajectory:
    source_path = Path(path).expanduser().resolve()
    data = _load(source_path)
    rb3 = _first_optional(data, ("rb3_joints",))
    revo2 = _first_optional(data, ("revo2_joints", "robot_joints"))
    combined = _first_optional(data, ("reference_joints",))
    if combined is not None:
        combined = np.asarray(combined, dtype=float)
        if combined.ndim != 2 or combined.shape[1] != 12:
            raise ValueError(f"reference_joints must have shape (T,12), got {combined.shape}")
        if rb3 is None:
            rb3 = combined[:, :6]
        if revo2 is None:
            revo2 = combined[:, 6:]
    if rb3 is None or revo2 is None:
        raise KeyError("need rb3_joints + revo2_joints, or reference_joints")
    rb3 = np.asarray(rb3, dtype=float)
    revo2 = np.asarray(revo2, dtype=float)
    if rb3.ndim != 2 or rb3.shape[1] != 6:
        raise ValueError(f"rb3_joints must have shape (T,6), got {rb3.shape}")
    if revo2.shape != rb3.shape:
        raise ValueError(f"revo2_joints must match rb3_joints, got {revo2.shape}")
    expected_combined = np.concatenate((rb3, revo2), axis=1)
    if combined is not None and not np.allclose(
        combined, expected_combined, rtol=0.0, atol=1.0e-12, equal_nan=True
    ):
        raise ValueError("reference_joints disagrees with rb3_joints/revo2_joints")
    if combined is None:
        combined = expected_combined

    revo2_follower_joints = _first_optional(data, ("revo2_follower_joints",))
    if revo2_follower_joints is not None:
        revo2_follower_joints = np.asarray(revo2_follower_joints, dtype=float)
        if revo2_follower_joints.shape != (len(rb3), 5):
            raise ValueError(
                "revo2_follower_joints must have shape "
                f"{(len(rb3), 5)}, got {revo2_follower_joints.shape}"
            )
    revo2_joint_drive_target = _first_optional(
        data, ("revo2_joint_drive_target",)
    )
    if revo2_joint_drive_target is not None:
        revo2_joint_drive_target = np.asarray(
            revo2_joint_drive_target, dtype=float
        )
        if revo2_joint_drive_target.shape != (len(rb3), 6):
            raise ValueError(
                "revo2_joint_drive_target must have shape "
                f"{(len(rb3), 6)}, got {revo2_joint_drive_target.shape}"
            )
    if any(
        value is not None and not np.isfinite(value).all()
        for value in (revo2_follower_joints, revo2_joint_drive_target)
    ):
        raise ValueError("Revo2 follower state/drive target contains NaN/Inf")

    follower_name_value = data.get("revo2_follower_joint_names")
    if follower_name_value is None:
        follower_names = DEFAULT_REVO2_FOLLOWER_JOINT_NAMES
    else:
        follower_names = tuple(
            item.decode() if isinstance(item, bytes) else str(item)
            for item in np.asarray(follower_name_value).reshape(-1)
        )
        if follower_names != DEFAULT_REVO2_FOLLOWER_JOINT_NAMES:
            raise ValueError(
                "unexpected Revo2 follower joint order: "
                f"expected={DEFAULT_REVO2_FOLLOWER_JOINT_NAMES}, got={follower_names}"
            )

    quaternion_order = _decode_scalar(
        data.get("quat_convention", data.get("quaternion_order")), "xyzw"
    ).lower()
    wrist_pos = _first_optional(
        data, ("target_wrist_pos", "wrist_pos", "wrist_pos_world")
    )
    wrist_quat = _first_optional(
        data, ("target_wrist_quat", "wrist_quat", "wrist_quat_world")
    )
    if (wrist_pos is None) != (wrist_quat is None):
        raise ValueError("wrist position and quaternion must either both exist or both be absent")
    if wrist_pos is not None:
        wrist_pos = np.asarray(wrist_pos, dtype=float)
        wrist_quat = np.asarray(wrist_quat, dtype=float)
        if wrist_pos.shape != (len(rb3), 3) or wrist_quat.shape != (len(rb3), 4):
            raise ValueError("wrist reference must have shapes (T,3) and (T,4)")
        wrist_quat = _to_xyzw(wrist_quat, quaternion_order)

    object_pos = _first_optional(data, ("object_pos", "object_pos_world"))
    object_quat = _first_optional(data, ("object_quat", "object_quat_world"))
    if object_pos is None and object_quat is not None:
        raise ValueError("object quaternion exists but object position is absent")
    if object_pos is not None:
        object_pos = np.asarray(object_pos, dtype=float)
        if object_pos.shape != (len(rb3), 3):
            raise ValueError(f"object_pos must have shape (T,3), got {object_pos.shape}")
        if object_quat is None:
            object_quat = np.zeros((len(rb3), 4), dtype=float)
            object_quat[:, 3] = 1.0
        else:
            object_quat = np.asarray(object_quat, dtype=float)
            if object_quat.shape != (len(rb3), 4):
                raise ValueError(
                    f"object_quat must have shape (T,4), got {object_quat.shape}"
                )
            object_quat = _to_xyzw(object_quat, quaternion_order)

    # New reference files carry the pre-retargeting MANO skeleton directly.
    # Older files remain usable by following their recorded world-trajectory
    # source, avoiding an expensive rerun of RB3 IK just for visualization.
    mano_source_data = data
    mano_joint_world = _first_optional(
        data,
        (
            "mano_joint_world_mano21",
            "mano_joint_world",
            "mano_joint_coords_world",
            "mano_joint_coords",
        ),
    )
    if mano_joint_world is None and "source_retargeting_file" in data:
        retargeting_path = Path(
            _decode_scalar(data["source_retargeting_file"], "")
        ).expanduser()
        if not retargeting_path.is_absolute():
            retargeting_path = source_path.parent / retargeting_path
        if retargeting_path.is_file():
            retargeting_data = _load(retargeting_path)
            mano_joint_world = _first_optional(
                retargeting_data,
                (
                    "mano_joint_world_mano21",
                    "mano_joint_world",
                    "mano_joint_coords_world",
                    "mano_joint_coords",
                    "human_hand_keypoints",
                ),
            )
            if mano_joint_world is not None:
                mano_source_data = retargeting_data
    if mano_joint_world is not None:
        mano_joint_world = np.asarray(mano_joint_world, dtype=float)
        expected_mano_shape = (len(rb3), 21, 3)
        if mano_joint_world.shape != expected_mano_shape:
            raise ValueError(
                "pre-retargeting MANO skeleton must have shape "
                f"{expected_mano_shape}, got {mano_joint_world.shape}"
            )
        if not np.isfinite(mano_joint_world).all():
            raise ValueError("pre-retargeting MANO skeleton contains NaN/Inf")
        mano_order = _decode_scalar(
            mano_source_data.get("mano_joint_order"),
            "mano21_sequential_thumb_index_middle_ring_little",
        ).lower()
        if "revo_semantic" in mano_order or "revo2_semantic" in mano_order:
            mano_joint_world = mano_joint_world[
                :, REVO_SEMANTIC_TO_MANO21_SEQUENTIAL
            ]
        elif "sequential" not in mano_order:
            raise ValueError(f"unsupported mano_joint_order {mano_order!r}")

    fingertips = _first_optional(data, ("revo2_fingertip_pos",))
    if fingertips is not None and (
        fingertips.shape != (len(rb3), 5, 3) or not np.isfinite(fingertips).all()
    ):
        raise ValueError("revo2_fingertip_pos must be finite and have shape (T,5,3)")
    fps = float(np.asarray(data.get("fps", 30.0)).item())
    if fps <= 0:
        raise ValueError("stored FPS must be > 0")
    dt = 1.0 / fps if dt_override is None else float(dt_override)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and > 0")
    return ReferenceTrajectory(
        source_path=source_path,
        rb3_joints=rb3,
        revo2_joints=revo2,
        revo2_follower_joints=revo2_follower_joints,
        revo2_joint_drive_target=revo2_joint_drive_target,
        revo2_fingertip_pos=fingertips,
        reference_joints=combined,
        rb3_joint_names=_decode_names(data.get("rb3_joint_names"), DEFAULT_RB3_JOINT_NAMES),
        revo2_joint_names=_decode_revo2_names(
            data.get("revo2_joint_names", data.get("actuated_joint_names"))
        ),
        revo2_follower_joint_names=follower_names,
        wrist_pos=wrist_pos,
        wrist_quat_xyzw=wrist_quat,
        object_pos=object_pos,
        object_quat_xyzw=object_quat,
        mano_joint_world=mano_joint_world,
        fps=fps,
        dt=dt,
    )


def analyze_continuity(
    trajectory: ReferenceTrajectory,
    discontinuity_step_threshold_rad: float = 0.5,
) -> ContinuityAnalysis:
    if discontinuity_step_threshold_rad <= 0:
        raise ValueError("discontinuity threshold must be > 0")
    q = trajectory.reference_joints
    finite_rows = np.isfinite(q).all(axis=1)
    nonfinite_frames = np.flatnonzero(~finite_rows)
    step = np.diff(q, axis=0)
    step_norm = np.linalg.norm(step, axis=1)
    velocity = step / trajectory.dt
    acceleration = np.diff(velocity, axis=0) / trajectory.dt
    discontinuity_frames = np.flatnonzero(
        step_norm > discontinuity_step_threshold_rad
    ) + 1
    max_velocity = (
        np.nanmax(np.abs(velocity), axis=0) if len(velocity) else np.zeros(12)
    )
    max_acceleration = (
        np.nanmax(np.abs(acceleration), axis=0)
        if len(acceleration)
        else np.zeros(12)
    )
    return ContinuityAnalysis(
        dt=trajectory.dt,
        joint_step=step,
        joint_step_norm=step_norm,
        velocity=velocity,
        acceleration=acceleration,
        max_joint_step_norm=float(np.nanmax(step_norm)) if len(step_norm) else 0.0,
        max_abs_velocity_per_joint=max_velocity,
        max_abs_acceleration_per_joint=max_acceleration,
        discontinuity_frames=discontinuity_frames,
        nonfinite_frames=nonfinite_frames,
    )


def joint_limit_violations(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance_rad: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if q.ndim != 2 or lower.shape != (q.shape[1],) or upper.shape != lower.shape:
        raise ValueError("joint limit shapes do not match trajectory")
    mask = (q < lower[None] - tolerance_rad) | (q > upper[None] + tolerance_rad)
    return mask, np.flatnonzero(mask.any(axis=1))


def quaternion_angular_error_xyzw(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    actual = actual / np.linalg.norm(actual)
    reference = reference / np.linalg.norm(reference)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(actual, reference)), 0.0, 1.0)))
