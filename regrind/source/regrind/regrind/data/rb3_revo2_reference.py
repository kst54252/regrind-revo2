"""Strict loader for RB3-730 + Revo2 12-DoF reference trajectories.

This module is deliberately independent of Isaac Sim so trajectory files can
be checked in unit tests and preprocessing jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


RB3_JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
REVO2_JOINT_NAMES = (
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)
REFERENCE_JOINT_NAMES = RB3_JOINT_NAMES + REVO2_JOINT_NAMES
MANO21_SEQUENTIAL_TO_REVO_SEMANTIC = np.asarray(
    (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20),
    dtype=np.int64,
)


def _decode_scalar(value, default: str) -> str:
    if value is None:
        return default
    value = np.asarray(value).item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode_names(value, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return fallback
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(value).reshape(-1)
    )


def _read(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as h5_file:
            return {name: h5_file[name][()] for name in h5_file}
    raise ValueError(f"reference must be .h5, .hdf5, or .npz: {path}")


def _require_shape(name: str, value: np.ndarray, shape_tail: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    expected_rank = 1 + len(shape_tail)
    if value.ndim != expected_rank or value.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (T,{','.join(map(str, shape_tail))}), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


@dataclass(frozen=True)
class RB3Revo2Reference:
    """Validated reference arrays in the exact 12-action order."""

    path: Path
    reference_joints: np.ndarray
    rb3_joints: np.ndarray
    revo2_joints: np.ndarray
    object_pos: np.ndarray
    object_quat_xyzw: np.ndarray
    wrist_pos: np.ndarray
    wrist_quat_xyzw: np.ndarray
    mano_joint_world_semantic: np.ndarray | None
    fps: float
    joint_names: tuple[str, ...]
    phase_total_frames: int

    @property
    def frames(self) -> int:
        return int(self.reference_joints.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


def load_rb3_revo2_reference(
    path: str | Path,
    *,
    require_success: bool = True,
) -> RB3Revo2Reference:
    """Load and validate a final RB3+Revo2 reference trajectory.

    Quaternion input follows the file's ``quat_convention`` metadata and is
    returned as XYZW, matching this project's Isaac Lab 3.x code path.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RB3+Revo2 reference not found: {path}")
    data = _read(path)

    if "rb3_joints" not in data or "revo2_joints" not in data:
        raise KeyError("reference requires rb3_joints and revo2_joints")
    rb3 = _require_shape("rb3_joints", data["rb3_joints"], (6,))
    revo2 = _require_shape("revo2_joints", data["revo2_joints"], (6,))
    combined = np.concatenate((rb3, revo2), axis=1)
    if "reference_joints" in data:
        stored = _require_shape("reference_joints", data["reference_joints"], (12,))
        if not np.allclose(stored, combined, atol=1.0e-10, rtol=0.0):
            raise ValueError("reference_joints does not equal [rb3_joints, revo2_joints]")
        combined = stored

    object_pos = _require_shape("object_pos", data["object_pos"], (3,))
    object_quat = _require_shape("object_quat", data["object_quat"], (4,))
    wrist_pos_key = "wrist_pos" if "wrist_pos" in data else "target_wrist_pos"
    wrist_quat_key = "wrist_quat" if "wrist_quat" in data else "target_wrist_quat"
    if wrist_pos_key not in data or wrist_quat_key not in data:
        raise KeyError("reference requires wrist_pos/wrist_quat (or target_wrist_pos/target_wrist_quat)")
    wrist_pos = _require_shape(wrist_pos_key, data[wrist_pos_key], (3,))
    wrist_quat = _require_shape(wrist_quat_key, data[wrist_quat_key], (4,))
    frame_counts = {
        rb3.shape[0],
        revo2.shape[0],
        object_pos.shape[0],
        object_quat.shape[0],
        wrist_pos.shape[0],
        wrist_quat.shape[0],
    }
    if len(frame_counts) != 1 or next(iter(frame_counts)) < 2:
        raise ValueError(f"reference arrays must share T>=2, got frame counts {sorted(frame_counts)}")
    frames = next(iter(frame_counts))

    mano_joint_world = None
    mano_key = next(
        (key for key in ("mano_joint_world", "mano_joint_world_mano21") if key in data),
        None,
    )
    if mano_key is not None:
        mano_joint_world = _require_shape(mano_key, data[mano_key], (21, 3))
        if mano_joint_world.shape[0] != frames:
            raise ValueError(
                f"{mano_key} must share reference frame count {frames}, got {mano_joint_world.shape[0]}"
            )
        mano_order = _decode_scalar(
            data.get("mano_joint_order"),
            "mano21_sequential_thumb_index_middle_ring_little",
        ).lower()
        if "sequential" in mano_order:
            mano_joint_world = mano_joint_world[:, MANO21_SEQUENTIAL_TO_REVO_SEMANTIC]

    quat_order = _decode_scalar(data.get("quat_convention"), "wxyz").lower()
    if quat_order == "wxyz":
        object_quat = object_quat[:, (1, 2, 3, 0)]
        wrist_quat = wrist_quat[:, (1, 2, 3, 0)]
    elif quat_order != "xyzw":
        raise ValueError(f"unsupported quat_convention {quat_order!r}")
    norms = np.linalg.norm(object_quat, axis=1)
    if np.any(norms < 1.0e-12):
        raise ValueError("object_quat contains a zero quaternion")
    object_quat = object_quat / norms[:, None]
    wrist_norms = np.linalg.norm(wrist_quat, axis=1)
    if np.any(wrist_norms < 1.0e-12):
        raise ValueError("wrist_quat contains a zero quaternion")
    wrist_quat = wrist_quat / wrist_norms[:, None]

    rb3_names = _decode_names(data.get("rb3_joint_names"), RB3_JOINT_NAMES)
    revo2_names = _decode_names(data.get("revo2_joint_names"), REVO2_JOINT_NAMES)
    joint_names = rb3_names + revo2_names
    if joint_names != REFERENCE_JOINT_NAMES:
        raise ValueError(
            "reference joint order does not match the RL action order; "
            f"expected={REFERENCE_JOINT_NAMES}, got={joint_names}"
        )

    fps = float(np.asarray(data.get("fps", 30.0)).item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    phase_total_frames = int(np.asarray(data.get("phase_total_frames", frames)).item())
    if phase_total_frames < frames:
        raise ValueError(
            f"phase_total_frames must be >= stored frames ({frames}), got {phase_total_frames}"
        )
    if require_success:
        if "ik_success" in data and not np.asarray(data["ik_success"], dtype=bool).all():
            failed = np.flatnonzero(~np.asarray(data["ik_success"], dtype=bool)).tolist()
            raise ValueError(f"reference contains failed IK frames: {failed}")
        if "joint_limit_violation" in data and np.asarray(
            data["joint_limit_violation"], dtype=bool
        ).any():
            failed = np.flatnonzero(np.asarray(data["joint_limit_violation"], dtype=bool)).tolist()
            raise ValueError(f"reference contains joint-limit violations: {failed}")

    return RB3Revo2Reference(
        path=path,
        reference_joints=combined,
        rb3_joints=rb3,
        revo2_joints=revo2,
        object_pos=object_pos,
        object_quat_xyzw=object_quat,
        wrist_pos=wrist_pos,
        wrist_quat_xyzw=wrist_quat,
        mano_joint_world_semantic=mano_joint_world,
        fps=fps,
        joint_names=joint_names,
        phase_total_frames=phase_total_frames,
    )
