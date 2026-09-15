"""Transport-independent contracts. Metres, radians, seconds, world XYZW.

Times must be mapped to the host's monotonic clock by the acquisition adapter,
not stamped on old cached data at read time. Joint arrays always carry names.
"""
from dataclasses import dataclass

import numpy as np


def vector(value, size, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected finite shape ({size},)")
    result.setflags(write=False)
    return result


def timestamp(value, label):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{label}: expected nonnegative monotonic seconds")


def joint_order(names):
    names = tuple(names)
    if len(names) != 12 or len(set(names)) != 12 or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Exactly 12 unique arm + leader joint names are required")
    return names


def reorder(values, source_names, target_names):
    source_names, target_names = joint_order(source_names), joint_order(target_names)
    if set(source_names) != set(target_names):
        raise ValueError(f"Joint name mismatch: missing={set(target_names)-set(source_names)}, "
                         f"extra={set(source_names)-set(target_names)}")
    values = vector(values, 12, "joint values")
    return values[[source_names.index(name) for name in target_names]]


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    frame: str

    def __post_init__(self):
        object.__setattr__(self, "position", vector(self.position, 3, "position [m]"))
        q = vector(self.quaternion_xyzw, 4, "quaternion XYZW")
        if abs(np.linalg.norm(q) - 1) > 1e-5:
            raise ValueError("Quaternion must be unit XYZW; convert at the acquisition boundary")
        if not self.frame:
            raise ValueError("A calibrated world frame name is required")
        object.__setattr__(self, "quaternion_xyzw", q)


@dataclass(frozen=True)
class RobotState:
    sequence: int
    sample_time: float
    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    wrist: Pose
    wrist_source: str  # e.g. physx_body_pose, encoder_fk (NOT a desired pose)
    object_pose: Pose | None = None
    object_sample_time: float | None = None
    ready: bool = True
    protective_stop: bool = False

    def __post_init__(self):
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("State sequence must be a nonnegative integer")
        timestamp(self.sample_time, "state time")
        object.__setattr__(self, "joint_names", joint_order(self.joint_names))
        object.__setattr__(self, "position", vector(self.position, 12, "actual q [rad]"))
        object.__setattr__(self, "velocity", vector(self.velocity, 12, "actual dq [rad/s]"))
        if self.wrist_source not in {"physx_body_pose", "encoder_fk", "external_tracking", "mock_fk"}:
            raise ValueError("Explicit measured-state provenance is required")
        if (self.object_pose is None) != (self.object_sample_time is None):
            raise ValueError("Object pose and acquisition time must be supplied together")
        if self.object_pose is not None:
            timestamp(self.object_sample_time, "object acquisition time")
            if self.object_pose.frame != self.wrist.frame:
                raise ValueError("Wrist and object must share the calibrated world frame")


@dataclass(frozen=True)
class DecodedTarget:
    """AFTER the existing reference/residual decoder, never a raw policy action."""
    wrist: Pose
    hand_position: np.ndarray
    hand_names: tuple[str, ...]
    valid_until: float
    reference_frame: int

    def __post_init__(self):
        object.__setattr__(self, "hand_position", vector(self.hand_position, 6, "leader target [rad]"))
        object.__setattr__(self, "hand_names", tuple(self.hand_names))
        if len(self.hand_names) != 6 or len(set(self.hand_names)) != 6:
            raise ValueError("Six unique leader names required; do not send mimic followers")
        timestamp(self.valid_until, "target deadline")
        if not isinstance(self.reference_frame, int) or self.reference_frame < 0:
            raise ValueError("Invalid reference frame")


@dataclass(frozen=True)
class JointCommand:
    sequence: int
    state_sequence: int
    created_at: float
    valid_until: float
    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity_target: np.ndarray  # simulator/driver target, NOT position-path speed
    reference_frame: int

    def __post_init__(self):
        object.__setattr__(self, "joint_names", joint_order(self.joint_names))
        object.__setattr__(self, "position", vector(self.position, 12, "q command [rad]"))
        object.__setattr__(self, "velocity_target", vector(self.velocity_target, 12, "dq target [rad/s]"))
        timestamp(self.created_at, "command time")
        timestamp(self.valid_until, "command deadline")
        if self.valid_until <= self.created_at:
            raise ValueError("Command already expired")
