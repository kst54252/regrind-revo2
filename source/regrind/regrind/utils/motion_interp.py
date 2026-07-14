import numpy as np
import math
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy.spatial.transform import Rotation, Slerp
from typing import Literal


def rescale_motion(
    # positions: np.ndarray,          # (N, 3)
    # orientations_quat: np.ndarray,  # (N, 4) in xyzw format
    # joint_angles: np.ndarray,       # (N, num_joints)
    motion_data: np.ndarray,
    motion_type: Literal["position", "orientation", "joint"],
    speed_factor: float,
    dt: float,
    target_dt: float | None = None,
    interp_mode: Literal["spline", "zoh"] = "spline",
):
    """
    Rescale motion and compute velocities analytically from spline derivatives.

    Args:
        motion_data: (N, 3) or (N, 4) or (N, num_joints)
            - (N, 3): positions
            - (N, 4): orientations
            - (N, num_joints): joint positions
        motion_type: "position", "orientation", "joint"
        speed_factor: float
        dt: float | None
        target_dt: float | None
        interp_mode: "spline" or "zoh" (zero-order hold)

    Returns:
        new_positions, new_linear_vel,       # (M, 3), (M, 3)
        new_orientations, new_angular_vel,   # (M, 4), (M, 3)
        new_joints, new_joint_vel,           # (M, J), (M, J)
    """

    N = len(motion_data)

    # Timing
    original_duration = (N - 1) * dt
    new_duration = original_duration / speed_factor
    if target_dt is None:
        target_dt = dt

    # new_N = math.floor(new_duration / target_dt) + 1
    # Use rounding to avoid off-by-one from float precision.
    new_N = int(round(new_duration / target_dt)) + 1

    old_times = np.linspace(0, new_duration, N, endpoint=True)
    new_times = np.linspace(0, new_duration, new_N, endpoint=True)
    # Effective dt between original frames after time-rescaling
    rescaled_dt = dt / speed_factor

    if motion_type == "position":
        if interp_mode == "zoh":
            indices = np.searchsorted(old_times, new_times, side="right") - 1
            indices = np.clip(indices, 0, N - 1)
            new_positions = motion_data[indices]
            orig_linear_vel = np.zeros_like(motion_data)
            if motion_data.shape[0] > 1:
                orig_linear_vel[1:] = np.diff(motion_data, axis=0) / rescaled_dt
            new_linear_vel = orig_linear_vel[indices]
            return new_positions, new_linear_vel
        pos_interp = Akima1DInterpolator(old_times, motion_data, method='makima')
        new_positions = pos_interp(new_times)
        new_linear_vel = pos_interp(new_times, nu=1)  # 1st derivative
        # velocity at the first timestep should be zero
        # assert np.allclose(new_linear_vel[0], 0.0), "Velocity at the first timestep should be zero"
        new_linear_vel[0] = 0.0
        return new_positions, new_linear_vel

    elif motion_type == "orientation":
        assert motion_data.shape[1] == 4, "Orientations must be (N, 4)"
        if interp_mode == "zoh":
            indices = np.searchsorted(old_times, new_times, side="right") - 1
            indices = np.clip(indices, 0, N - 1)
            new_orientations = motion_data[indices]
            orig_rotations = Rotation.from_quat(motion_data, scalar_first=True)
            orig_angular_vel = compute_angular_velocity(orig_rotations, rescaled_dt)
            new_angular_vel = orig_angular_vel[indices]
            return new_orientations, new_angular_vel
        rotations = Rotation.from_quat(motion_data, scalar_first=True)
        slerp = Slerp(old_times, rotations)
        new_rotations = slerp(new_times)
        new_orientations = new_rotations.as_quat(scalar_first=True)
        new_angular_vel = compute_angular_velocity(new_rotations, target_dt)
        # velocity at the first timestep should be zero
        # assert np.allclose(new_angular_vel[0], 0.0), "Velocity at the first timestep should be zero"
        return new_orientations, new_angular_vel

    elif motion_type == "joint":
        if interp_mode == "zoh":
            indices = np.searchsorted(old_times, new_times, side="right") - 1
            indices = np.clip(indices, 0, N - 1)
            new_joints = motion_data[indices]
            orig_joint_vel = np.zeros_like(motion_data)
            if motion_data.shape[0] > 1:
                orig_joint_vel[1:] = np.diff(motion_data, axis=0) / rescaled_dt
            new_joint_vel = orig_joint_vel[indices]
            return new_joints, new_joint_vel
        joint_interp = PchipInterpolator(old_times, motion_data)
        new_joints = joint_interp(new_times)
        new_joint_vel = joint_interp(new_times, nu=1)  # 1st derivative
        # velocity at the first timestep should be zero
        # assert np.allclose(new_joint_vel[0], 0.0), "Velocity at the first timestep should be zero"
        new_joint_vel[0] = 0.0
        return new_joints, new_joint_vel

    else:
        raise ValueError(f"Invalid motion type: {motion_type}")


def compute_angular_velocity(rotations: Rotation, dt: float) -> np.ndarray:
    """
    Compute angular velocity from a sequence of rotations.

    ω = rotation_vector(R_{i+1} * R_i^{-1}) / dt

    Returns: (N, 3) angular velocity in rad/s
    """
    N = len(rotations)
    angular_vel = np.zeros((N, 3))

    for i in range(1, N):
        # Relative rotation from i-1 to i
        r_rel = rotations[i] * rotations[i - 1].inv()
        # Convert to rotation vector (axis * angle)
        angular_vel[i] = r_rel.as_rotvec() / dt

    return angular_vel
