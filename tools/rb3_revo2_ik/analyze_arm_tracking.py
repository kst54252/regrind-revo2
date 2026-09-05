#!/usr/bin/env python3
"""Analyze RB3 target-to-measured response captured during online policy play.

The telemetry is intentionally independent of Isaac Sim so that the same
analysis can be used for simulated and, later, real-robot joint logs.
Quaternion arrays use Isaac's ``xyzw`` convention.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DelayEstimate:
    steps: int
    seconds: float
    rmse: float


def _as_2d(name: str, value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (T,{width}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def quaternion_error_xyzw(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return shortest angular distance between quaternion rows in radians."""

    actual = _as_2d("actual_wrist_quat_xyzw", actual, 4)
    target = _as_2d("target_wrist_quat_xyzw", target, 4)
    if actual.shape != target.shape:
        raise ValueError("actual/target quaternion arrays must have identical shapes")
    actual = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    dot = np.abs(np.sum(actual * target, axis=1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def estimate_integer_delay(
    target: np.ndarray,
    actual: np.ndarray,
    dt: float,
    max_lag_steps: int = 8,
) -> DelayEstimate:
    """Find the causal integer lag minimizing normalized joint-velocity RMSE.

    Position trajectories can have large static offsets and long holds.  Their
    first differences isolate response timing.  Each moving joint is normalized
    by its target velocity RMS so one high-range joint cannot dominate the fit.
    """

    target = _as_2d("target_rb3_joints", target, 6)
    actual = _as_2d("actual_rb3_joints", actual, 6)
    if target.shape != actual.shape:
        raise ValueError("target/actual RB3 joint arrays must have identical shapes")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    if max_lag_steps < 0:
        raise ValueError("max_lag_steps must be non-negative")
    if len(target) < 4:
        return DelayEstimate(steps=0, seconds=0.0, rmse=float("nan"))

    target_velocity = np.diff(target, axis=0)
    actual_velocity = np.diff(actual, axis=0)
    scale = np.sqrt(np.mean(target_velocity**2, axis=0))
    moving = scale > 1.0e-8
    if not np.any(moving):
        return DelayEstimate(steps=0, seconds=0.0, rmse=0.0)

    best_lag = 0
    best_rmse = float("inf")
    max_lag = min(max_lag_steps, len(target_velocity) - 2)
    for lag in range(max_lag + 1):
        count = len(target_velocity) - lag
        if count < 2:
            continue
        target_slice = target_velocity[:count, moving] / scale[moving]
        actual_slice = actual_velocity[lag : lag + count, moving] / scale[moving]
        rmse = float(np.sqrt(np.mean((actual_slice - target_slice) ** 2)))
        if rmse < best_rmse:
            best_lag = lag
            best_rmse = rmse
    return DelayEstimate(steps=best_lag, seconds=best_lag * dt, rmse=best_rmse)


def analyze_arrays(data: dict[str, np.ndarray], max_lag_steps: int = 8) -> dict[str, object]:
    target_q = _as_2d("target_rb3_joints", data["target_rb3_joints"], 6)
    actual_q = _as_2d("actual_rb3_joints", data["actual_rb3_joints"], 6)
    target_pos = _as_2d("target_wrist_pos", data["target_wrist_pos"], 3)
    actual_pos = _as_2d("actual_wrist_pos", data["actual_wrist_pos"], 3)
    target_quat = _as_2d("target_wrist_quat_xyzw", data["target_wrist_quat_xyzw"], 4)
    actual_quat = _as_2d("actual_wrist_quat_xyzw", data["actual_wrist_quat_xyzw"], 4)
    lengths = {len(array) for array in (target_q, actual_q, target_pos, actual_pos, target_quat, actual_quat)}
    if len(lengths) != 1:
        raise ValueError(f"telemetry arrays have inconsistent lengths: {sorted(lengths)}")
    dt = float(np.asarray(data["dt"]).reshape(()))

    delay = estimate_integer_delay(target_q, actual_q, dt, max_lag_steps)
    q_error = actual_q - target_q
    pos_error = np.linalg.norm(actual_pos - target_pos, axis=1)
    ori_error = quaternion_error_xyzw(actual_quat, target_quat)
    q_norm = np.linalg.norm(q_error, axis=1)

    lag = delay.steps
    aligned_q_error = actual_q[lag:] - target_q[: len(target_q) - lag] if lag else q_error
    return {
        "frames": len(target_q),
        "dt": dt,
        "delay": delay,
        "joint_rmse_rad": np.sqrt(np.mean(q_error**2, axis=0)),
        "joint_max_abs_rad": np.max(np.abs(q_error), axis=0),
        "joint_error_norm_mean_rad": float(np.mean(q_norm)),
        "joint_error_norm_max_rad": float(np.max(q_norm)),
        "delay_aligned_joint_rmse_rad": np.sqrt(np.mean(aligned_q_error**2, axis=0)),
        "wrist_position_mean_m": float(np.mean(pos_error)),
        "wrist_position_max_m": float(np.max(pos_error)),
        "wrist_orientation_mean_rad": float(np.mean(ori_error)),
        "wrist_orientation_max_rad": float(np.max(ori_error)),
    }


def print_report(report: dict[str, object], source: Path | None = None) -> None:
    delay = report["delay"]
    assert isinstance(delay, DelayEstimate)
    if source is not None:
        print(f"[RB3 tracking telemetry] {source}")
    print(f"  samples / control dt:       {report['frames']} / {report['dt']:.8g} s")
    print(
        "  estimated response delay:   "
        f"{delay.steps} control steps = {delay.seconds * 1000.0:.3f} ms "
        f"(velocity-fit RMSE {delay.rmse:.6g})"
    )
    print(
        "  arm q error norm mean/max:  "
        f"{report['joint_error_norm_mean_rad']:.8g} / "
        f"{report['joint_error_norm_max_rad']:.8g} rad"
    )
    print(f"  per-joint RMSE [rad]:       {np.asarray(report['joint_rmse_rad']).round(6).tolist()}")
    print(f"  per-joint max abs [rad]:    {np.asarray(report['joint_max_abs_rad']).round(6).tolist()}")
    print(
        "  lag-aligned RMSE [rad]:     "
        f"{np.asarray(report['delay_aligned_joint_rmse_rad']).round(6).tolist()}"
    )
    print(
        "  wrist position mean/max:    "
        f"{report['wrist_position_mean_m']:.8g} / {report['wrist_position_max_m']:.8g} m"
    )
    print(
        "  wrist orientation mean/max: "
        f"{report['wrist_orientation_mean_rad']:.8g} / "
        f"{report['wrist_orientation_max_rad']:.8g} rad"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path, help="NPZ written by play.py --arm-tracking-path")
    parser.add_argument("--max-lag-steps", type=int, default=8)
    args = parser.parse_args()
    source = args.telemetry.expanduser().resolve()
    with np.load(source, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    report = analyze_arrays(data, args.max_lag_steps)
    scale_keys = ("rb3_stiffness_scale", "rb3_damping_scale", "rb3_effort_scale")
    if all(key in data for key in scale_keys):
        print(
            "[RB3 actuator scales] "
            + ", ".join(
                f"{key.removeprefix('rb3_').removesuffix('_scale')}="
                f"{float(np.asarray(data[key]).reshape(())):g}"
                for key in scale_keys
            )
        )
    print_report(report, source)


if __name__ == "__main__":
    main()
