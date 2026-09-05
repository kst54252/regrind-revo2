#!/usr/bin/env python3
"""Validate a tabletop XY rectangle using complete strict RB3 pose IK sequences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from rb3_kinematics import RB3730Kinematics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "outputs"
    / "isaac"
    / "dexycb"
    / "20200709_143747_left"
    / "rb3_revo2_reference.h5"
)
WORKCELL_CONFIG = PROJECT_ROOT / "config" / "workcell" / "rb3_revo2_table.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--x-range", nargs=2, type=float, default=(0.40, 0.50))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-0.20, 0.20))
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--orientation-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")

    with WORKCELL_CONFIG.open("r", encoding="utf-8") as config_file:
        layout = json.load(config_file)
    base_position = tuple(layout["robot_mount"]["position"])
    table = layout["table"]
    table_x = (
        table["center_xy"][0] - table["size_xy"][0] / 2.0,
        table["center_xy"][0] + table["size_xy"][0] / 2.0,
    )
    table_y = (
        table["center_xy"][1] - table["size_xy"][1] / 2.0,
        table["center_xy"][1] + table["size_xy"][1] / 2.0,
    )
    if args.x_range[0] < table_x[0] or args.x_range[1] > table_x[1]:
        raise ValueError(f"X range {args.x_range} is outside table bounds {table_x}")
    if args.y_range[0] < table_y[0] or args.y_range[1] > table_y[1]:
        raise ValueError(f"Y range {args.y_range} is outside table bounds {table_y}")

    reference_path = args.reference.expanduser().resolve()
    with h5py.File(reference_path, "r") as reference:
        wrist_pos = np.asarray(reference["wrist_pos"], dtype=float)
        wrist_quat = np.asarray(reference["wrist_quat"], dtype=float)
        object_pos = np.asarray(reference["object_pos"], dtype=float)
        neutral_q = np.asarray(reference["rb3_joints"], dtype=float)[0]
    kinematics = RB3730Kinematics(base_position=base_position)

    rows = []
    for y in np.linspace(*args.y_range, args.grid_size):
        symbols = []
        for x in np.linspace(*args.x_range, args.grid_size):
            delta = np.asarray((x - object_pos[0, 0], y - object_pos[0, 1], 0.0))
            warm_q = neutral_q.copy()
            successes = []
            position_errors = []
            orientation_errors = []
            for target_position, target_quaternion in zip(wrist_pos + delta, wrist_quat):
                result = kinematics.inverse(
                    target_position,
                    target_quaternion,
                    initial_q=warm_q,
                    neutral_q=neutral_q,
                    position_tolerance_m=args.position_tolerance,
                    orientation_tolerance_rad=args.orientation_tolerance,
                    max_nfev=args.max_nfev,
                )
                successes.append(result.success)
                position_errors.append(result.position_error_m)
                orientation_errors.append(result.orientation_error_rad)
                if result.finite:
                    warm_q = result.q
            full_success = bool(all(successes))
            symbols.append("O" if full_success else "X")
            rows.append(
                {
                    "object_x_m": float(x),
                    "object_y_m": float(y),
                    "success_frames": int(sum(successes)),
                    "total_frames": len(successes),
                    "full_sequence_success": full_success,
                    "max_position_error_m": float(max(position_errors)),
                    "max_orientation_error_rad": float(max(orientation_errors)),
                }
            )
        print(f"y={y:+.3f}: {' '.join(symbols)}", flush=True)

    passed = sum(row["full_sequence_success"] for row in rows)
    print(f"strict IK complete-sequence grid: {passed}/{len(rows)}")
    if args.out:
        output = args.out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {output}")
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
