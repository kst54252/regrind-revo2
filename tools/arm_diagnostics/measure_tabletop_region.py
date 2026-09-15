"""Offline XY scan of a full reference motion; NOT a collision/grasp certificate.

Reuses the maintained reference loader and mounted-wrist strict FK/IK. Stops a
placement at its first failed frame; numerical failure is not proof of impossibility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/rb3_revo2_ik"))
from rb3_kinematics import RB3730Kinematics
from reference_trajectory import load_reference_trajectory
from warm_start_ik import WarmStartIK


def solve_placement(job):
    xy, pos, quat, origin, neutral, mount, nfev, dt = job
    kin = RB3730Kinematics(base_position=mount["position"],
                          base_quaternion_xyzw=mount["quaternion_xyzw"])
    solver = WarmStartIK(kin)
    targets = pos + np.r_[np.asarray(xy) - origin[:2], 0.0]
    q = np.full((len(pos), 6), np.nan)
    errors = np.full((len(pos), 2), np.nan)
    warm = neutral.copy()
    failed = -1
    for i, (p, r) in enumerate(zip(targets, quat)):
        result = solver.inverse(p, r, initial_q=warm, neutral_q=neutral,
                                position_tolerance_m=1e-4,
                                orientation_tolerance_rad=1e-3, max_nfev=nfev)
        q[i], errors[i] = result.q, (result.position_error_m, result.orientation_error_rad)
        if not result.success:
            failed = i
            break
        warm = result.q
    complete = failed == -1
    speed = float(np.max(np.abs(np.diff(q, axis=0))) / dt) if complete else None
    margin = float(np.min(np.minimum(q - kin.joint_lower, kin.joint_upper - q))) if complete else None
    row = dict(x_m=float(xy[0]), y_m=float(xy[1]), full_sequence_ik=complete,
               first_failed_frame=failed, evaluated_frames=len(pos) if complete else failed + 1,
               max_evaluated_position_error_m=float(np.nanmax(errors[:, 0])),
               max_evaluated_orientation_error_rad=float(np.nanmax(errors[:, 1])),
               max_reference_path_speed_rad_s=speed, min_joint_margin_rad=margin)
    return row, q, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--x-range", nargs=2, type=float)
    parser.add_argument("--y-range", nargs=2, type=float)
    parser.add_argument("--speed-screen", type=float, default=10.,
                        help="Diagnostic rad/s bound, not an actuator simulation")
    args = parser.parse_args()
    if args.spacing <= 0 or args.workers < 1 or args.max_nfev < 1 or args.speed_screen <= 0:
        parser.error("spacing, workers, max-nfev and speed-screen must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    reference = load_reference_trajectory(args.reference)
    if any(v is None for v in (reference.wrist_pos, reference.wrist_quat_xyzw,
                               reference.object_pos, reference.object_quat_xyzw)):
        raise ValueError("Reference must include wrist and object poses")
    layout_path = ROOT / "config/workcell/rb3_revo2_table.json"
    layout = json.loads(layout_path.read_text())
    table = layout["table"]
    lower = np.asarray(table["center_xy"]) - np.asarray(table["size_xy"]) / 2
    upper = np.asarray(table["center_xy"]) + np.asarray(table["size_xy"]) / 2
    import trimesh
    mesh_path = ROOT / "007_tuna_fish_can/textured_simple.obj"
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = Rotation.from_quat(reference.object_quat_xyzw[0]).apply(mesh.vertices)
    footprint_min, footprint_max = vertices[:, :2].min(0), vertices[:, :2].max(0)
    # Grid anchored to world origin; include only centres with the full initial
    # mesh footprint on the tabletop. No arm/hand collision check is implied.
    ranges = [args.x_range, args.y_range]
    axes = []
    for i in range(2):
        lo, hi = lower[i] - footprint_min[i], upper[i] - footprint_max[i]
        if ranges[i] is not None:
            lo, hi = max(lo, ranges[i][0]), min(hi, ranges[i][1])
        axes.append(np.arange(np.ceil((lo-1e-9)/args.spacing),
                              np.floor((hi+1e-9)/args.spacing)+1) * args.spacing)
    xy = np.asarray([(x, y) for y in axes[1] for x in axes[0]])
    if len(xy) == 0:
        raise ValueError("No grid positions inside the mesh-safe table bounds")
    started = time.monotonic()
    jobs = [(point, reference.wrist_pos, reference.wrist_quat_xyzw,
             reference.object_pos[0], reference.rb3_joints[0], layout["robot_mount"],
             args.max_nfev, reference.dt) for point in xy]
    rows, configurations, all_errors = [], [], []
    print(f"Scan: {len(xy)} placements x {reference.frames} frames; {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, q, error in pool.map(solve_placement, jobs, chunksize=1):
            rows.append(row); configurations.append(q); all_errors.append(error)
            if len(rows) % 15 == 0 or len(rows) == len(xy):
                print(f"{len(rows)}/{len(xy)} complete, IK passes={sum(r['full_sequence_ik'] for r in rows)}, "
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    passed = np.array([r["full_sequence_ik"] for r in rows])
    speed_ok = np.array([r["full_sequence_ik"] and
                        r["max_reference_path_speed_rad_s"] <= args.speed_screen for r in rows])
    for row, ok in zip(rows, speed_ok):
        row["reference_speed_screen_pass"] = bool(ok)
    with (args.out / "placements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.out / "solutions.npz", xy=xy, q=np.asarray(configurations),
                        errors=np.asarray(all_errors), full_sequence_ik=passed,
                        speed_screen_pass=speed_ok, dt=reference.dt)
    def extent(mask):
        return dict(x=[float(xy[mask, 0].min()), float(xy[mask, 0].max())],
                    y=[float(xy[mask, 1].min()), float(xy[mask, 1].max())]) if mask.any() else None
    summary = dict(reference=str(args.reference.resolve()),
                   reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                   model_sha256=hashlib.sha256((ROOT / "tools/rb3_revo2_ik/rb3_model.json").read_bytes()).hexdigest(),
                   workcell=layout, mesh=str(mesh_path), frames=reference.frames, dt=reference.dt,
                   initial_object_pose_xyzw=np.r_[reference.object_pos[0], reference.object_quat_xyzw[0]].tolist(),
                   table_bounds_xy=[lower.tolist(), upper.tolist()],
                   mesh_footprint_xy=[footprint_min.tolist(), footprint_max.tolist()],
                   grid_spacing_m=args.spacing, placements=len(rows), workers=args.workers,
                   max_nfev=args.max_nfev, position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3,
                   ik_passes=int(passed.sum()), speed_screen_passes=int(speed_ok.sum()),
                   speed_screen_rad_s=args.speed_screen, ik_extent_m=extent(passed),
                   speed_screen_extent_m=extent(speed_ok), elapsed_s=time.monotonic()-started,
                   limitations=["Reference motion only, no live policy residual rollout",
                                "No collision, actuator, contact or physical grasp validation",
                                "Failure means not solved with these seeds/budget, not proven unreachable",
                                "Only sampled poses; inter-frame IK not tested",
                                "Speed is raw bounded-joint difference / reference dt; no angle wrapping",
                                "No speed check for initial approach; strict IK failure stops that placement",
                                "Axis extents are NOT a fully feasible rectangle"])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8, 9), layout="constrained")
    ax.add_patch(Rectangle(lower, *(upper-lower), fill=False, linewidth=2, label="Tabletop (Z=0)"))
    base = layout["robot_base"]
    ax.add_patch(Rectangle(np.asarray(base["center"][:2])-np.asarray(base["size"][:2])/2,
                           *base["size"][:2], color="gray", alpha=.25, label="Robot pedestal"))
    for mask, color, label in [(~passed, "#c6c9ce", "IK not solved (not proof of unreachable)"),
                                (passed & ~speed_ok, "#e59f24", "Full IK; reference speed > limit"),
                                (speed_ok, "#21885d", "Full IK + sampled speed screen")]:
        ax.scatter(*xy[mask].T, c=color, marker="s", s=28, label=f"{label}: {mask.sum()}")
    ax.scatter(*layout["robot_mount"]["position"][:2], marker="+", c="black", s=120)
    ax.scatter(*reference.object_pos[0, :2], marker="*", c="#155ad0", s=150, label="Reference can start")
    ax.set(aspect="equal", xlabel="World X [m]", ylabel="World Y [m]",
           title=f"Current can motion: tabletop IK screening\n{reference.frames} frames / fixed can orientation / {args.spacing*100:g} cm grid")
    ax.grid(alpha=.2); ax.legend(loc="upper left", fontsize=8)
    fig.savefig(args.out / "tabletop_region.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
