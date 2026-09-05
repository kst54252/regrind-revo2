#!/usr/bin/env python3
"""Create strict-IK RB3+Revo2 Isaac references for every retargeted sequence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLD_SCRIPT = PROJECT_ROOT / "tools" / "dexycb_world_transform" / "transform_trajectory.py"
IK_SCRIPT = PROJECT_ROOT / "tools" / "rb3_revo2_ik" / "build_reference_trajectory.py"
TUNA_MESH = PROJECT_ROOT / "007_tuna_fish_can" / "textured_simple.obj"
DEFAULT_RETARGETED = PROJECT_ROOT / "outputs" / "retargeted" / "dexycb"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "isaac" / "dexycb"
WORKCELL_CONFIG = PROJECT_ROOT / "config" / "workcell" / "rb3_revo2_table.json"

# Keep gravity tied to the second camera, never to the tuna can's local axes.
# DexYCB optical camera -Y maps to Isaac world +Z; the per-sequence yaw only
# rotates the table plane and was selected by strict full-pose IK search at
# X=0.40 m, Y=0.00 m.  Thus a standing can whose annotated local +Z points
# downward remains valid without flipping the entire hand/object trajectory.
ISAAC_ALIGNMENT = {
    "20200709_143626_right": {"yaw_deg": 120.0},
    "20200709_143703_right": {"yaw_deg": -120.0},
    "20200709_143747_left": {"yaw_deg": 150.0},
    "20200709_143826_left": {"yaw_deg": -30.0},
    "20200709_143907_right": {"yaw_deg": -120.0},
}


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _validate(sequence: str, world_path: Path, reference_path: Path) -> dict:
    with WORKCELL_CONFIG.open("r", encoding="utf-8") as config_file:
        workcell = json.load(config_file)
    expected_base_position = np.asarray(workcell["robot_mount"]["position"], dtype=float)
    with h5py.File(world_path, "r") as world:
        object_pos = np.asarray(world["object_pos_world"][()], dtype=float)
        mesh_bottom_z = float(world["first_mesh_world_min_z"][()])
        camera = world["source_camera_serial"][()]
        if isinstance(camera, bytes):
            camera = camera.decode("utf-8")
        convention = world["camera_frame_convention"][()]
        if isinstance(convention, bytes):
            convention = convention.decode("utf-8")
        mano = np.asarray(world["mano_joint_world_mano21"][()], dtype=float)
    with h5py.File(reference_path, "r") as reference:
        success = np.asarray(reference["ik_success"][()], dtype=bool)
        rb3 = np.asarray(reference["rb3_joints"][()], dtype=float)
        revo2 = np.asarray(reference["revo2_joints"][()], dtype=float)
        combined = np.asarray(reference["reference_joints"][()], dtype=float)
        position_error = np.asarray(reference["position_error_m"][()], dtype=float)
        orientation_error = np.asarray(reference["orientation_error_rad"][()], dtype=float)
        limit = np.asarray(reference["joint_limit_violation"][()], dtype=bool)
        skeleton = np.asarray(reference["mano_joint_world"][()], dtype=float)
        base_position = np.asarray(reference["rb3_base_position"][()], dtype=float)
    frames = len(success)
    if not success.all():
        raise RuntimeError(
            f"{sequence}: strict IK failed at {np.flatnonzero(~success).tolist()}"
        )
    expected = {
        "rb3": (frames, 6),
        "revo2": (frames, 6),
        "combined": (frames, 12),
        "skeleton": (frames, 21, 3),
    }
    actual = {
        "rb3": rb3.shape,
        "revo2": revo2.shape,
        "combined": combined.shape,
        "skeleton": skeleton.shape,
    }
    if actual != expected:
        raise RuntimeError(f"{sequence}: invalid reference shapes {actual}")
    arrays = (object_pos, mano, rb3, revo2, combined, skeleton)
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError(f"{sequence}: Isaac reference contains NaN/Inf")
    if limit.any():
        raise RuntimeError(f"{sequence}: joint-limit violation")
    if not np.allclose(base_position, expected_base_position, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(
            f"{sequence}: RB3 base position {base_position.tolist()} does not match "
            f"workcell mount {expected_base_position.tolist()}"
        )
    if camera != "839512060362":
        raise RuntimeError(f"{sequence}: expected second camera, got {camera}")
    return {
        "sequence": sequence,
        "camera": camera,
        "camera_frame_convention": convention,
        "frames": frames,
        "ik_success_count": int(success.sum()),
        "ik_success_rate": float(success.mean()),
        "failed_indices": np.flatnonzero(~success).tolist(),
        "max_position_error_m": float(position_error.max()),
        "max_orientation_error_rad": float(orientation_error.max()),
        "max_rb3_step_rad": float(
            np.linalg.norm(np.diff(rb3, axis=0), axis=1).max(initial=0.0)
        ),
        "joint_limit_violation": bool(limit.any()),
        "finite": True,
        "first_object_position": object_pos[0].tolist(),
        "last_object_position": object_pos[-1].tolist(),
        "object_delta_z_m": float(object_pos[-1, 2] - object_pos[0, 2]),
        "object_z_range_m": [float(object_pos[:, 2].min()), float(object_pos[:, 2].max())],
        "first_mesh_min_z": mesh_bottom_z,
        "rb3_base_position": base_position.tolist(),
        "world_trajectory": str(world_path.resolve()),
        "reference_trajectory": str(reference_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retargeted-root", type=Path, default=DEFAULT_RETARGETED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sequence", action="append", help="Only prepare this sequence (repeatable).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    retargeted_root = args.retargeted_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    python = args.python.expanduser().absolute()
    names = set(args.sequence or ISAAC_ALIGNMENT)
    unknown = names - set(ISAAC_ALIGNMENT)
    if unknown:
        raise ValueError(f"no Isaac alignment configured for {sorted(unknown)}")
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/regrind_mplconfig")
    summaries = []
    for sequence in sorted(names):
        source = retargeted_root / sequence / "revo2_retargeted.h5"
        if not source.is_file():
            raise FileNotFoundError(f"retargeted trajectory not found: {source}")
        destination = output_root / sequence
        world_path = destination / "world_trajectory.h5"
        reference_path = destination / "rb3_revo2_reference.h5"
        destination.mkdir(parents=True, exist_ok=True)
        alignment = ISAAC_ALIGNMENT[sequence]
        if args.force or not world_path.is_file():
            _run(
                [
                    str(python), str(WORLD_SCRIPT), str(source),
                    "--mesh", str(TUNA_MESH),
                    "--out", str(world_path),
                    "--desired-x", "0.4", "--desired-y", "0.0",
                    "--camera-frame-convention", "dexycb_y_down",
                    "--world-yaw-deg", str(alignment["yaw_deg"]),
                ],
                environment,
            )
        if args.force or not reference_path.is_file():
            _run(
                [
                    str(python), str(IK_SCRIPT), str(world_path),
                    "--out", str(reference_path),
                    "--input-quat-convention", "auto",
                ],
                environment,
            )
        summary = _validate(sequence, world_path, reference_path)
        summary.update(alignment)
        summaries.append(summary)
        print(
            f"[{sequence}] strict IK {summary['ik_success_count']}/{summary['frames']} | "
            f"max step {summary['max_rb3_step_rad']:.6g} rad",
            flush=True,
        )
    manifest = output_root / "manifest.json"
    if args.sequence and manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        merged = {item["sequence"]: item for item in previous}
        merged.update({item["sequence"]: item for item in summaries})
        summaries = [merged[name] for name in sorted(merged)]
    manifest.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved Isaac reference manifest: {manifest}")


if __name__ == "__main__":
    main()
