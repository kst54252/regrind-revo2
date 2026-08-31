#!/usr/bin/env python3
"""Retarget every preprocessed DexYCB sequence to Revo2 and build HTML views."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGRIND_ROOT = PROJECT_ROOT / "regrind"
RETARGET_SCRIPT = REGRIND_ROOT / "scripts" / "retarget_hand_object.py"
VISUALIZE_SCRIPT = REGRIND_ROOT / "scripts" / "visualize_retargeted_sequence_interactive.py"
DEFAULT_PREPROCESSED = PROJECT_ROOT / "outputs" / "preprocessed" / "dexycb"
DEFAULT_RETARGETED = PROJECT_ROOT / "outputs" / "retargeted" / "dexycb"
DEFAULT_VISUALIZATIONS = PROJECT_ROOT / "outputs" / "visualizations" / "dexycb"


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REGRIND_ROOT, env=environment, check=True)


def _summary(sequence: str, result_path: Path, html_path: Path) -> dict:
    with h5py.File(result_path, "r") as result:
        success = np.asarray(result["solver_success"][()], dtype=bool)
        objective = np.asarray(result["objective_value"][()], dtype=float)
        robot_keypoints = np.asarray(result["robot_keypoints"][()], dtype=float)
        joints = np.asarray(result["robot_joints"][()], dtype=float)
        limits = np.asarray(result["joint_limit_violation"][()], dtype=bool)
        mano = np.asarray(result["mano_joint_coords"][()], dtype=float)
    successful_objective = objective[success & np.isfinite(objective)]
    finite_success = bool(np.isfinite(robot_keypoints[success]).all())
    return {
        "sequence": sequence,
        "frames": int(len(success)),
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()),
        "failure_indices": np.flatnonzero(~success).tolist(),
        "mean_objective": (
            float(successful_objective.mean()) if len(successful_objective) else None
        ),
        "max_objective": (
            float(successful_objective.max()) if len(successful_objective) else None
        ),
        "joint_limit_violation": bool(limits.any()),
        "robot_keypoint_shape": list(robot_keypoints.shape),
        "mano_shape": list(mano.shape),
        "robot_joint_shape": list(joints.shape),
        "finite_successful_keypoints": finite_success,
        "result": str(result_path.resolve()),
        "html": str(html_path.resolve()),
    }


def _attach_original_mano21(input_path: Path, result_path: Path) -> None:
    """Carry the sequential MANO21 skeleton into the retargeted HDF5.

    REGRIND's optimization array is reordered to Revo2 semantic kp_00..kp_20.
    Isaac visualization needs the original sequential MANO21 topology, so the
    two arrays are intentionally stored under different names.
    """
    with np.load(input_path, allow_pickle=False) as source:
        mano21 = np.asarray(source["mano_joint_coords_right_mano21"])
        source_indices = np.asarray(source["source_frame_indices"])
        camera = np.asarray(source["source_camera_serial"])
    with h5py.File(result_path, "a") as result:
        if len(mano21) != len(result["mano_joint_coords"]):
            raise ValueError(
                f"{input_path}: MANO21 and retargeting frame counts differ"
            )
        for name in (
            "mano_joint_coords_mano21",
            "source_frame_indices",
            "source_camera_serial",
        ):
            if name in result:
                del result[name]
        result.create_dataset("mano_joint_coords_mano21", data=mano21)
        result.create_dataset("source_frame_indices", data=source_indices)
        result.create_dataset(
            "source_camera_serial",
            data=camera.item(),
            dtype=h5py.string_dtype("utf-8"),
        )


def _write_gallery(items: list[dict], visualization_root: Path) -> Path:
    rows = []
    for item in items:
        relative = Path(item["html"]).relative_to(visualization_root).as_posix()
        status_class = "ok" if item["success_rate"] == 1.0 else "warn"
        rows.append(
            "<tr>"
            f"<td><a href='{html.escape(relative)}'>{html.escape(item['sequence'])}</a></td>"
            f"<td>{item['frames']}</td>"
            f"<td class='{status_class}'>{100.0 * item['success_rate']:.2f}%</td>"
            f"<td>{item['mean_objective']:.6g}</td>"
            f"<td>{html.escape(str(item['failure_indices']))}</td>"
            "</tr>"
        )
    page = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DexYCB → Revo2 retargeting</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 18px;color:#202124}
table{border-collapse:collapse;width:100%;box-shadow:0 2px 12px #0001}
th,td{padding:13px 15px;border-bottom:1px solid #ddd;text-align:left}th{background:#f4f7fb}
a{color:#1769aa;font-weight:650;text-decoration:none}a:hover{text-decoration:underline}
.note{color:#5f6368;margin-bottom:24px}.ok{color:#188038;font-weight:700}.warn{color:#b06000;font-weight:700}
</style></head><body>
<h1>DexYCB 전체 Revo2 리타게팅</h1>
<p class="note">두 번째 카메라 기준 MANO 21점, Revo2 semantic 21점, tuna fish can 50점을
같은 Plotly 화면에 표시합니다. 각 sequence 링크를 눌러 재생·정지·프레임 이동·3D 회전·확대를 할 수 있습니다.</p>
<table><thead><tr><th>Sequence</th><th>Frames</th><th>Solver success</th><th>Mean objective</th><th>Failures</th></tr></thead>
<tbody>""" + "\n".join(rows) + """</tbody></table></body></html>\n"""
    index_path = visualization_root / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page, encoding="utf-8")
    (visualization_root / "retarget_manifest.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument("--retargeted-root", type=Path, default=DEFAULT_RETARGETED)
    parser.add_argument("--visualization-root", type=Path, default=DEFAULT_VISUALIZATIONS)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sequence", action="append", help="Only retarget this sequence (repeatable).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing HDF5/HTML outputs.")
    args = parser.parse_args()

    preprocessed_root = args.preprocessed_root.expanduser().resolve()
    retargeted_root = args.retargeted_root.expanduser().resolve()
    visualization_root = args.visualization_root.expanduser().resolve()
    # Keep the venv symlink path intact. Resolving it to the underlying uv
    # interpreter bypasses pyvenv.cfg and loses the IsaacLab site-packages.
    python = args.python.expanduser().absolute()
    names = set(args.sequence or ())
    inputs = sorted(preprocessed_root.glob("*/dexycb_right_hand_preprocessed.npz"))
    if names:
        inputs = [path for path in inputs if path.parent.name in names]
    if not inputs:
        raise FileNotFoundError(f"no matching preprocessed inputs under {preprocessed_root}")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REGRIND_ROOT / "source" / "regrind")
    environment["REGRIND_DATA_DIR"] = str(REGRIND_ROOT / "data")
    environment.setdefault("MPLCONFIGDIR", "/tmp/regrind_mplconfig")
    summaries = []
    for input_path in inputs:
        sequence = input_path.parent.name
        result_path = retargeted_root / sequence / "revo2_retargeted.h5"
        html_path = visualization_root / sequence / "retargeted_interactive.html"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        if args.force or not result_path.is_file():
            _run(
                [
                    str(python),
                    str(RETARGET_SCRIPT),
                    "--robot", "revo2",
                    "--object", "tuna_fish_can",
                    "--demo", str(input_path),
                    "--demo-type", "dexycb",
                    "--input-quat-convention", "wxyz",
                    "--solver", "clarabel",
                    "--penetration-tolerance", "0.002",
                    "--no-visualize",
                    "--out", str(result_path),
                ],
                environment,
            )
        _attach_original_mano21(input_path, result_path)
        if args.force or not html_path.is_file():
            _run(
                [
                    str(python),
                    str(VISUALIZE_SCRIPT),
                    str(result_path),
                    "--out", str(html_path),
                ],
                environment,
            )
        item = _summary(sequence, result_path, html_path)
        summaries.append(item)
        print(
            f"[{sequence}] {item['success_count']}/{item['frames']} success | "
            f"HTML {html_path}",
            flush=True,
        )
    index_path = _write_gallery(summaries, visualization_root)
    print(f"Saved retargeting gallery: {index_path}")


if __name__ == "__main__":
    main()
