#!/usr/bin/env python3
"""Preprocess every DexYCB sequence under ``dataset/``.

The compact output contains only trajectory data needed by REGRIND and the
interactive viewer. Raw RGB/depth/label files remain untouched in ``dataset``.
For left-hand captures, ``mano_joint_coords`` is converted to a right-hand
interaction by reflecting the points in the grasped object's local X plane.
The unmodified points are retained as ``mano_joint_coords_original``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessed" / "dexycb"
DEFAULT_TUNA_DIR = PROJECT_ROOT / "007_tuna_fish_can"

# Raw DexYCB ``joint_3d`` in this dataset uses the common sequential MANO21
# topology: wrist, then four joints for thumb/index/middle/ring/little.
# REGRIND/Revo2 uses wrist, three non-tip joints for index/middle/little/ring,
# four thumb joints, then the four non-thumb fingertips. Reorder explicitly at
# the input boundary so correspondence and skeleton topology cannot be mixed.
MANO21_SEQUENTIAL_TO_REVO_SEMANTIC = np.array(
    (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20),
    dtype=np.int32,
)


def _mirror_in_object_local_x(
    points: np.ndarray, object_pos: np.ndarray, object_quat_wxyz: np.ndarray
) -> np.ndarray:
    rotations = Rotation.from_quat(object_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    relative = points - object_pos[:, None, :]
    local = np.einsum("tji,tkj->tki", rotations, relative)
    local[:, :, 0] *= -1.0
    return np.einsum("tij,tkj->tki", rotations, local) + object_pos[:, None, :]


def _object_assets(tuna_dir: Path) -> tuple[Path, np.ndarray]:
    """Load the project tuna asset selected for these five demonstrations."""
    mesh_path = tuna_dir / "textured_simple.obj"
    points_path = tuna_dir / "object_points_50.npy"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"YCB mesh not found: {mesh_path}")
    if not points_path.is_file():
        raise FileNotFoundError(f"YCB surface cloud not found: {points_path}")
    points = np.asarray(np.load(points_path, allow_pickle=False), dtype=np.float64)
    if points.shape != (50, 3) or not np.isfinite(points).all():
        raise ValueError(f"tuna object points must be finite shape (50,3): {points_path}")
    return mesh_path.resolve(), points


def preprocess_sequence(
    sequence_dir: Path,
    output_root: Path,
    tuna_dir: Path,
    camera_serial: str | None = None,
) -> dict:
    meta_path = sequence_dir / "meta.yml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as stream:
        meta = yaml.safe_load(stream)

    serials = [str(value) for value in meta["serials"]]
    if len(serials) < 2 and camera_serial is None:
        raise ValueError(f"{sequence_dir.name} has no second camera: {serials}")
    serial = camera_serial or serials[1]
    if serial not in serials:
        raise ValueError(
            f"camera {serial!r} is not present in {sequence_dir.name}: {serials}"
        )
    labels = sorted((sequence_dir / serial).glob("labels_*.npz"))
    expected = int(meta["num_frames"])
    if len(labels) != expected:
        raise ValueError(
            f"{sequence_dir.name}: expected {expected} label files for {serial}, "
            f"found {len(labels)}"
        )

    ycb_ids = [int(value) for value in meta["ycb_ids"]]
    grasp_index = int(meta["ycb_grasp_ind"])
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(f"invalid ycb_grasp_ind={grasp_index} in {meta_path}")
    source_grasped_ycb_id = ycb_ids[grasp_index]
    mesh_path, object_points = _object_assets(tuna_dir)

    mano_all = []
    object_pos_all = []
    object_quat_all = []
    source_indices = []
    dropped_indices = []
    rotation_determinants = []
    for frame_index, label_path in enumerate(labels):
        with np.load(label_path, allow_pickle=False) as label:
            mano = np.asarray(label["joint_3d"], dtype=np.float64)
            poses = np.asarray(label["pose_y"], dtype=np.float64)
        if mano.shape != (1, 21, 3):
            raise ValueError(f"{label_path}: joint_3d has shape {mano.shape}")
        if poses.shape != (len(ycb_ids), 3, 4):
            raise ValueError(
                f"{label_path}: pose_y has shape {poses.shape}, expected "
                f"({len(ycb_ids)},3,4)"
            )
        mano = mano[0]
        pose = poses[grasp_index]
        valid = (
            np.isfinite(mano).all()
            and not np.any(np.isclose(mano, -1.0))
            and np.isfinite(pose).all()
        )
        if not valid:
            dropped_indices.append(frame_index)
            continue
        rotation = pose[:, :3]
        determinant = float(np.linalg.det(rotation))
        orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        if determinant <= 0.0 or abs(determinant - 1.0) > 5.0e-3 or orthogonality_error > 5.0e-3:
            raise ValueError(
                f"{label_path}: invalid object rotation, det={determinant}, "
                f"orthogonality error={orthogonality_error}"
            )
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        mano_all.append(mano)
        object_pos_all.append(pose[:, 3])
        object_quat_all.append(quaternion_xyzw[[3, 0, 1, 2]])
        source_indices.append(frame_index)
        rotation_determinants.append(determinant)

    if not mano_all:
        raise RuntimeError(f"{sequence_dir.name}: no valid annotated frames")
    mano_original = np.asarray(mano_all, dtype=np.float64)
    object_pos = np.asarray(object_pos_all, dtype=np.float64)
    object_quat = np.asarray(object_quat_all, dtype=np.float64)
    source_side = str(meta["mano_sides"][0]).lower()
    if source_side not in ("left", "right"):
        raise ValueError(f"unsupported MANO side {source_side!r} in {meta_path}")
    mano_right_sequential = (
        _mirror_in_object_local_x(mano_original, object_pos, object_quat)
        if source_side == "left"
        else mano_original.copy()
    )
    mano_right = mano_right_sequential[:, MANO21_SEQUENTIAL_TO_REVO_SEMANTIC]

    original_dist = np.linalg.norm(mano_original - object_pos[:, None, :], axis=2)
    right_dist = np.linalg.norm(
        mano_right_sequential - object_pos[:, None, :], axis=2
    )
    distance_error = float(np.max(np.abs(original_dist - right_dist)))
    arrays = (
        mano_original,
        mano_right_sequential,
        mano_right,
        object_pos,
        object_quat,
        object_points,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError(f"{sequence_dir.name}: preprocessed output contains NaN/Inf")

    output_dir = output_root / sequence_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dexycb_right_hand_preprocessed.npz"
    np.savez_compressed(
        output_path,
        sequence_name=sequence_dir.name,
        source_sequence_dir=str(sequence_dir.resolve()),
        source_camera_serial=serial,
        source_hand_side=source_side,
        target_hand_side="right",
        hand_conversion=("object_local_x_reflection" if source_side == "left" else "none"),
        quaternion_order="wxyz",
        fps=np.asarray(30.0),
        source_frame_count=np.asarray(expected, dtype=np.int32),
        source_frame_indices=np.asarray(source_indices, dtype=np.int32),
        dropped_frame_indices=np.asarray(dropped_indices, dtype=np.int32),
        mano_joint_coords_original=mano_original.astype(np.float32),
        mano_joint_coords_right_mano21=mano_right_sequential.astype(np.float32),
        mano_joint_coords=mano_right.astype(np.float32),
        human_hand_keypoints=mano_right.astype(np.float32),
        wrist_pos=mano_right[:, 0].astype(np.float32),
        wrist_quat=np.tile(np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (len(mano_right), 1)),
        object_pos=object_pos.astype(np.float32),
        object_quat=object_quat.astype(np.float32),
        object_points_local=object_points.astype(np.float32),
        mano_source_joint_order="mano21_sequential_thumb_index_middle_ring_little",
        mano_retarget_joint_order="revo2_semantic_kp00_to_kp20",
        mano_source_to_revo_indices=MANO21_SEQUENTIAL_TO_REVO_SEMANTIC,
        source_grasped_ycb_id=np.asarray(source_grasped_ycb_id, dtype=np.int32),
        grasped_object_name="007_tuna_fish_can",
        object_mesh_path=str(mesh_path),
    )
    result = {
        "sequence": sequence_dir.name,
        "source_frames": expected,
        "valid_frames": len(mano_right),
        "dropped_frames": dropped_indices,
        "camera": serial,
        "source_hand_side": source_side,
        "target_hand_side": "right",
        "source_object_id": source_grasped_ycb_id,
        "object_name": "007_tuna_fish_can",
        "max_rotation_det_error": float(
            np.max(np.abs(np.asarray(rotation_determinants) - 1.0))
        ),
        "max_mirror_distance_error_m": distance_error,
        "output": str(output_path.resolve()),
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tuna-asset-dir", type=Path, default=DEFAULT_TUNA_DIR)
    parser.add_argument(
        "--camera-serial",
        help="Use this camera in every sequence; default uses the first serial in meta.yml.",
    )
    parser.add_argument("--sequence", action="append", help="Only process this sequence (repeatable).")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tuna_dir = args.tuna_asset_dir.expanduser().resolve()
    names = set(args.sequence or ())
    sequences = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (not names or path.name in names)
    )
    if not sequences:
        raise FileNotFoundError(f"no matching sequences under {dataset_root}")
    missing = names - {path.name for path in sequences}
    if missing:
        raise FileNotFoundError(f"requested sequences not found: {sorted(missing)}")

    summaries = []
    for sequence in sequences:
        summary = preprocess_sequence(
            sequence, output_root, tuna_dir, camera_serial=args.camera_serial
        )
        summaries.append(summary)
        print(
            f"[{summary['sequence']}] {summary['valid_frames']}/"
            f"{summary['source_frames']} valid | {summary['source_hand_side']} -> right | "
            f"{summary['object_name']}"
        )
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
