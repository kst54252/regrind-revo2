#!/usr/bin/env python3
"""Remove a sequence tail from raw DexYCB data and all primary trajectory artifacts.

The operation is intentionally strict: every raw camera stream must contain a
complete, identically numbered frame set and every primary processed artifact
must have the expected pre-trim length. NPZ/HDF5 rewrites are atomic. Raw image,
depth and label files in the removed range are permanently deleted only after
all trajectory files have been rewritten successfully.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_ARTIFACTS = (
    "outputs/preprocessed/dexycb/{sequence}/dexycb_right_hand_preprocessed.npz",
    "outputs/retargeted/dexycb/{sequence}/revo2_retargeted.h5",
    "outputs/isaac/dexycb/{sequence}/world_trajectory.h5",
    "outputs/isaac/dexycb/{sequence}/rb3_revo2_reference.h5",
)
RAW_PREFIXES = ("color", "aligned_depth_to_color", "labels")


def _atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trim_npz(path: Path, before: int, keep: int, phase_total: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    temporal = 0
    for name, value in tuple(values.items()):
        array = np.asarray(value)
        if array.ndim and array.shape[0] == before:
            values[name] = array[:keep]
            temporal += 1
    if not temporal:
        raise RuntimeError(f"no {before}-frame arrays found in {path}")
    values["phase_total_frames"] = np.asarray(phase_total, dtype=np.int32)
    values["trimmed_final_valid_frames"] = np.asarray(before - keep, dtype=np.int32)
    _atomic_npz(path, values)


def _create_dataset(target: h5py.File, source: h5py.Dataset, data) -> None:
    dataset = target.create_dataset(source.name.lstrip("/"), data=data)
    for key, value in source.attrs.items():
        dataset.attrs[key] = value


def _trim_h5(path: Path, before: int, keep: int, phase_total: int) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_", suffix=path.suffix, dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    temporal = 0
    try:
        with h5py.File(path, "r") as source, h5py.File(temporary, "w") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            for name, item in source.items():
                if not isinstance(item, h5py.Dataset):
                    raise TypeError(f"nested HDF5 groups are not supported: {path}:{name}")
                data = item[()]
                if item.ndim and item.shape[0] == before:
                    data = data[:keep]
                    temporal += 1
                _create_dataset(target, item, data)
            for name, value in (
                ("phase_total_frames", np.asarray(phase_total, dtype=np.int32)),
                ("trimmed_final_frames", np.asarray(before - keep, dtype=np.int32)),
            ):
                if name in target:
                    del target[name]
                target.create_dataset(name, data=value)
        if not temporal:
            raise RuntimeError(f"no {before}-frame datasets found in {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trim_raw_pose(path: Path, before: int, keep: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    changed = False
    for name, value in tuple(values.items()):
        array = np.asarray(value)
        if array.ndim and array.shape[0] == before:
            values[name] = array[:keep]
            changed = True
    if not changed:
        raise RuntimeError(f"no {before}-frame arrays found in {path}")
    _atomic_npz(path, values)


def _raw_tail_files(sequence_dir: Path, serials: list[str], before: int, keep: int) -> list[Path]:
    delete: list[Path] = []
    expected = set(range(before))
    for serial in serials:
        camera_dir = sequence_dir / serial
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"camera directory missing: {camera_dir}")
        for prefix in RAW_PREFIXES:
            files = sorted(camera_dir.glob(f"{prefix}_*"))
            indices = {
                int(path.stem.rsplit("_", 1)[1])
                for path in files
            }
            if indices != expected:
                missing = sorted(expected - indices)
                extra = sorted(indices - expected)
                raise RuntimeError(
                    f"{camera_dir}/{prefix}: frame mismatch; missing={missing}, extra={extra}"
                )
            delete.extend(path for path in files if int(path.stem.rsplit("_", 1)[1]) >= keep)
    return delete


def _update_json_metadata(
    project_root: Path,
    sequence: str,
    raw_keep: int,
    keep: int,
    drop_last: int,
) -> None:
    summary_path = project_root / "outputs/preprocessed/dexycb" / sequence / "preprocess_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            source_frames=raw_keep,
            valid_frames=keep,
            trimmed_final_valid_frames=drop_last,
            kept_source_frame_range=[int(summary["kept_source_frame_range"][0]), raw_keep - 1],
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = project_root / "outputs/isaac/dexycb/manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reference_path = project_root / f"outputs/isaac/dexycb/{sequence}/rb3_revo2_reference.h5"
        world_path = project_root / f"outputs/isaac/dexycb/{sequence}/world_trajectory.h5"
        with h5py.File(reference_path, "r") as reference, h5py.File(world_path, "r") as world:
            success = np.asarray(reference["ik_success"], dtype=bool)
            pos_error = np.asarray(reference["position_error_m"], dtype=float)
            ori_error = np.asarray(reference["orientation_error_rad"], dtype=float)
            rb3 = np.asarray(reference["rb3_joints"], dtype=float)
            object_pos = np.asarray(world["object_pos_world"], dtype=float)
        for item in manifest:
            if item.get("sequence") != sequence:
                continue
            item.update(
                frames=keep,
                ik_success_count=int(success.sum()),
                ik_success_rate=float(success.mean()),
                failed_indices=np.flatnonzero(~success).tolist(),
                max_position_error_m=float(pos_error.max()),
                max_orientation_error_rad=float(ori_error.max()),
                max_rb3_step_rad=float(np.linalg.norm(np.diff(rb3, axis=0), axis=1).max(initial=0.0)),
                last_object_position=object_pos[-1].tolist(),
                object_delta_z_m=float(object_pos[-1, 2] - object_pos[0, 2]),
                object_z_range_m=[float(object_pos[:, 2].min()), float(object_pos[:, 2].max())],
            )
            break
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def trim(project_root: Path, sequence: str, drop_last: int, *, dry_run: bool = False) -> dict:
    sequence_dir = project_root / "dataset" / sequence
    meta_path = sequence_dir / "meta.yml"
    pose_path = sequence_dir / "pose.npz"
    if not meta_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(f"incomplete raw sequence: {sequence_dir}")
    meta_text = meta_path.read_text(encoding="utf-8")
    meta = yaml.safe_load(meta_text)
    raw_before = int(meta["num_frames"])
    raw_keep = raw_before - drop_last
    if drop_last <= 0 or raw_keep < 2:
        raise ValueError(f"invalid drop_last={drop_last} for {raw_before} raw frames")
    serials = [str(value) for value in meta["serials"]]
    delete = _raw_tail_files(sequence_dir, serials, raw_before, raw_keep)

    preprocessed = project_root / PRIMARY_ARTIFACTS[0].format(sequence=sequence)
    with np.load(preprocessed, allow_pickle=False) as archive:
        processed_before = len(archive["source_frame_indices"])
        source_indices = np.asarray(archive["source_frame_indices"], dtype=int)
    processed_keep = processed_before - drop_last
    if processed_keep < 2 or int(source_indices[processed_keep - 1]) != raw_keep - 1:
        raise RuntimeError(
            "raw and processed tails do not align: "
            f"raw keep ends at {raw_keep - 1}, processed keep ends at {source_indices[processed_keep - 1]}"
        )

    artifact_paths = [
        project_root / template.format(sequence=sequence)
        for template in PRIMARY_ARTIFACTS
    ]
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(f"primary trajectory missing: {path}")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                found = any(
                    archive[name].ndim and archive[name].shape[0] == processed_before
                    for name in archive.files
                )
        else:
            with h5py.File(path, "r") as archive:
                found = any(
                    isinstance(item, h5py.Dataset)
                    and item.ndim
                    and item.shape[0] == processed_before
                    for item in archive.values()
                )
        if not found:
            raise RuntimeError(f"no {processed_before}-frame arrays found in {path}")

    result = {
        "raw_before": raw_before,
        "raw_after": raw_keep,
        "processed_before": processed_before,
        "processed_after": processed_keep,
        "deleted_raw_files": len(delete),
        "kept_source_range": [int(source_indices[0]), int(source_indices[processed_keep - 1])],
        "phase_total_frames": processed_before,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    for path in artifact_paths:
        if path.suffix == ".npz":
            _trim_npz(path, processed_before, processed_keep, processed_before)
        else:
            _trim_h5(path, processed_before, processed_keep, processed_before)

    # Keep the preprocessed source count consistent with the physically
    # shortened raw sequence.
    with np.load(preprocessed, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["source_frame_count"] = np.asarray(raw_keep, dtype=np.int32)
    values["trimmed_final_source_frame_indices"] = np.arange(raw_keep, raw_before, dtype=np.int32)
    _atomic_npz(preprocessed, values)

    _trim_raw_pose(pose_path, raw_before, raw_keep)
    updated_meta = re.sub(
        r"(?m)^num_frames:[ \t]*\d+[ \t]*$",
        f"num_frames: {raw_keep}",
        meta_text,
        count=1,
    )
    if updated_meta == meta_text:
        raise RuntimeError(f"could not update num_frames in {meta_path}")
    meta_path.write_text(updated_meta, encoding="utf-8")
    for path in delete:
        path.unlink()

    _update_json_metadata(project_root, sequence, raw_keep, processed_keep, drop_last)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence")
    parser.add_argument("--drop-last", type=int, default=20)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = trim(
        args.project_root.expanduser().resolve(),
        args.sequence,
        args.drop_last,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
