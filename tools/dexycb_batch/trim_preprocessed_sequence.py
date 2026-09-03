#!/usr/bin/env python3
"""Atomically trim leading valid frames from one preprocessed DexYCB NPZ."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def trim_in_place(path: Path, drop_first: int, summary_path: Path | None) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    source_indices = np.asarray(data["source_frame_indices"], dtype=np.int32)
    frames = len(source_indices)
    if not 0 <= drop_first < frames:
        raise ValueError(f"drop_first must be in [0,{frames - 1}], got {drop_first}")

    removed_source_indices = source_indices[:drop_first]
    for key, value in tuple(data.items()):
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == frames:
            data[key] = array[drop_first:]

    previous_dropped = np.asarray(
        data.get("dropped_frame_indices", np.empty(0, dtype=np.int32)),
        dtype=np.int32,
    )
    data["dropped_frame_indices"] = np.unique(
        np.concatenate((previous_dropped, removed_source_indices))
    ).astype(np.int32)
    data["trimmed_initial_valid_frames"] = np.asarray(drop_first, dtype=np.int32)
    data["trimmed_initial_source_frame_indices"] = removed_source_indices

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_", suffix=".npz", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **data)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    remaining = frames - drop_first
    required_shapes = {
        "mano_joint_coords": (remaining, 21, 3),
        "mano_joint_coords_right_mano21": (remaining, 21, 3),
        "object_pos": (remaining, 3),
        "object_quat": (remaining, 4),
        "wrist_pos": (remaining, 3),
    }
    with np.load(path, allow_pickle=False) as result:
        actual = {name: result[name].shape for name in required_shapes}
        finite = all(np.isfinite(result[name]).all() for name in required_shapes)
        kept_source_indices = np.asarray(result["source_frame_indices"], dtype=int)
    invalid = {
        name: actual[name]
        for name, expected in required_shapes.items()
        if actual[name] != expected
    }
    if invalid or not finite:
        raise RuntimeError(f"trim validation failed: shapes={invalid}, finite={finite}")

    if summary_path is not None and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["valid_frames"] = remaining
        summary["dropped_frames"] = data["dropped_frame_indices"].tolist()
        summary["trimmed_initial_valid_frames"] = drop_first
        summary["kept_source_frame_range"] = [
            int(kept_source_indices[0]),
            int(kept_source_indices[-1]),
        ]
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "before": frames,
        "after": remaining,
        "removed": removed_source_indices,
        "first_source": int(kept_source_indices[0]),
        "last_source": int(kept_source_indices[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Preprocessed NPZ to overwrite atomically")
    parser.add_argument("--drop-first", type=int, default=12)
    parser.add_argument("--summary", type=Path, help="Optional preprocess_summary.json")
    args = parser.parse_args()
    path = args.input.expanduser().resolve()
    summary = trim_in_place(
        path,
        args.drop_first,
        args.summary.expanduser().resolve() if args.summary else None,
    )
    print(f"Trimmed in place: {path}")
    print(f"  frames: {summary['before']} -> {summary['after']}")
    print(f"  removed source frames: {summary['removed'].tolist()}")
    print(f"  kept source frames: {summary['first_source']}..{summary['last_source']}")


if __name__ == "__main__":
    main()
