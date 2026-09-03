#!/usr/bin/env python3
"""Trim leading frames from a world trajectory without changing its geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def trim_h5(input_path: Path, output_path: Path, drop_first: int) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different; the source is never overwritten")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    with h5py.File(input_path, "r") as source:
        if "object_pos_world" not in source:
            raise KeyError("input does not contain object_pos_world")
        frames = int(source["object_pos_world"].shape[0])
        if not 0 <= drop_first < frames:
            raise ValueError(f"drop_first must be in [0, {frames - 1}], got {drop_first}")
        remaining = frames - drop_first

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "x") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            for key, dataset in source.items():
                data = dataset[()]
                if dataset.ndim > 0 and dataset.shape[0] == frames:
                    data = data[drop_first:]
                if key == "frame_index":
                    data = np.arange(remaining, dtype=np.int32)
                created = target.create_dataset(key, data=data)
                for attr_key, attr_value in dataset.attrs.items():
                    created.attrs[attr_key] = attr_value

            target.create_dataset("dropped_initial_frames", data=np.int32(drop_first))
            target.create_dataset("source_world_trajectory_file", data=str(input_path.resolve()))
            target.create_dataset("trajectory_edit", data="trim_first_frames_only")

    with h5py.File(output_path, "r") as result:
        object_pos = np.asarray(result["object_pos_world"], dtype=float)
        wrist_pos = np.asarray(result["wrist_pos_world"], dtype=float)
        mano = np.asarray(result["mano_joint_world_mano21"], dtype=float)
        source_indices = np.asarray(result["source_frame_indices"], dtype=int)

    if object_pos.shape != (remaining, 3):
        raise RuntimeError(f"invalid trimmed object shape: {object_pos.shape}")
    if wrist_pos.shape != (remaining, 3) or mano.shape != (remaining, 21, 3):
        raise RuntimeError(
            f"invalid trimmed hand shapes: wrist={wrist_pos.shape}, MANO={mano.shape}"
        )
    if not all(np.isfinite(value).all() for value in (object_pos, wrist_pos, mano)):
        raise RuntimeError("trimmed trajectory contains NaN/Inf")

    displacement = object_pos[-1] - object_pos[0]
    displacement_norm = float(np.linalg.norm(displacement))
    z_fraction = abs(float(displacement[2])) / max(displacement_norm, np.finfo(float).eps)
    if displacement[2] <= 0.0:
        raise RuntimeError(
            "the trimmed object trajectory does not lift along +Z: "
            f"displacement={displacement}"
        )
    if z_fraction < 0.8:
        raise RuntimeError(
            "the object motion is not predominantly along Z: "
            f"displacement={displacement}, z_fraction={z_fraction:.3f}"
        )
    return {
        "input_frames": frames,
        "output_frames": remaining,
        "source_first": int(source_indices[0]),
        "source_last": int(source_indices[-1]),
        "object_start": object_pos[0],
        "object_end": object_pos[-1],
        "object_displacement": displacement,
        "z_fraction": z_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input world trajectory HDF5")
    parser.add_argument("--out", required=True, type=Path, help="New trimmed HDF5")
    parser.add_argument("--drop-first", type=int, default=12)
    args = parser.parse_args()

    summary = trim_h5(
        args.input.expanduser().resolve(),
        args.out.expanduser().resolve(),
        args.drop_first,
    )
    print(f"Saved trimmed trajectory: {args.out.expanduser().resolve()}")
    print(f"  frames: {summary['input_frames']} -> {summary['output_frames']}")
    print(f"  source frames: {summary['source_first']}..{summary['source_last']}")
    print(f"  object start: {summary['object_start']}")
    print(f"  object end:   {summary['object_end']}")
    print(f"  displacement: {summary['object_displacement']}")
    print(f"  +Z dominance: {summary['z_fraction'] * 100.0:.2f}%")


if __name__ == "__main__":
    main()
