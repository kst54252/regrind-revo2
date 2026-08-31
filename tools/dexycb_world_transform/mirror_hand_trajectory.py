"""Convert a compact left-hand DexYCB demo to a right-hand interaction.

Only the MANO points are reflected, in each frame's object-local coordinates.
The observed object pose remains unchanged, so the converted hand preserves
its object-relative timing and distances while changing chirality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def _load(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as h5_file:
            return {key: h5_file[key][()] for key in h5_file}
    raise ValueError("input must be .npz, .h5, or .hdf5")


def _write(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, **{key: np.asarray(value) for key, value in data.items()})
        return
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("output must be .npz, .h5, or .hdf5")
    with h5py.File(path, "w") as h5_file:
        for key, value in data.items():
            array = np.asarray(value)
            if array.dtype.kind in ("U", "O"):
                h5_file.create_dataset(
                    key,
                    data=array.astype(object),
                    dtype=h5py.string_dtype("utf-8"),
                )
            else:
                h5_file.create_dataset(key, data=array)


def _first(data: dict, names: tuple[str, ...], description: str) -> tuple[str, np.ndarray]:
    for name in names:
        if name in data:
            return name, np.asarray(data[name])
    raise KeyError(f"input has no {description}; tried {names}")


def _decode_scalar(value, default: str) -> str:
    if value is None:
        return default
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def mirror_hand_in_object_frame(
    mano_joint_coords: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    quaternion_order: str = "wxyz",
    mirror_axis: int = 0,
) -> np.ndarray:
    """Reflect MANO points about one object-local coordinate plane."""
    mano = np.asarray(mano_joint_coords, dtype=float)
    object_pos = np.asarray(object_pos, dtype=float)
    object_quat = np.asarray(object_quat, dtype=float)
    frames = len(mano)
    if mano.shape != (frames, 21, 3):
        raise ValueError(f"MANO points must have shape (T,21,3), got {mano.shape}")
    if object_pos.shape != (frames, 3) or object_quat.shape != (frames, 4):
        raise ValueError("object pose must have shapes (T,3) and (T,4)")
    if mirror_axis not in (0, 1, 2):
        raise ValueError("mirror_axis must be 0, 1, or 2")
    if not all(np.isfinite(value).all() for value in (mano, object_pos, object_quat)):
        raise ValueError("input contains NaN/Inf")
    if quaternion_order == "wxyz":
        quaternion_xyzw = object_quat[:, [1, 2, 3, 0]]
    elif quaternion_order == "xyzw":
        quaternion_xyzw = object_quat
    else:
        raise ValueError("quaternion_order must be 'wxyz' or 'xyzw'")
    norms = np.linalg.norm(quaternion_xyzw, axis=1)
    if np.any(norms < 1.0e-12):
        raise ValueError("object quaternion contains a zero-length value")
    rotations = Rotation.from_quat(quaternion_xyzw / norms[:, None]).as_matrix()

    relative_world = mano - object_pos[:, None, :]
    local = np.einsum("tji,tkj->tki", rotations, relative_world)
    local[:, :, mirror_axis] *= -1.0
    mirrored = (
        np.einsum("tij,tkj->tki", rotations, local)
        + object_pos[:, None, :]
    )
    if not np.isfinite(mirrored).all():
        raise RuntimeError("mirrored MANO trajectory contains NaN/Inf")
    return mirrored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--mirror-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Object-local coordinate whose sign is reflected (default: x).",
    )
    parser.add_argument(
        "--input-hand-side", choices=("left", "right"), default="left"
    )
    parser.add_argument(
        "--output-hand-side", choices=("left", "right"), default="right"
    )
    args = parser.parse_args()

    source_path = args.input.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    data = _load(source_path)
    hand_key, hand = _first(
        data,
        ("mano_joint_coords", "human_hand_keypoints", "mano_joint_coord"),
        "MANO keypoints",
    )
    _, object_pos = _first(
        data, ("object_pos", "obj_pos", "object_positions"), "object position"
    )
    _, object_quat = _first(
        data, ("object_quat", "obj_quat", "object_quaternions"), "object quaternion"
    )
    order = _decode_scalar(
        data.get("quaternion_order", data.get("quat_convention")), "wxyz"
    ).lower()
    axis = {"x": 0, "y": 1, "z": 2}[args.mirror_axis]
    mirrored = mirror_hand_in_object_frame(
        hand, object_pos, object_quat, quaternion_order=order, mirror_axis=axis
    )

    result = dict(data)
    result[hand_key] = mirrored.astype(np.asarray(hand).dtype, copy=False)
    result.update(
        {
            "source_trajectory_file": str(source_path),
            "source_hand_side": args.input_hand_side,
            "target_hand_side": args.output_hand_side,
            "hand_mirror_frame": "object_local",
            "hand_mirror_axis": args.mirror_axis,
        }
    )
    _write(output_path, result)

    original_radius = np.linalg.norm(hand - object_pos[:, None, :], axis=2)
    mirrored_radius = np.linalg.norm(mirrored - object_pos[:, None, :], axis=2)
    print("[DexYCB hand chirality conversion]")
    print(f"  source / target side:       {args.input_hand_side} -> {args.output_hand_side}")
    print(f"  mirror:                     object-local {args.mirror_axis}=0 plane")
    print(f"  frames / shape:             {len(mirrored)} / {mirrored.shape}")
    print(
        "  max object-distance error:  "
        f"{np.max(np.abs(original_radius - mirrored_radius)):.9g} m"
    )
    print(f"  finite:                     {bool(np.isfinite(mirrored).all())}")
    print(f"Saved right-hand demo to {output_path}")


if __name__ == "__main__":
    main()
