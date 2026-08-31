"""Visualize one transformed frame with the object mesh, MANO, and wrist."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


FINGER_CHAINS = (
    (0, 1, 2, 3, 17),
    (0, 4, 5, 6, 18),
    (0, 10, 11, 12, 19),
    (0, 7, 8, 9, 20),
    (0, 13, 14, 15, 16),
)


def _load(path: str) -> dict[str, np.ndarray]:
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    with h5py.File(path, "r") as h5_file:
        return {key: h5_file[key][()] for key in h5_file}


def _mesh(path: str, scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return mesh


def _equal_axes(ax, points: np.ndarray) -> None:
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) * 0.58, 0.06)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(min(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="World-frame .h5/.hdf5/.npz")
    parser.add_argument("--mesh", required=True, help="YCB textured_simple.obj")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--mesh-scale", type=float, help="Override stored mesh scale")
    parser.add_argument("--max-faces", type=int, default=5000)
    parser.add_argument("--save", help="Optional output image path")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    data = _load(args.trajectory)
    object_pos = np.asarray(data["object_pos_world"], dtype=float)
    object_quat = np.asarray(data["object_quat_world"], dtype=float)
    mano = np.asarray(data["mano_joint_world"], dtype=float)
    wrist = np.asarray(data["wrist_pos_world"], dtype=float)
    T = len(object_pos)
    if not 0 <= args.frame < T:
        raise ValueError(f"--frame must be in [0,{T - 1}]")
    if mano.shape != (T, 21, 3) or wrist.shape != (T, 3) or object_quat.shape != (T, 4):
        raise ValueError("invalid world trajectory shapes")

    scale = float(
        args.mesh_scale
        if args.mesh_scale is not None
        else np.asarray(data.get("mesh_scale", 1.0)).item()
    )
    mesh = _mesh(args.mesh, scale)
    quaternion_xyzw = object_quat[args.frame, (1, 2, 3, 0)]
    vertices = (
        Rotation.from_quat(quaternion_xyzw).apply(np.asarray(mesh.vertices))
        + object_pos[args.frame]
    )
    faces = np.asarray(mesh.faces)
    if len(faces) > args.max_faces:
        indices = np.linspace(0, len(faces) - 1, args.max_faces, dtype=int)
        faces = faces[indices]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    mesh_collection = Poly3DCollection(
        vertices[faces], facecolor="#f2c14e", edgecolor="#805500", linewidth=0.08, alpha=0.72
    )
    ax.add_collection3d(mesh_collection)
    points = mano[args.frame]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#1f77b4", s=28, label="MANO 21")
    for chain in FINGER_CHAINS:
        chain_points = points[np.asarray(chain)]
        ax.plot(chain_points[:, 0], chain_points[:, 1], chain_points[:, 2], color="#1f77b4")
    wrist_point = wrist[args.frame]
    ax.scatter(*wrist_point, color="#d62728", marker="x", s=110, linewidth=3, label="Wrist")

    visible = np.concatenate((vertices, points, wrist_point[None]), axis=0)
    _equal_axes(ax, visible)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    gx, gy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="#aaaaaa", alpha=0.18)
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.set_zlabel("World Z [m]")
    ax.set_title(f"DexYCB in Isaac/RB3 world | frame {args.frame}/{T - 1}")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if args.save:
        output = Path(args.save).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Saved visualization to {output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
