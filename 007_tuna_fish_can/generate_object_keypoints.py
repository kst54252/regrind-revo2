#!/usr/bin/env python3
"""Sample 50 reproducible surface keypoints from a YCB OBJ mesh."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

# Use an interactive window on a desktop, while retaining headless compatibility.
INTERACTIVE = bool(os.environ.get("DISPLAY"))
matplotlib.use("TkAgg" if INTERACTIVE else "Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


NUM_POINTS = 50
RANDOM_SEED = 42
OUTPUT_FILENAME = "object_points_50.npy"
VIS_FILENAME = "object_points_50.png"


def load_obj_mesh(obj_path: Path) -> trimesh.Trimesh:
    """Load an OBJ without changing its object-local vertex coordinates."""
    mesh = trimesh.load(obj_path, force="mesh", process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangular mesh, but loaded {type(mesh).__name__}")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"The OBJ contains no mesh surface: {obj_path}")

    return mesh


def sample_surface_points(mesh: trimesh.Trimesh) -> np.ndarray:
    """Uniformly sample points by triangle area with a fixed random seed."""
    points, _ = trimesh.sample.sample_surface(
        mesh,
        count=NUM_POINTS,
        seed=RANDOM_SEED,
    )
    points = np.asarray(points, dtype=np.float64)

    if points.shape != (NUM_POINTS, 3):
        raise RuntimeError(f"Unexpected sampled point shape: {points.shape}")
    return points


def set_axes_equal(ax: plt.Axes, bounds: np.ndarray) -> None:
    """Use equal XYZ scale so the mesh is not visually distorted."""
    center = bounds.mean(axis=0)
    radius = max(float(np.ptp(bounds, axis=0).max()) / 2.0, 1e-12)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_visualization(
    mesh: trimesh.Trimesh, points: np.ndarray, output_path: Path
) -> None:
    """Save a 3D view and, when possible, display it interactively."""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="lightgray",
        edgecolor="none",
        alpha=0.45,
        shade=True,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="crimson",
        s=32,
        depthshade=False,
        label=f"{NUM_POINTS} sampled points",
    )

    set_axes_equal(ax, np.asarray(mesh.bounds))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Uniform mesh surface samples (seed={RANDOM_SEED})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if INTERACTIVE:
        print("Interactive viewer controls:")
        print("  Rotate: left-click and drag")
        print("  Zoom: scroll wheel or right-click and drag")
        print("  Close the window to finish")
        plt.show()
    else:
        print("No DISPLAY detected; skipping the interactive window.")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 50 reproducible REGRIND object keypoints from a YCB OBJ."
    )
    parser.add_argument("obj_path", type=Path, help="Path to textured_simple.obj")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj_path = args.obj_path.expanduser().resolve()
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    if obj_path.suffix.lower() != ".obj":
        raise ValueError(f"Expected an .obj file: {obj_path}")

    mesh = load_obj_mesh(obj_path)
    points = sample_surface_points(mesh)

    points_path = obj_path.parent / OUTPUT_FILENAME
    image_path = obj_path.parent / VIS_FILENAME
    np.save(points_path, points)
    save_visualization(mesh, points, image_path)

    print(f"Saved points: {points_path}")
    print(f"Shape: {points.shape}")
    print(f"Saved visualization: {image_path}")


if __name__ == "__main__":
    main()
