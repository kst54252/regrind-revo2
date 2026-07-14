import torch
import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict


def compute_interaction_mesh_neighborhood(
    object_keypoints: np.ndarray,
    mano_keypoints: np.ndarray,
    max_neighbors: int
) -> np.ndarray:
    """
    Compute Delaunay tetrahedralization for the given object and mano keypoints.

    Args:
        object_keypoints: (N, 3), tensor of object keypoint coordinates
        mano_keypoints: (M, 3), tensor of mano keypoint coordinates
        max_neighbors (K): int, maximum number of neighbors for each vertex
    Returns:
        (N + N, K) array of neighbor indices (-1 for padding)
    """
    interaction_mesh = Delaunay(np.concatenate([object_keypoints, mano_keypoints], axis=0))

    n_vertices = len(interaction_mesh.points)
    assert n_vertices == len(object_keypoints) + len(mano_keypoints)
    neighbors = defaultdict(set)

    for simplex in interaction_mesh.simplices:
        for i in range(4):
            for j in range(4):
                if i != j:
                    neighbors[simplex[i]].add(simplex[j])

    actual_max_neighbors = max(len(neighs) for neighs in neighbors.values())
    if actual_max_neighbors > max_neighbors:
        print(f"Actual max neighbors: {actual_max_neighbors} > {max_neighbors}")

    neighborhood = np.full((n_vertices, max_neighbors), -1, dtype=np.int64)  # (N + M, K)

    for vertex_id, neighs in neighbors.items():
        neighs_list = list(neighs)[:max_neighbors]
        neighborhood[vertex_id, :len(neighs_list)] = neighs_list

    return neighborhood


@torch.jit.script
def compute_laplacian_coordinates(points: torch.Tensor, neighborhood: torch.Tensor) -> torch.Tensor:
    """
    Laplacian coordinate computation (batched).

    Args:
        points: (B, N, 3) tensor of point positions
        neighborhood: (B, N, K) tensor of neighbor indices (-1 for padding)

    Returns:
        (B, N, 3) tensor of laplacian coordinates
    """
    assert points.ndim == 3 and neighborhood.ndim == 3, ("points and neighborhood must be 3D tensors, "
    f"got points.shape: {points.shape}, neighborhood.shape: {neighborhood.shape}.")
    # neighborhood uses -1 as padding; build a valid mask
    valid_mask = neighborhood >= 0  # (B, N, K) boolean
    # Replace invalid indices with 0 for safe indexing; cast to long for indexing
    safe_neighbors = torch.where(valid_mask, neighborhood, torch.zeros_like(neighborhood)).long()  # (B, N, K)

    B = points.shape[0]
    N = points.shape[1]
    K = safe_neighbors.shape[2]

    # Gather neighbor points per (B, N, K) using torch.gather along vertex dimension
    # Prepare inputs for gather: expand points to (B, N, K, 3), and indices to same shape
    points_expanded = points.unsqueeze(2).expand(B, N, K, 3)  # (B, N, K, 3)
    index_expanded = safe_neighbors.unsqueeze(-1).expand(B, N, K, 3)  # (B, N, K, 3)
    # neighbor_points[b, n, k, c] = points_expanded[b, index_expanded[b, n, k, c], k, c]
    #                             = points[b, safe_neighbors[b, n, k], c]
    neighbor_points = torch.gather(points_expanded, dim=1, index=index_expanded)  # (B, N, K, 3)

    # Uniform weights over valid neighbors
    counts = valid_mask.sum(dim=2, keepdim=True)  # (B, N, 1)
    counts = counts.clamp_min(1)  # avoid divide-by-zero when a row has no neighbors
    weights = valid_mask.to(points.dtype) / counts.to(points.dtype)  # (B, N, K)

    # Compute weighted average of neighbor positions
    weighted_avg = (weights.unsqueeze(-1) * neighbor_points).sum(dim=2)  # (B, N, 3)

    # Laplacian coordinates
    laplacian = points - weighted_avg  # (B, N, 3)
    return laplacian
