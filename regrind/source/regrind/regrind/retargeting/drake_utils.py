"""Drake/geometry utilities for hand-object interaction-mesh retargeting.

Migrated subset of ``omniretarget/src/utils.py`` covering only the hand-object path:
Drake plant construction, per-hand collision exclusions, interaction-mesh /
Laplacian helpers, and frame-transform helpers. The full-body humanoid helpers
(SMPL loading, URDF templating, foot-sticking, etc.) are intentionally left behind,
along with their heavy ``smplx`` / ``jinja2`` dependencies.

Sim-free w.r.t. Isaac Sim, but depends on ``pydrake`` (and therefore must not be
imported at ``regrind`` package import time).
"""

import os
import tempfile

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CollisionFilterDeclaration,
    DiagramBuilder,
    GeometrySet,
    MultibodyPlant,
    Parser,
    RigidTransform,
)


# ---------------------------------------------------------------------------
# Drake plant construction & collision exclusions
# ---------------------------------------------------------------------------
def _collision_geometry_ids_for_link(plant, link_name):
    """All proximity collision geometry IDs registered on ``link_name``."""
    body = plant.GetBodyByName(link_name)
    return list(plant.GetCollisionGeometriesForBody(body))


def _create_geometry_set_for_link(plant, link_names):
    """Create a GeometrySet for a specific link's collision geometry."""
    geometry_ids = []
    for link_name in link_names:
        body = plant.GetBodyByName(link_name)
        geometry_id = plant.GetCollisionGeometriesForBody(body)[0]
        geometry_ids.append(geometry_id)
    return GeometrySet(geometry_ids)


def _setup_leaphand_collision_exclusions(plant, scene_graph):
    collision_filter_declaration = CollisionFilterDeclaration()
    filter_manager = scene_graph.collision_filter_manager()
    # Palm
    palm_set = _create_geometry_set_for_link(
        plant,
        ["palm_lower"],
    )
    # Finger links except fingertips
    fingers_set = _create_geometry_set_for_link(
        plant,
        [
            "mcp_joint", "pip", "dip",
            "mcp_joint_2", "pip_2", "dip_2",
            "mcp_joint_3", "pip_3", "dip_3",
            "thumb_temp_base", "thumb_pip", "thumb_dip"
        ],
    )
    filter_manager.Apply(
        collision_filter_declaration.ExcludeBetween(palm_set, fingers_set)
    )


_WUJIHAND_COLLISION_EXCLUDE_BODY_PAIRS = (
    # Palm ↔ proximal / middle phalanges (MuJoCo-style excludes).
    ("right_palm_link", "right_finger1_link2"),
    ("right_palm_link", "right_finger1_link3"),
    ("right_palm_link", "right_finger2_link2"),
    ("right_palm_link", "right_finger2_link3"),
    ("right_palm_link", "right_finger3_link2"),
    ("right_palm_link", "right_finger3_link3"),
    ("right_palm_link", "right_finger4_link2"),
    ("right_palm_link", "right_finger4_link3"),
    ("right_palm_link", "right_finger5_link2"),
    ("right_palm_link", "right_finger5_link3"),
    # Thumb — adjacent non-neighbor pairs.
    ("right_finger1_link1", "right_finger1_link3"),
    ("right_finger1_link1", "right_finger1_link4"),
    ("right_finger1_link2", "right_finger1_link4"),
    # Remaining digits.
    ("right_finger2_link1", "right_finger2_link3"),
    ("right_finger2_link1", "right_finger2_link4"),
    ("right_finger2_link2", "right_finger2_link4"),
    ("right_finger3_link1", "right_finger3_link3"),
    ("right_finger3_link1", "right_finger3_link4"),
    ("right_finger3_link2", "right_finger3_link4"),
    ("right_finger4_link1", "right_finger4_link3"),
    ("right_finger4_link1", "right_finger4_link4"),
    ("right_finger4_link2", "right_finger4_link4"),
    ("right_finger5_link1", "right_finger5_link3"),
    ("right_finger5_link1", "right_finger5_link4"),
    ("right_finger5_link2", "right_finger5_link4"),
)


def _setup_wujihand_collision_exclusions(plant, scene_graph):
    """Pairwise collision suppression matching the Wuji hand MuJoCo exclude list."""
    filter_manager = scene_graph.collision_filter_manager()
    declaration = CollisionFilterDeclaration()
    for body_a, body_b in _WUJIHAND_COLLISION_EXCLUDE_BODY_PAIRS:
        ids_a = _collision_geometry_ids_for_link(plant, body_a)
        ids_b = _collision_geometry_ids_for_link(plant, body_b)
        if not ids_a or not ids_b:
            continue
        declaration.ExcludeBetween(GeometrySet(ids_a), GeometrySet(ids_b))
    filter_manager.Apply(declaration)


def _create_table_box_sdf(size_x: float, size_y: float, size_z: float):
    """Create or reuse an SDF file describing a thin table box.

    Written to a process-local temp directory so the retargeter does not depend
    on the current working directory being writable.
    """
    out_dir = os.path.join(tempfile.gettempdir(), "regrind_retarget_tables")
    os.makedirs(out_dir, exist_ok=True)
    file_name = f"table_box_{size_x:.3f}_{size_y:.3f}_{size_z:.3f}.sdf"
    sdf_path = os.path.join(out_dir, file_name)
    if os.path.exists(sdf_path):
        return sdf_path

    sdf_content = f"""<sdf version='1.6'>
  <model name='table'>
    <link name='table_link'>
      <pose>0 0 0 0 0 0</pose>
      <visual name='visual'>
        <pose frame=''>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
          </box>
        </geometry>
        <material>
          <diffuse>0.55 0.35 0.2 1.0</diffuse>
        </material>
      </visual>
      <collision name='collision'>
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
    with open(sdf_path, "w") as f:
        f.write(sdf_content)
    return sdf_path


def create_plant(
    robot_model_path: str,
    object_model_path: str | list[str] | None = None,
    collision_exclusion_setter: callable = None,
    table_height: float | None = None,
    table_size_xy: tuple[float, float] = (1.0, 1.0),
    table_position_xy: tuple[float, float] = (0.0, 0.0),
    table_top_thickness: float = 0.02,
):
    """Create a Drake plant with the robot hand and optional object + tabletop.

    When ``table_height`` is set, it is the world z coordinate of the **top** of the
    table surface; a thin box of height ``table_top_thickness`` is welded just below it.
    ``collision_exclusion_setter`` is the per-hand collision-filter setup callable
    (e.g. :func:`_setup_leaphand_collision_exclusions`).
    """
    builder = DiagramBuilder()
    plant = MultibodyPlant(1e-3)
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    parser.AddModels(robot_model_path)
    if object_model_path is not None:
        if isinstance(object_model_path, list):
            for model_path in object_model_path:
                parser.AddModels(model_path)
        else:
            parser.AddModels(object_model_path)

    # Optionally add a fixed tabletop (thin slab at z ≈ table_height).
    if table_height is not None:
        if table_height <= 0:
            raise ValueError(f"table_height must be positive, got {table_height}")
        table_x, table_y = table_size_xy
        if table_x <= 0 or table_y <= 0:
            raise ValueError(
                f"table_size_xy entries must be positive, got {table_size_xy}"
            )
        if table_top_thickness <= 0:
            raise ValueError(
                f"table_top_thickness must be positive, got {table_top_thickness}"
            )
        thickness = min(table_top_thickness, table_height)
        z_tabletop_center = table_height - thickness / 2

        table_model_path = _create_table_box_sdf(table_x, table_y, thickness)
        parser.AddModels(table_model_path)
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("table_link"),
            RigidTransform(
                np.array(
                    [
                        table_position_xy[0],
                        table_position_xy[1],
                        z_tabletop_center,
                    ]
                )
            ),
        )

    if collision_exclusion_setter is not None:
        collision_exclusion_setter(plant, scene_graph)
    plant.Finalize()
    return plant, scene_graph, builder


# ---------------------------------------------------------------------------
# Interaction mesh & Laplacian coordinates
# ---------------------------------------------------------------------------
def create_interaction_mesh(vertices: np.ndarray):
    """Create a tetrahedral mesh from points using Delaunay triangulation.

    Returns ``(vertices, tetrahedra)``.
    """
    tri = Delaunay(vertices)
    return vertices, tri.simplices


def get_adjacency_list(tetrahedra, num_vertices):
    """Creates an adjacency list from the tetrahedra."""
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_laplacian_coordinates(
    vertices,
    adj_list,
    epsilon=1e-6,
    uniform_weight=True,
    neighbor_vertex_weights=None,
):
    """Calculate the Laplacian coordinate (vertex minus weighted neighbor centroid) per vertex.

    ``neighbor_vertex_weights`` is an optional length-N non-negative weighting; each neighbor
    ``j`` contributes proportionally to ``neighbor_vertex_weights[j]``. Must match
    :func:`calculate_laplacian_matrix` for ``L @ v`` to agree with these coordinates.
    """
    laplacian = np.zeros_like(vertices)
    n = len(vertices)

    for i in range(n):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            vi = vertices[i]
            neighbor_positions = vertices[neighbors_indices]
            distances = np.linalg.norm(vi - neighbor_positions, axis=1)

            if uniform_weight:
                weights = np.ones(len(neighbors_indices), dtype=float)
            else:
                weights = 1.0 / (1.5 * distances + epsilon)

            if neighbor_vertex_weights is not None:
                weights = weights * neighbor_vertex_weights[neighbors_indices]

            sum_of_weights = np.sum(weights)
            if sum_of_weights <= epsilon:
                weights = np.ones(len(neighbors_indices), dtype=float)
                sum_of_weights = np.sum(weights)

            weighted_sum_of_neighbors = np.sum(
                weights[:, np.newaxis] * neighbor_positions, axis=0
            )
            center_of_neighbors = weighted_sum_of_neighbors / sum_of_weights
            laplacian[i] = vi - center_of_neighbors

    return laplacian


def calculate_laplacian_matrix(
    vertices,
    adj_list,
    epsilon=1e-6,
    uniform_weight=True,
    neighbor_vertex_weights=None,
):
    """Calculate the (N, N) Laplacian matrix matching :func:`calculate_laplacian_coordinates`."""
    N = len(vertices)
    laplacian_matrix = np.zeros((N, N))

    for i in range(N):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            if uniform_weight:
                raw = np.ones(len(neighbors_indices), dtype=float)
            else:
                vi = vertices[i]
                neighbor_positions = vertices[neighbors_indices]
                distances = np.linalg.norm(vi - neighbor_positions, axis=1)
                raw = 1.0 / (distances + epsilon)

            if neighbor_vertex_weights is not None:
                raw = raw * neighbor_vertex_weights[neighbors_indices]

            sum_weights = np.sum(raw)
            if sum_weights <= epsilon:
                raw = np.ones(len(neighbors_indices), dtype=float)
                sum_weights = np.sum(raw)
            weights = raw / sum_weights

            laplacian_matrix[i, i] = 1.0

            for j, neighbor_idx in enumerate(neighbors_indices):
                laplacian_matrix[i, neighbor_idx] = -weights[j]

    return laplacian_matrix


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------
def _quaternion_matrix(quat):
    """4x4 homogeneous matrix from a scalar-first (w, x, y, z) quaternion."""
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return transform


def transform_points_local_to_world(quat, trans, points_local):
    """Transform points from local frame to world frame. ``quat`` is scalar-first (w, x, y, z)."""
    transform_matrix = _quaternion_matrix(quat)
    transform_matrix[:3, 3] = trans
    hom_points = np.hstack([points_local, np.ones((points_local.shape[0], 1))])
    transformed_points_hom = (transform_matrix @ hom_points.T).T
    return transformed_points_hom[:, :3]


def transform_points_world_to_local(quat, trans, points_world):
    """Transform points from world frame to local frame. ``quat`` is scalar-first (w, x, y, z)."""
    transform_matrix = _quaternion_matrix(quat)
    transform_matrix[:3, 3] = trans
    inverse_transform_matrix = np.linalg.inv(transform_matrix)

    hom_points = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    transformed_points_hom = (inverse_transform_matrix @ hom_points.T).T
    return transformed_points_hom[:, :3]


def rotate_points_along_axis(points, axis, angle):
    """Rotate points about an axis by ``angle`` radians (right-hand rule).

    ``axis`` is one of ``{'x','y','z'}`` or a 3D vector. Returns same shape as input.
    """
    pts = np.asarray(points, dtype=float)
    original_shape = pts.shape
    pts_2d = pts.reshape(-1, 3)

    # Determine axis unit vector
    if isinstance(axis, str):
        ax = axis.lower()
        if ax == "x":
            axis_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        elif ax == "y":
            axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        elif ax == "z":
            axis_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            raise ValueError("axis must be one of {'x','y','z'} or a 3D vector")
    else:
        axis_arr = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis_arr)
        if norm == 0:
            raise ValueError("axis vector must be non-zero")
        axis_vec = axis_arr / norm

    # Build rotation from axis-angle and apply (row-vector convention)
    rotvec = axis_vec * float(angle)
    rot_mat = R.from_rotvec(rotvec).as_matrix()
    rotated = pts_2d @ rot_mat.T
    return rotated.reshape(original_shape)
