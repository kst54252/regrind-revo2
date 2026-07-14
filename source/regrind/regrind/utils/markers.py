from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
import torch
from isaaclab.utils.math import normalize, quat_from_angle_axis


FRAME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        ),
        "connecting_line": sim_utils.CylinderCfg(
            radius=0.002,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
        ),
    }
)

MANO_JOINT_CFG = sim_utils.SphereCfg(
    radius=0.01,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
)

MANO_FINGERTIP_CFG = sim_utils.SphereCfg(
    radius=0.01,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
)

MANO_CONNECTING_LINE_CFG = sim_utils.CylinderCfg(
    radius=0.002,
    height=1.0,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
)

MANO_CONNECTING_LINE_INDICES = [
    (0, 13), (13, 14), (14, 15), (15, 16),  # thumb
    (0, 1), (1, 2), (2, 3), (3, 17),        # index
    (0, 4), (4, 5), (5, 6), (6, 18),        # middle
    (0, 10), (10, 11), (11, 12), (12, 19),  # ring
    (0, 7), (7, 8), (8, 9), (9, 20),        # pinky
]

MANO_TO_OCULUS_MAPPING = {
    # Wrist
    0: 0,
    # Thumb
    13: 3,
    14: 4,
    15: 5,
    16: 19,
    # Index
    1: 6,
    2: 7,
    3: 8,
    17: 21,
    # Middle
    4: 9,
    5: 10,
    6: 11,
    18: 21,
    # Ring
    10: 12,
    11: 13,
    12: 14,
    19: 22,
    # Little
    7: 16,
    8: 17,
    9: 18,
    20: 23,
}

MANO_TO_MANUS_MAPPING = {
    # Wrist
    0: 0,
    # Thumb
    13: 1,
    14: 2,
    15: 3,
    16: 4,
    # Index
    1: 6,
    2: 7,
    3: 8,
    17: 9,
    # Middle
    4: 11,
    5: 12,
    6: 13,
    18: 14,
    # Ring
    10: 16,
    11: 17,
    12: 18,
    19: 19,
    # Little
    7: 21,
    8: 22,
    9: 23,
    20: 24,
}

"""
Mano joint convention: https://github.com/zc-alexfan/arctic/blob/master/docs/data/mano_right.png
"""
MANO_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        **{f"joint_{i}": MANO_JOINT_CFG for i in range(16)},
        **{f"joint_{i}": MANO_FINGERTIP_CFG for i in range(16, 21)},
        **{f"connecting_line_{s}_{t}": MANO_CONNECTING_LINE_CFG for (s, t) in MANO_CONNECTING_LINE_INDICES}
    }
)


class ManoMarker:
    def __init__(self, prim_path: str, device: torch.device, scale: float = 1.0):
        self.mano_marker = VisualizationMarkers(MANO_MARKER_CFG.replace(prim_path=prim_path))
        self.device = device
        self.scale = scale

    def visualize(self, joint_pos_w: torch.Tensor):
        """
        Args:
            joint_pos_w: (n_envs, 21, 3) joint positions in world coordinates.
                See mano joint convention: https://github.com/zc-alexfan/arctic/blob/master/docs/data/mano_right.png
        """
        assert joint_pos_w.ndim == 3, "joint_pos_w must be of shape (n_envs, 21, 3)"
        n_envs = joint_pos_w.size(0)
        source_joint_indices = [x[0] for x in MANO_CONNECTING_LINE_INDICES]
        target_joint_indices = [x[1] for x in MANO_CONNECTING_LINE_INDICES]
        lines_pos, lines_quat, lines_length = self._get_connecting_lines(
            start_pos=joint_pos_w[..., source_joint_indices, :].view(-1, 3),
            end_pos=joint_pos_w[..., target_joint_indices, :].view(-1, 3),
        )

        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n_envs * 21, -1)
        marker_joint_indices = torch.arange(21, device=self.device).repeat(n_envs)
        marker_line_indices = torch.arange(21, 21 + len(MANO_CONNECTING_LINE_INDICES), device=self.device).repeat(n_envs)

        s = self.scale
        marker_scales = torch.ones((n_envs * 21 + lines_pos.size(0), 3), device=self.device)
        marker_scales[: n_envs * 21] *= s
        n_lines = lines_length.size(0)
        marker_scales[-n_lines:, 0] = s
        marker_scales[-n_lines:, 1] = s
        marker_scales[-n_lines:, 2] = lines_length

        self.mano_marker.visualize(
            translations=torch.cat((joint_pos_w.view(-1, 3), lines_pos), dim=0),
            orientations=torch.cat((default_quat, lines_quat), dim=0),
            scales=marker_scales,
            marker_indices=torch.cat((marker_joint_indices, marker_line_indices), dim=0),
        )

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adapted from IsaacLab's frame_transformer.py
        Given start and end points, compute the positions (mid-point), orientations, and lengths of the connecting lines.

        Args:
            start_pos: The start positions of the connecting lines. Shape is (N, 3).
            end_pos: The end positions of the connecting lines. Shape is (N, 3).

        Returns:
            positions: The position of each connecting line. Shape is (N, 3).
            orientations: The orientation of each connecting line in quaternion. Shape is (N, 4).
            lengths: The length of each connecting line. Shape is (N,).
        """
        direction = end_pos - start_pos
        lengths = torch.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2

        # Get default direction (along z-axis)
        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(start_pos.size(0), -1)

        # Normalize direction vector
        direction_norm = normalize(direction)

        # Calculate rotation from default direction to target direction
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1)

        # Handle case where vectors are parallel
        mask = rotation_axis_norm > 1e-6
        rotation_axis = torch.where(
            mask.unsqueeze(-1),
            normalize(rotation_axis),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(start_pos.size(0), -1),
        )

        # Calculate rotation angle
        cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        orientations = quat_from_angle_axis(angle, rotation_axis)

        return positions, orientations, lengths


class RGBAxesMarker:
    """Visualize RGB coordinate axes (X=red, Y=green, Z=blue) given SE(3) transformations.

    Uses FRAME_MARKER_CFG which contains a frame marker with RGB axes visualization.
    """

    def __init__(self, prim_path: str, device: torch.device, scale: float = 0.1):
        """
        Args:
            prim_path: Path where markers will be spawned
            device: torch device
            scale: Scale factor for the frame marker (default 0.1 = 10cm, larger than FRAME_MARKER_CFG default)
        """
        self.frame_marker = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path=prim_path))
        self.device = device
        self.scale = scale

    def visualize(self, positions: torch.Tensor, orientations: torch.Tensor):
        """
        Visualize RGB axes at given SE(3) transformations.

        Args:
            positions: (n_envs, 3) positions in world frame
            orientations: (n_envs, 4) quaternions (w, x, y, z) in world frame
        """
        assert positions.ndim == 2 and positions.shape[1] == 3, "positions must be (n_envs, 3)"
        assert orientations.ndim == 2 and orientations.shape[1] == 4, "orientations must be (n_envs, 4)"
        assert positions.shape[0] == orientations.shape[0], "positions and orientations must have same batch size"

        n_envs = positions.shape[0]

        # Use marker index 0 for the "frame" marker (which contains RGB axes)
        marker_indices = torch.zeros(n_envs, dtype=torch.long, device=self.device)

        # Scales: uniform scale for the frame marker
        scales = torch.full((n_envs, 3), self.scale, device=self.device)

        self.frame_marker.visualize(
            translations=positions,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )
