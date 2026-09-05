from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools" / "rb3_revo2_ik"
sys.path.insert(0, str(TOOLS_DIR))

from build_reference_trajectory import (  # noqa: E402
    _align_trajectory_to_object_start,
    _stable_upright_pose_on_table,
)


class FloatingAlignmentTest(unittest.TestCase):
    def test_leveling_preserves_axis_sign_and_places_mesh_above_table(self):
        source_rotation = Rotation.from_euler(
            "ZYX", [63.0, 11.0, 180.0], degrees=True
        )
        vertices = np.asarray(
            [
                [-0.04, -0.04, -0.021],
                [0.04, 0.04, 0.013],
                [0.04, -0.04, 0.013],
                [-0.04, 0.04, -0.021],
            ]
        )
        position, quaternion, tilt = _stable_upright_pose_on_table(
            np.asarray([0.4, -0.1, 0.02]),
            source_rotation.as_quat(),
            vertices,
            clearance=0.001,
        )
        rotation = Rotation.from_quat(quaternion).as_matrix()
        world_vertices = vertices @ rotation.T + position

        np.testing.assert_allclose(rotation[:2, 2], 0.0, atol=1.0e-12)
        self.assertEqual(np.sign(rotation[2, 2]), np.sign(source_rotation.as_matrix()[2, 2]))
        self.assertAlmostEqual(float(world_vertices[:, 2].min()), 0.001, places=12)
        self.assertAlmostEqual(np.degrees(tilt), 11.0, places=10)

    def test_object_wrist_share_one_rigid_transform(self):
        object_pos = np.asarray([[0.1, -0.2, 0.3], [0.12, -0.18, 0.35]])
        object_quat = Rotation.from_euler("z", [[20.0], [25.0]], degrees=True).as_quat()
        wrist_pos = object_pos + np.asarray([[0.04, 0.01, 0.08], [0.03, 0.02, 0.09]])
        wrist_quat = Rotation.from_euler("xyz", [[5.0, 10.0, 20.0], [7.0, 8.0, 25.0]], degrees=True).as_quat()
        desired_pos = np.asarray([0.5, 0.1, 0.02])
        desired_quat = Rotation.from_euler("z", -90.0, degrees=True).as_quat()

        aligned = _align_trajectory_to_object_start(
            wrist_pos,
            wrist_quat,
            object_pos,
            object_quat,
            desired_pos,
            desired_quat,
        )
        aligned_wrist_pos, aligned_wrist_quat, aligned_object_pos, aligned_object_quat, rotation, translation = aligned

        np.testing.assert_allclose(aligned_object_pos[0], desired_pos, atol=1.0e-12)
        self.assertLess(
            (Rotation.from_quat(aligned_object_quat[0]).inv() * Rotation.from_quat(desired_quat)).magnitude(),
            1.0e-12,
        )
        np.testing.assert_allclose(
            aligned_wrist_pos,
            rotation.apply(wrist_pos) + translation,
            atol=1.0e-12,
        )
        expected_wrist_rotation = rotation * Rotation.from_quat(wrist_quat)
        np.testing.assert_allclose(
            (Rotation.from_quat(aligned_wrist_quat).inv() * expected_wrist_rotation).magnitude(),
            0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
