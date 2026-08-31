"""Tests for camera-to-world rigid trajectory conversion."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.dexycb_world_transform.transform_trajectory import transform_trajectory


class TransformTrajectoryTest(unittest.TestCase):
    def test_first_object_and_rigid_geometry(self):
        object_pos = np.array(((0.1, 0.2, 0.3), (0.12, 0.2, 0.31)))
        object_quat = np.array(((1.0, 0.0, 0.0, 0.0),) * 2)
        wrist_pos = np.array(((0.2, 0.3, 0.4), (0.21, 0.3, 0.4)))
        wrist_quat = object_quat.copy()
        mano = np.zeros((2, 21, 3))
        mano[:, :, 0] = np.linspace(0.0, 0.2, 21)
        vertices = np.array(((-0.04, -0.04, -0.02), (0.04, 0.04, 0.03)))

        result = transform_trajectory(
            object_pos, object_quat, wrist_pos, wrist_quat, mano, vertices
        )
        np.testing.assert_allclose(result["object_pos_world"][0], (0.5, 0.0, 0.02))
        self.assertAlmostEqual(float(result["first_mesh_world_min_z"]), 0.0)
        self.assertAlmostEqual(float(result["R_world_camera_determinant"]), 1.0)
        before = np.linalg.norm(mano[1, 20] - wrist_pos[1])
        after = np.linalg.norm(
            result["mano_joint_world"][1, 20] - result["wrist_pos_world"][1]
        )
        self.assertAlmostEqual(before, after)

    def test_rejects_non_upright_desired_orientation(self):
        with self.assertRaisesRegex(ValueError, "must be upright"):
            transform_trajectory(
                np.zeros((1, 3)),
                np.array(((1.0, 0.0, 0.0, 0.0),)),
                np.zeros((1, 3)),
                np.array(((1.0, 0.0, 0.0, 0.0),)),
                np.zeros((1, 21, 3)),
                np.array(((0.0, 0.0, -0.1),)),
                desired_object_quat_wxyz=(0.70710678, 0.70710678, 0.0, 0.0),
            )

    def test_model_frame_y90_maps_observed_x_motion_to_world_z(self):
        object_pos = np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))
        identity = np.array(((1.0, 0.0, 0.0, 0.0),) * 2)
        result = transform_trajectory(
            object_pos,
            identity,
            object_pos.copy(),
            identity.copy(),
            np.zeros((2, 21, 3)),
            np.array(((-0.04, -0.04, -0.02), (0.04, 0.04, 0.03))),
            object_model_frame_rpy_deg=(0.0, 90.0, 0.0),
        )
        displacement = (
            result["object_pos_world"][1] - result["object_pos_world"][0]
        )
        np.testing.assert_allclose(displacement, (0.0, 0.0, 0.1), atol=1.0e-12)
        np.testing.assert_allclose(
            result["object_quat_world"][0], (1.0, 0.0, 0.0, 0.0), atol=1.0e-12
        )

    def test_dexycb_camera_up_is_independent_of_object_orientation(self):
        object_pos = np.array(((0.1, 0.2, 0.8), (0.1, 0.1, 0.8)))
        # Local +Z points camera-down in this upright annotation.  Camera -Y
        # must nevertheless remain world +Z for the whole scene.
        rotation = Rotation.from_euler("X", 90.0, degrees=True)
        quaternion_xyzw = rotation.as_quat()
        quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
        object_quat = np.repeat(quaternion_wxyz[None], 2, axis=0)
        vertices = np.array(((-0.04, -0.04, -0.02), (0.04, 0.04, 0.03)))
        result = transform_trajectory(
            object_pos,
            object_quat,
            object_pos.copy(),
            object_quat.copy(),
            np.zeros((2, 21, 3)),
            vertices,
            camera_frame_convention="dexycb_y_down",
            world_yaw_deg=35.0,
        )
        displacement = result["object_pos_world"][1] - result["object_pos_world"][0]
        np.testing.assert_allclose(displacement, (0.0, 0.0, 0.1), atol=1.0e-12)
        np.testing.assert_allclose(result["object_pos_world"][0, :2], (0.5, 0.0))
        self.assertAlmostEqual(float(result["first_mesh_world_min_z"]), 0.0)


if __name__ == "__main__":
    unittest.main()
