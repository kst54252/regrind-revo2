"""Regression tests for the standalone Revo2 semantic-keypoint FK."""

import unittest

import numpy as np

from tools.revo2_kinematics import Revo2Kinematics


class Revo2KinematicsTest(unittest.TestCase):
    def setUp(self):
        self.fk = Revo2Kinematics()

    def test_keypoints_have_expected_shape_order_and_finite_values(self):
        lower, upper = self.fk.get_joint_limits()
        q = 0.5 * (lower + upper)
        keypoints = self.fk.get_keypoints(q)

        self.assertEqual(keypoints.shape, (21, 3))
        self.assertTrue(np.isfinite(keypoints).all())
        self.assertEqual(self.fk.keypoint_names[0], "kp_00_wrist")
        self.assertEqual(self.fk.keypoint_names[-1], "kp_20_little_tip")

    def test_joint_limits_are_six_valid_intervals(self):
        lower, upper = self.fk.get_joint_limits()
        self.assertEqual(lower.shape, (6,))
        self.assertEqual(upper.shape, (6,))
        self.assertTrue(np.all(lower <= upper))

    def test_rejects_wrong_q_shape(self):
        with self.assertRaisesRegex(ValueError, r"shape \(6,\)"):
            self.fk.get_keypoints(np.zeros(5))

    def test_joint_motion_changes_non_base_keypoints(self):
        open_keypoints = self.fk.get_keypoints(np.zeros(6))
        _, upper = self.fk.get_joint_limits()
        bent_keypoints = self.fk.get_keypoints(0.5 * upper)

        # Wrist and the four non-thumb MCP keypoints are attached to the base.
        np.testing.assert_allclose(
            bent_keypoints[[0, 1, 4, 7, 10]],
            open_keypoints[[0, 1, 4, 7, 10]],
        )
        self.assertGreater(
            np.linalg.norm(bent_keypoints[17:] - open_keypoints[17:]),
            1.0e-3,
        )

    def test_mimic_joint_positions_follow_robot_constraints(self):
        q = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        actual = self.fk.get_mimic_joint_positions(q)
        expected = np.asarray([0.2, 1.155 * 0.3, 1.155 * 0.4, 1.155 * 0.5, 1.155 * 0.6])
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.shape, (5,))


if __name__ == "__main__":
    unittest.main()
