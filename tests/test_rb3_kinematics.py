"""Regression tests for the extracted RB3-730 kinematic model."""

from __future__ import annotations

import unittest

import numpy as np

from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics


class RB3730KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RB3730Kinematics()

    def test_model_order_limits_and_mount(self):
        self.assertEqual(
            self.robot.joint_names,
            ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"),
        )
        lower, upper = self.robot.get_joint_limits()
        self.assertEqual(lower.shape, (6,))
        self.assertEqual(upper.shape, (6,))
        self.assertTrue(np.all(lower < upper))
        np.testing.assert_allclose(
            self.robot.link6_to_wrist_position, (0.0, 0.0, 0.141304972), atol=1.0e-12
        )

    def test_zero_configuration_fk(self):
        position, quaternion = self.robot.forward(np.zeros(6))
        np.testing.assert_allclose(position, (0.0, -0.00645, 0.916604972), atol=1.0e-12)
        np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1.0e-12)

    def test_ik_recovers_reachable_pose(self):
        expected_q = np.array((0.35, -0.75, 1.05, 0.4, -0.55, 0.2))
        target_position, target_quaternion = self.robot.forward(expected_q)
        result = self.robot.inverse(
            target_position,
            target_quaternion,
            initial_q=np.zeros(6),
            position_tolerance_m=1.0e-6,
            orientation_tolerance_rad=1.0e-6,
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.finite)
        self.assertFalse(result.joint_limit_violation)
        self.assertLessEqual(result.position_error_m, 1.0e-6)
        self.assertLessEqual(result.orientation_error_rad, 1.0e-6)

    def test_batch_fk_matches_scalar_fk(self):
        q_batch = np.array(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.2, -0.6, 0.9, 0.3, -0.4, 0.7),
                (-1.1, 0.8, -1.2, -0.2, 1.0, -0.5),
            )
        )
        position_batch, quaternion_batch = self.robot.forward_batch(q_batch)
        for index, q in enumerate(q_batch):
            position, quaternion = self.robot.forward(q)
            np.testing.assert_allclose(position_batch[index], position, atol=1.0e-12)
            # Quaternion signs can differ while representing the same rotation.
            self.assertAlmostEqual(abs(np.dot(quaternion_batch[index], quaternion)), 1.0)

    def test_position_only_ik(self):
        expected_q = np.array((0.2, -0.5, 0.8, 0.25, -0.35, 0.45))
        target_position = self.robot.forward(expected_q)[0]
        result = self.robot.inverse_position(
            target_position,
            initial_q=np.zeros(6),
            additional_seeds=expected_q[None],
            position_tolerance_m=1.0e-7,
        )
        self.assertTrue(result.success, result.message)
        self.assertLessEqual(result.position_error_m, 1.0e-7)
        self.assertFalse(result.joint_limit_violation)

    def test_chain_endpoint_matches_fk(self):
        q = np.array((0.4, -0.7, 1.0, 0.2, -0.3, 0.6))
        chain = self.robot.forward_chain_points(q)
        self.assertEqual(chain.shape, (5, 3))
        np.testing.assert_allclose(chain[-1], self.robot.forward(q)[0], atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
