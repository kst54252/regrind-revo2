"""Regression checks for units and timestamp attribution in substep reports."""
import unittest
import numpy as np

from tools.arm_diagnostics.analyze_arm_execution import pose_errors, statistics


class ArmExecutionAnalysisTest(unittest.TestCase):
    def test_pose_error_is_euclidean_and_shortest_angle(self):
        p = np.zeros((1, 3))
        p2 = np.array([[0.03, 0.04, 0]])
        q = np.array([[0, 0, 0, 1]])
        q2 = -np.array([[0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]])
        position, angle = pose_errors(p, q, p2, q2)
        np.testing.assert_allclose(position, [0.05])
        np.testing.assert_allclose(angle, [np.pi/2])

    def test_masked_max_keeps_original_physics_timestamp(self):
        samples = [dict(episode=i//2, state_time_s=(i+1)/120, physics_step=i+100,
                        command_id=i, reference_frame_at_command=i+3, interpolation_step=4)
                   for i in range(4)]
        result = statistics(np.array([100., 2., 3., 50.]), samples, np.array([False, True, True, False]))
        self.assertEqual(result["max"], 3.)
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["at"]["physics_step"], 102)
        self.assertEqual(result["at"]["reference_frame"], 5)
        self.assertIsNone(statistics(np.ones(4), samples, np.zeros(4, dtype=bool)))
