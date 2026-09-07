import unittest

import numpy as np

from tools.arm_diagnostics.analyze_arm_tracking import (
    analyze_arrays,
    estimate_integer_delay,
    quaternion_error_xyzw,
)


class ArmTrackingAnalysisTest(unittest.TestCase):
    def test_estimate_integer_delay_from_joint_velocity(self):
        rng = np.random.default_rng(4)
        target = np.cumsum(rng.normal(size=(80, 6)), axis=0)
        actual = np.vstack((np.repeat(target[:1], 3, axis=0), target[:-3]))

        delay = estimate_integer_delay(target, actual, dt=1.0 / 30.0, max_lag_steps=6)

        self.assertEqual(delay.steps, 3)
        self.assertTrue(np.isclose(delay.seconds, 0.1))

    def test_quaternion_error_handles_sign_equivalence(self):
        identity = np.asarray([[0.0, 0.0, 0.0, 1.0]])
        np.testing.assert_allclose(quaternion_error_xyzw(-identity, identity), 0.0)

    def test_tracking_report_shapes_and_finite_values(self):
        frames = 12
        target_q = np.zeros((frames, 6))
        actual_q = np.full((frames, 6), 0.01)
        identity = np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1))
        data = {
            "target_rb3_joints": target_q,
            "actual_rb3_joints": actual_q,
            "target_wrist_pos": np.zeros((frames, 3)),
            "actual_wrist_pos": np.tile([0.001, 0.0, 0.0], (frames, 1)),
            "target_wrist_quat_xyzw": identity,
            "actual_wrist_quat_xyzw": identity,
            "dt": np.asarray(1.0 / 30.0),
        }

        report = analyze_arrays(data)

        self.assertEqual(report["frames"], frames)
        self.assertTrue(np.isclose(report["wrist_position_mean_m"], 0.001))
        self.assertTrue(np.isfinite(report["joint_rmse_rad"]).all())


if __name__ == "__main__":
    unittest.main()
