"""Offline scan preserves frame, bounded-joint and failure-report contracts."""
import unittest
import numpy as np

from tools.arm_diagnostics.measure_tabletop_region import solve_placement
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics


class TabletopRegionTest(unittest.TestCase):
    def test_shifted_reference_recovers_same_world_wrist(self):
        mount = dict(position=[0., 0., -.02], quaternion_xyzw=[0., 0., 0., 1.])
        kin = RB3730Kinematics(base_position=mount['position'])
        q = np.array([.35, -.75, 1.05, .4, -.55, .2])
        pos, quat = kin.forward(q)
        # Translating a source scene by the scan delta must recover the known
        # mounted-wrist pose, not the flange pose or a double-applied offset.
        source_origin = np.array([.4, 0., .02])
        xy = np.array([.5, -.1])
        delta = np.r_[xy-source_origin[:2], 0.]
        row, result, errors = solve_placement((xy, np.tile(pos-delta, (3, 1)),
            np.tile(quat, (3, 1)), source_origin, q, mount, 100, 1/30))
        self.assertTrue(row['full_sequence_ik'])
        self.assertEqual(row['evaluated_frames'], 3)
        self.assertEqual(row['first_failed_frame'], -1)
        self.assertLess(row['max_reference_path_speed_rad_s'], 1e-6)
        np.testing.assert_allclose(result, np.tile(q, (3, 1)), atol=1e-7)
        self.assertLess(np.max(errors), 1e-7)

    def test_unsolved_does_not_claim_later_frames_were_tested(self):
        mount = dict(position=[0., 0., -.02], quaternion_xyzw=[0., 0., 0., 1.])
        row, q, errors = solve_placement(([.4, 0.], np.tile([4., 0., 4.], (3, 1)),
            np.tile([0., 0., 0., 1.], (3, 1)), np.array([.4, 0., .02]),
            np.zeros(6), mount, 10, 1/30))
        self.assertFalse(row['full_sequence_ik'])
        self.assertEqual(row['evaluated_frames'], 1)
        self.assertEqual(row['first_failed_frame'], 0)
        self.assertIsNone(row['max_reference_path_speed_rad_s'])
        self.assertTrue(np.isnan(q[1:]).all())
        self.assertTrue(np.isnan(errors[1:]).all())


if __name__ == '__main__':
    unittest.main()
