import unittest
import numpy as np

from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.sequence_branch_ik import (
    centerline_clear, continue_path, minimax_path, path_score, segment_intersects_box, solve_local, workcell_boxes,
)


class SequenceBranchIKTest(unittest.TestCase):
    def test_minimax_uses_future_frames_not_greedy_first_seed(self):
        layers = [np.array([[0.]*6, [2.]*6]), np.array([[0.]*6, [3.]*6]), np.array([[4.]*6])]
        q = minimax_path(layers, 1.)
        np.testing.assert_array_equal(q[:, 0], [2, 3, 4])
        self.assertEqual(path_score(q, 1.)[0], 1.)

    def test_no_periodic_wrapping_hides_actual_jump(self):
        layers = [np.array([[3.1]*6]), np.array([[-3.1]*6])]
        q = minimax_path(layers, 1.)
        self.assertAlmostEqual(path_score(q, 1.)[0], 6.2)

    def test_existing_solver_is_unchanged_and_mount_stays_exact(self):
        kin = RB3730Kinematics(base_position=[0, 0, -.02])
        q = np.array([.2, -.6, 1., .4, .3, -.2])
        pos, quat = kin.forward(q)
        n = len(kin._candidate_seeds(q, q))
        result = solve_local(kin, pos, quat, q+.001)
        self.assertTrue(result.success)
        self.assertEqual(len(kin._candidate_seeds(q, q)), n)
        self.assertGreater(n, 1)
        p, r = kin.forward(result.q)
        np.testing.assert_allclose(p, pos, atol=1e-8)
        self.assertAlmostEqual(abs(np.dot(r, quat)), 1.)
        path = continue_path(kin, np.tile(pos, (3, 1)), np.tile(quat, (3, 1)), q)
        self.assertLess(path_score(path, 1/30)[0], 1e-6)

    def test_failed_path_does_not_score_as_smooth(self):
        self.assertEqual(path_score(np.full((3, 6), np.nan), 1/30)[0], float('inf'))

    def test_box_screen_detects_between_endpoints(self):
        self.assertTrue(segment_intersects_box([-2, 0, 0], [2, 0, 0], [-1]*3, [1]*3))
        self.assertFalse(segment_intersects_box([-2, 2, 0], [2, 2, 0], [-1]*3, [1]*3))
        self.assertTrue(segment_intersects_box([0, 0, 0], [0, 0, 0], [-1]*3, [1]*3))

    def test_graph_is_exact_over_small_candidate_set(self):
        import itertools
        rng = np.random.default_rng(14)
        layers = [rng.normal(size=(3, 6)) for _ in range(4)]
        actual = path_score(minimax_path(layers, .1), .1)[0]
        expected = min(path_score(np.array([layers[t][j] for t, j in enumerate(indices)]), .1)[0]
                       for indices in itertools.product(range(3), repeat=4))
        self.assertAlmostEqual(actual, expected)

    def test_actual_workcell_rejects_under_pedestal_elbow_branch(self):
        import json
        from pathlib import Path
        layout = json.loads((Path(__file__).resolve().parents[1]/'config/workcell/rb3_revo2_table.json').read_text())
        kin = RB3730Kinematics(base_position=layout['robot_mount']['position'])
        boxes = workcell_boxes(layout)
        q = np.deg2rad([156.6, -157.1, 141.7, -137.6, 99.3, -28.])
        self.assertFalse(centerline_clear(kin, q, boxes, layout['floor_z']))

    def test_empty_graph_and_nonpositive_dt_rejected(self):
        with self.assertRaises(ValueError):
            minimax_path([], .1)
        with self.assertRaises(ValueError):
            minimax_path([np.zeros((1, 6))], 0.)


if __name__ == '__main__':
    unittest.main()
