import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.arm_diagnostics.compare_singularity_methods import (
    bounded_path, evaluate_path, load_episode,
)
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics


class SingularityMethodsTest(unittest.TestCase):
    def setUp(self):
        self.kin = RB3730Kinematics(base_position=[0, 0, -.02])
        self.q0 = np.array([.1, -.3, 1.5, .7, .8, -.4])

    def fixture(self, directory):
        directory = Path(directory)
        # Native array deliberately differs from the six-joint model order.
        names = ['hand'] + list(self.kin.joint_names)
        p, quat = self.kin.forward(self.q0)
        header = dict(base_position=[0,0,-.02], base_quaternion_xyzw=[0,0,0,1])
        bank = directory/'bank.jsonl'
        bank.write_text(json.dumps(header)+'\n')
        meta = dict(initial_states=[dict(episode=11, all_q=[.2]+self.q0.tolist(),
                    object_state=[.4,0,.0126,0,0,0,1,0,0,0,0,0,0])],
                    state_bank=str(bank), physics_dt=1/120, control_dt=1/30,
                    joint_names=names, arm_ids=list(range(1,7)),
                    ik_policy_rate=False, arm_response_physics=True,
                    recovery_runtime=dict(velocity_limit=[100]+[10]*6))
        (directory/'metadata.json').write_text(json.dumps(meta))
        rows = [dict(episode=11, episode_step=i, time_s=i/120,
                     command_time_s=((i-1)//4)/30, ik_input_pos=p.tolist(),
                     ik_input_quat=quat.tolist(), raw_solve=dict(q=self.q0.tolist(), success=True),
                     q_cmd=self.q0.tolist(), hand_target=[0.]*6) for i in range(1,9)]
        (directory/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return rows

    def test_loader_distinguishes_policy_time_from_physics_command_time(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            _, data = load_episode(directory, 11)
            np.testing.assert_allclose(data['command_t'], np.arange(8)/120, atol=1e-15)
            np.testing.assert_allclose(data['policy_t'], np.repeat([0, 1/30], 4))
            np.testing.assert_array_equal(data['q'][0], self.q0)
            np.testing.assert_array_equal(data['speed'], [10.]*6)

    def test_missing_physics_sample_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.fixture(directory)
            del rows[3]
            (Path(directory)/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'timesteps'):
                load_episode(directory, 11)

    def test_zero_quaternion_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.fixture(directory)
            rows[0]['ik_input_quat'] = [0.]*4
            (Path(directory)/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'quaternions'):
                load_episode(directory, 11)

    def test_raw_solver_joint_order_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            p = Path(directory)/'metadata.json'
            meta = json.loads(p.read_text())
            meta['arm_ids'] = list(reversed(meta['arm_ids']))
            p.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'indexing'):
                load_episode(directory, 11)

    def test_path_metrics_do_not_wrap_large_joint_change(self):
        q = np.tile(self.q0, (3,1))
        q[:,5] = [3.1, -3.1, -3.0]
        p, r = self.kin.forward_batch(q)
        result, _ = evaluate_path(self.kin, q, p, r, np.arange(3)/120, np.full(6,10.), [], -100)
        self.assertAlmostEqual(result['max_step_rad'], 6.2)
        self.assertFalse(result['velocity_ok'])
        self.assertTrue(result['strict_pose'])
        self.assertFalse(result['screened_pass'])

    def test_failed_path_is_not_smooth_success(self):
        p, r = self.kin.forward_batch(np.tile(self.q0, (3,1)))
        result, arrays = evaluate_path(self.kin, np.full((3,6),np.nan), p, r,
                                      np.arange(3)/120, np.full(6,10.), [], -100)
        self.assertFalse(result['complete'])
        self.assertFalse(result['screened_pass'])
        self.assertEqual(arrays, {})

    def test_pose_error_not_erased_by_slow_or_stationary_solution(self):
        q = np.tile(self.q0, (3,1))
        p, r = self.kin.forward_batch(q)
        p[:,0] += .01
        result, _ = evaluate_path(self.kin, q, p, r, np.arange(3)/120, np.full(6,10.), [], -100)
        self.assertAlmostEqual(result['max_position_error_mm'], 10.)
        self.assertFalse(result['strict_pose'])
        self.assertFalse(result['screened_pass'])

    def test_bounded_path_reset_velocity_and_original_limits_preserved(self):
        q = np.tile(self.q0, (5,1)) + np.arange(5)[:,None]*.001
        p, r = self.kin.forward_batch(q)
        lower, upper = self.kin.get_joint_limits()
        solved, failures = bounded_path(self.kin, p, r, self.q0, 1/120, np.full(6,10.), 100, 250.)
        self.assertEqual(failures, [])
        np.testing.assert_array_equal(solved[0], self.q0)
        velocity = np.vstack([np.zeros(6), np.diff(solved,axis=0)*120])
        self.assertLessEqual(np.abs(np.diff(velocity,axis=0)*120).max(), 250.+1e-6)
        np.testing.assert_array_equal(self.kin.joint_lower, lower)
        np.testing.assert_array_equal(self.kin.joint_upper, upper)


if __name__ == '__main__':
    unittest.main()
