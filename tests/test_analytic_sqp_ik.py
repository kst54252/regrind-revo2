import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.rb3_revo2_ik.analytic_branch_ik import AnalyticBranchIK, periodic_lifts
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.sequence_branch_ik import minimax_path
from tools.rb3_revo2_ik.sqp_pose_ik import solve_sqp


class AnalyticBranchTest(unittest.TestCase):
    def setUp(self):
        self.kin = RB3730Kinematics(base_position=[0, 0, -.02])

    def test_random_fk_roundtrip_all_branches_and_periodic_limits(self):
        rng = np.random.default_rng(730)
        solver = AnalyticBranchIK(self.kin)
        for q in rng.uniform(self.kin.joint_lower, self.kin.joint_upper, size=(40,6)):
            result = solver.inverse_all(*self.kin.forward(q))
            self.assertTrue(result.exhaustive_isolated)
            self.assertEqual(result.geometric_count, 8)
            self.assertEqual(len(result.q), 256)
            self.assertLess(np.min(np.max(np.abs(result.q-q),axis=1)), 1e-7)
            self.assertTrue(np.all(result.q >= self.kin.joint_lower-1e-10))
            self.assertTrue(np.all(result.q <= self.kin.joint_upper+1e-10))

    def test_nonidentity_world_and_fixed_mount_rotation(self):
        k = RB3730Kinematics(base_position=[.1,-.2,.3],
                            base_quaternion_xyzw=Rotation.from_euler('xyz',[.2,.3,-.7]).as_quat())
        k.link6_to_wrist_rotation = Rotation.from_euler('xyz',[.2,-.4,.1]).as_matrix()
        k.link6_to_wrist_position = np.array([.01,-.02,.15])
        q = np.array([.4,-.6,1.8,-2.1,.6,3.9])
        result = AnalyticBranchIK(k).inverse_all(*k.forward(q))
        self.assertLess(np.min(np.max(np.abs(result.q-q),axis=1)), 1e-8)

    def test_unreachable_pose_is_empty_not_clamped_solution(self):
        result = AnalyticBranchIK(self.kin).inverse_all([10,0,0], [0,0,0,1])
        self.assertEqual(result.q.shape, (0,6))
        self.assertTrue(result.exhaustive_isolated)

    def test_exact_singular_family_does_not_claim_exhaustive_enumeration(self):
        for middle in (0., np.pi):
            q = np.array([.2,-.4,1.2,.6,middle,-.3])
            result = AnalyticBranchIK(self.kin).inverse_all(*self.kin.forward(q))
            self.assertFalse(result.exhaustive_isolated)
            self.assertTrue(any(x['kind']=='wrist_continuum' for x in result.singular_families))

    def test_near_singularity_is_not_rounded_to_a_continuum(self):
        q = np.array([.2,-.4,1.2,.6,1e-5,-.3])
        result = AnalyticBranchIK(self.kin).inverse_all(*self.kin.forward(q))
        self.assertTrue(result.exhaustive_isolated)
        self.assertLess(np.min(np.max(np.abs(result.q-q),axis=1)), 1e-7)

    def test_changed_geometry_is_rejected(self):
        k = copy.deepcopy(self.kin)
        k.joint_offsets[4,0] = .01
        with self.assertRaisesRegex(ValueError, 'offsets'):
            AnalyticBranchIK(k)

    def test_periodic_lifts_keep_real_bounded_coordinates(self):
        q = np.array([.2,.3,.4,.5,.6,.7])
        lifts = periodic_lifts(q, self.kin.joint_lower, self.kin.joint_upper)
        self.assertEqual(len(lifts), 32)
        self.assertTrue(any(row[0] < -6 for row in lifts))
        layers = [np.array([[3.1]*6]), np.array([[-3.1]*6, [3.2]*6])]
        path = minimax_path(layers, 1/120)
        np.testing.assert_allclose(path[-1], [3.2]*6)

    def test_bad_quaternion_rejected(self):
        with self.assertRaises(ValueError):
            AnalyticBranchIK(self.kin).inverse_all([.4,0,.1], [0]*4)


class SQPPoseTest(unittest.TestCase):
    def setUp(self):
        self.kin = RB3730Kinematics(base_position=[0,0,-.02])
        self.q = np.array([.1,-.3,1.5,.7,.8,-.4])

    def test_static_exact_target_is_accepted(self):
        p, r = self.kin.forward(self.q)
        result, info = solve_sqp(self.kin, p, r, self.q, np.zeros(6), 1/120, np.full(6,10.))
        self.assertTrue(info['accepted'], info)
        np.testing.assert_allclose(result, self.q, atol=1e-10)

    def test_small_motion_and_native_bounds(self):
        lower, upper = self.kin.get_joint_limits()
        target = self.q + np.array([.0002,-.0001,.0003,.0001,.0002,-.0003])
        p, r = self.kin.forward(target)
        result, info = solve_sqp(self.kin, p, r, self.q, np.zeros(6), 1/120, np.full(6,10.))
        self.assertTrue(info['accepted'], info)
        self.assertLessEqual(np.max(np.abs(result-self.q))*120**2, 250.+1e-6)
        np.testing.assert_array_equal(self.kin.joint_lower, lower)
        np.testing.assert_array_equal(self.kin.joint_upper, upper)

    def test_unreachable_target_is_not_reported_as_success(self):
        result, info = solve_sqp(self.kin, [9,0,0], [0,0,0,1], self.q, np.zeros(6),
                                1/120, np.full(6,10.), maxiter=15)
        self.assertFalse(info['accepted'])
        self.assertFalse(info['feasible'])
        self.assertTrue(info['box_feasible'])
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_timestep_and_velocity_rejected(self):
        p, r = self.kin.forward(self.q)
        for dt, speed in ((0.,np.ones(6)),(1/120,np.zeros(6))):
            with self.assertRaises(ValueError):
                solve_sqp(self.kin,p,r,self.q,np.zeros(6),dt,speed)


if __name__ == '__main__':
    unittest.main()
