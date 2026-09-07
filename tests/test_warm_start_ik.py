import unittest
import numpy as np
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.warm_start_ik import WarmStartIK,PoseJacobian
from scipy.spatial.transform import Rotation


class TestWarmStartIK(unittest.TestCase):
    def setUp(self):
        self.base=RB3730Kinematics(verify_model_hash=False)
        self.fast=WarmStartIK(self.base)

    def test_smooth_target_uses_existing_solver_and_same_solution(self):
        q=np.array([.2,-.6,1.,.4,.3,-.2]);p,r=self.base.forward(q)
        result=self.fast.inverse(p,r,initial_q=q+.002,neutral_q=q+.003)
        expected=self.base.inverse(p,r,initial_q=q+.002,neutral_q=q+.003)
        self.assertTrue(result.success)
        self.assertEqual(result.candidates_evaluated,1)
        np.testing.assert_allclose(result.q,expected.q,atol=1e-7,rtol=0)
        self.assertEqual(self.fast.fallbacks,0)

    def test_large_jump_uses_full_fallback_without_wrapping(self):
        q=np.array([.2,-.6,1.,.4,.3,-.2]);p,r=self.base.forward(q)
        previous=q.copy();previous[0]+=.3
        result=self.fast.inverse(p,r,initial_q=previous,neutral_q=previous)
        self.assertTrue(result.success)
        self.assertEqual(self.fast.fallbacks,1)
        self.assertGreater(result.candidates_evaluated,1)

    def test_unreachable_is_not_reported_as_success(self):
        result=self.fast.inverse([10,0,0],[0,0,0,1],initial_q=np.zeros(6),neutral_q=np.zeros(6),max_nfev=10)
        self.assertFalse(result.success)
        self.assertEqual(self.fast.fallbacks,1)

    def test_baseline_seed_selection_is_not_modified(self):
        q=np.array([.2,-.6,1.,.4,.3,-.2])
        self.assertGreater(len(self.base._candidate_seeds(q,q)),1)
        self.assertEqual(len(self.fast.local._candidate_seeds(q,q)),1)

    def test_jacobian_matches_central_difference_and_verified_residual(self):
        rng=np.random.default_rng(123)
        for _ in range(20):
            q=rng.uniform(-1,1,6);p,quat=self.base.forward(q+rng.uniform(-.3,.3,6))
            rotation=Rotation.from_quat(quat).as_matrix();d=PoseJacobian(self.base)
            value,jac=d.evaluate(q,p,rotation,10.)
            np.testing.assert_allclose(value,self.base._residual(q,p,rotation,10.),atol=2e-14)
            numeric=np.column_stack([(self.base._residual(q+np.eye(6)[j]*1e-6,p,rotation,10.)-
                                      self.base._residual(q-np.eye(6)[j]*1e-6,p,rotation,10.))/(2e-6) for j in range(6)])
            np.testing.assert_allclose(jac,numeric,atol=3e-8,rtol=1e-7)


if __name__=='__main__':unittest.main()
