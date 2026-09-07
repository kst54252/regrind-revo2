import unittest
import numpy as np
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.velocity_bounded_ik import VelocityBoundedIK


class VelocityBoundedIKTest(unittest.TestCase):
    def setUp(self):
        self.kin=RB3730Kinematics()
        self.solver=VelocityBoundedIK(self.kin)

    def solve(self,center,target,**kwargs):
        p,q=self.kin.forward(target)
        return self.solver.inverse(p,q,command_q=np.array(center),velocity_limit=np.full(6,10.),
            dt=1/120,position_lower=self.kin.joint_lower,position_upper=self.kin.joint_upper,**kwargs)

    def test_near_singularity_avoids_large_jump_with_explicit_pose_error(self):
        center=np.array([.14989033,-.60698032,2.34896326,1.27172816,.01367873,-1.31694984])
        target=np.array([.15461960,-.60873842,2.34256172,.47355318,.01578924,-.51944846])
        result=self.solve(center,target)
        self.assertTrue(result.success)
        self.assertLessEqual(np.max(abs(result.q-center)),10/120+1e-9)
        self.assertLess(result.position_error_m,.001)
        self.assertLess(result.orientation_error_rad,.02)
        self.assertGreater(result.orientation_error_rad,1e-3)  # NOT strict-pose IK.

    def test_normal_small_motion_and_original_model_limits_preserved(self):
        center=np.array([.1,-.1,1.5,1.,.7,-1.])
        lower=self.kin.joint_lower.copy();upper=self.kin.joint_upper.copy()
        result=self.solve(center,center+.005)
        self.assertTrue(result.success)
        self.assertLess(result.position_error_m,1e-6)
        self.assertLess(result.orientation_error_rad,1e-6)
        np.testing.assert_array_equal(lower,self.kin.joint_lower)
        np.testing.assert_array_equal(upper,self.kin.joint_upper)

    def test_large_unreachable_target_is_not_reported_as_accepted(self):
        p,q=self.kin.forward(np.zeros(6))
        result=self.solver.inverse(p+10,q,command_q=np.zeros(6),velocity_limit=np.ones(6),
            dt=1/120,position_lower=self.kin.joint_lower,position_upper=self.kin.joint_upper)
        self.assertFalse(result.success)
        self.assertTrue(result.finite)
        self.assertLessEqual(np.max(abs(result.q)),1/120+1e-9)

    def test_invalid_period_rejected(self):
        with self.assertRaises(ValueError):
            self.solver.inverse(np.zeros(3),np.array([0,0,0,1]),command_q=np.zeros(6),
                velocity_limit=np.ones(6),dt=0,position_lower=self.kin.joint_lower,position_upper=self.kin.joint_upper)

    def test_acceleration_box_respects_previous_command_velocity_not_actual_velocity(self):
        center=np.array([.14989033,-.60698032,2.34896326,1.27172816,.01367873,-1.31694984])
        target=center.copy();target[3]-=.8;target[5]+=.8
        velocity=np.array([0.,0.,0.,5.,0.,-5.])
        result=self.solve(center,target,previous_velocity=velocity,acceleration_limit=250.)
        self.assertTrue(result.optimizer_success)
        step_velocity=(result.q-center)*120
        self.assertLessEqual(np.max(abs(step_velocity-velocity)*120),250.+1e-6)
        self.assertGreater(step_velocity[3],0.)  # Cannot reverse in one timestep.
        self.assertLess(step_velocity[5],0.)


if __name__=='__main__':unittest.main()
