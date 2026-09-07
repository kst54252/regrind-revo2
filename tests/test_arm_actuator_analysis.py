"""Topology and threshold statistics must not manufacture saturation evidence."""
import unittest
import numpy as np
from tools.arm_diagnostics.analyze_arm_actuator import intervals, proximity, raw_path_speed


class ArmActuatorAnalysisTest(unittest.TestCase):
    def test_raw_angle_jump_is_not_wrapped(self):
        q=np.array([[3.13],[-3.13]])
        speed=raw_path_speed(q,np.array([3.12]),.01)
        np.testing.assert_allclose(speed[:,0],[1.,-626.])

    def test_longest_near_interval_uses_full_step_duration(self):
        result=intervals(np.array([1,1,0,1,1,1],bool),.1,np.arange(1,7)*.1)
        self.assertAlmostEqual(result["fraction"],5/6)
        self.assertAlmostEqual(result["longest_s"],.3)
        self.assertAlmostEqual(result["start_s"],.3)
        self.assertAlmostEqual(result["end_s"],.6)

    def test_asymmetric_limits(self):
        result=proximity(np.array([-9.6,18.8,-8.,19.1]),-10,20,.1,np.arange(1,5)*.1)
        self.assertEqual(result["fraction"],.5)
        self.assertEqual(result["longest_s"],.1)

    def test_unbounded_or_zero_limit_not_meaningful(self):
        for lower,upper in ((-np.inf,np.inf),(0,0),(-10,np.inf)):
            self.assertIsNone(proximity(np.zeros(3),lower,upper,.1,np.arange(1,4)*.1))
