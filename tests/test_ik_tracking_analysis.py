"""No simulator needed: diagnostic reset, topology and comparison contracts."""
import copy
import unittest

import numpy as np
from tools.arm_diagnostics.analyze_ik_tracking import differences, paired_initials


class IKTrackingAnalysisTest(unittest.TestCase):
    def test_differences_use_reset_sample_and_never_wrap_bounded_joints(self):
        path,command_acc,actual_acc=differences(np.array([[3.],[-3.]]),
            np.array([[1.],[2.]]),np.array([2.9]),np.array([0.]),.1)
        np.testing.assert_allclose(path[:,0],[1.,-60.])
        np.testing.assert_allclose(command_acc[:,0],[10.,-610.])
        np.testing.assert_allclose(actual_acc[:,0],[10.,10.])
        again,_,_=differences(np.array([[0.]]),np.array([[0.]]),np.array([0.]),np.array([0.]),.1)
        np.testing.assert_array_equal(again,[[0.]])

    def test_paired_validation_rejects_history_or_initial_state_changes(self):
        keys=('checkpoint_sha256','reference','state_bank','physics_dt','control_dt','limits',
              'gravity','robot_spawn','response_tau_s','arm_velocity_path','arm_response_physics','ik_policy_rate')
        a=dict.fromkeys(keys,0)
        a.update(ends=[{}],joint_names=list('abcdefg'),arm_ids=list(range(6)),
                 gains={'kp':[1]*7,'kd':[1]*7},initial_states=[dict.fromkeys(
                 ('wrist_pos','wrist_quat','wrist_velocity','all_q','all_v','hand_q','hand_v',
                  'follower_q','follower_v','object_state','robot_root','phase'),0)])
        a['initial_states'][0]['reset_buffers']={'action':[0]}
        b=copy.deepcopy(a);b['gains']['kp'][5]=2
        paired_initials(a,b)
        b['initial_states'][0]['reset_buffers']['action']=[1]
        with self.assertRaisesRegex(ValueError,'history'):paired_initials(a,b)
        b=copy.deepcopy(a);b['initial_states'][0]['object_state']=1
        with self.assertRaises(AssertionError):paired_initials(a,b)
        b=copy.deepcopy(a);b['gains']['kp'][6]=2
        with self.assertRaises(AssertionError):paired_initials(a,b)


if __name__=='__main__':unittest.main()
