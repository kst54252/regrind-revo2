"""No simulator needed: diagnostic reset, topology and comparison contracts."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
from tools.rb3_revo2_ik.analyze_ik120 import differences, paired_initials


class IK120AnalysisTest(unittest.TestCase):
    def test_differences_use_reset_sample_and_never_wrap_bounded_joints(self):
        path,command_acc,actual_acc=differences(np.array([[3.],[-3.]]),
            np.array([[1.],[2.]]),np.array([2.9]),np.array([0.]),.1)
        np.testing.assert_allclose(path[:,0],[1.,-60.])
        np.testing.assert_allclose(command_acc[:,0],[10.,-610.])
        np.testing.assert_allclose(actual_acc[:,0],[10.,10.])
        again,_,_=differences(np.array([[0.]]),np.array([[0.]]),np.array([0.]),np.array([0.]),.1)
        np.testing.assert_array_equal(again,[[0.]])

    def test_candidates_change_only_wrist3_and_keep_120hz_response_contract(self):
        root=Path(__file__).resolve().parents[1]
        table=json.loads((root/'config/experiments/rb3_ik120_gain_candidates.json').read_text())
        original=json.loads((root/'config/experiments/rb3_precision_candidates.json').read_text())['c3']
        self.assertEqual(table['baseline'],original)
        self.assertEqual(len(table)-1,3)
        for name in ('c1','c2','c3'):
            for gain in ('kp','kd'):
                self.assertEqual(table[name][gain][:5],original[gain][:5])
                self.assertTrue(np.isfinite(table[name][gain]).all())
        selected=json.loads((root/'config/experiments/rb3_ik120_wrist3_candidate.json').read_text())
        self.assertEqual(selected['response_tau'],.1)
        self.assertTrue(selected['arm_response_physics'])
        self.assertTrue(selected['arm_velocity_path'])

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
