"""Recorded velocity targets must retain source step and named command alignment."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from tools.arm_diagnostics.analyze_arm_velocity import load_velocity_path, sign_changes


class ArmVelocityAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.source_meta = dict(physics_dt=.1, joint_names=["arm"], user_joint_names=["arm", "finger"])
        self.meta = dict(condition="absent", physics_dt=.1, arm_names=["arm"], joint_names=["finger", "arm"])
        self.source = [dict(physics_step=10, state_time_s=.1, all_position_targets=[.2, .3],
                            all_velocity_targets=[0., 0.], all_effort_targets=[0., 0.], previous_accepted_q=[.1])]
        self.rows = [dict(source_physics_step=10, time_s=.1, all_position_targets=[.3, .2],
                          all_velocity_targets=[0., 0.], all_effort_targets=[0., 0.],
                          actuator=dict(v_path=[1.], q_cmd=[.2]))]

    def load(self, rows=None, meta=None):
        with patch("tools.arm_diagnostics.analyze_arm_velocity.read_run", return_value=(meta or self.meta, rows or self.rows)):
            return load_velocity_path("unused.jsonl", self.source_meta, self.source)

    def test_named_mapping_and_reset_seed(self):
        np.testing.assert_array_equal(self.load(), [[1.]])

    def test_shifted_physics_step_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["source_physics_step"] += 1
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.load(rows)

    def test_changed_finger_position_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["all_position_targets"][0] += .01
        with self.assertRaises(AssertionError):
            self.load(rows)

    def test_wrong_derivative_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["actuator"]["v_path"] = [2.]
        with self.assertRaises(AssertionError):
            self.load(rows)

    def test_changed_arm_order_rejected(self):
        meta = dict(self.meta, arm_names=["other"])
        with self.assertRaisesRegex(ValueError, "order"):
            self.load(meta=meta)

    def test_intervention_is_not_baseline(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.load(meta=dict(self.meta, velocity_path_source="prior.jsonl"))

    def test_present_trace_requires_matching_condition(self):
        meta = dict(self.meta, condition="present")
        with self.assertRaisesRegex(ValueError, "matching"):
            self.load(meta=meta)
        with patch("tools.arm_diagnostics.analyze_arm_velocity.read_run", return_value=(meta, self.rows)):
            np.testing.assert_array_equal(load_velocity_path("unused", self.source_meta, self.source, "present"), [[1.]])

    def test_deadband_crossings(self):
        self.assertEqual(sign_changes(np.array([.02, .001, -.001, .03, -.02]), .005), 1)
        self.assertEqual(sign_changes(np.array([.001, -.001]), .005), 0)
