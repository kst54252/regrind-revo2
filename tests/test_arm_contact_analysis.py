"""Guard the paired experiment against silent command/timing/config changes."""
import copy
import unittest

from tools.rb3_revo2_ik.analyze_arm_contact import verify_pair


class ContactPairTest(unittest.TestCase):
    def setUp(self):
        self.meta = dict(condition="present", source="record.jsonl", episode=15,
                         physics_dt=1/120, joint_names=["arm", "finger"],
                         actual_stiffness=[300, 3], actual_damping=[20, .1],
                         actual_effort_limits=[10, .5], actual_velocity_limits=[10, 100])
        self.off_meta = dict(self.meta, condition="absent")
        self.on = [dict(time_s=1/120, source_physics_step=2200, reference_frame=1,
                        interpolation_step=1, all_position_targets=[.2, .3],
                        all_velocity_targets=[0, 0], all_effort_targets=[0, 0],
                        backend_position_targets=[.2, .3])]
        self.off = copy.deepcopy(self.on)

    def test_identical_pair(self):
        verify_pair(self.meta,self.on,self.off_meta,self.off)

    def test_changed_finger_command_rejected(self):
        self.off[0]["all_position_targets"][1] += .001
        with self.assertRaisesRegex(ValueError,"all_position_targets"):
            verify_pair(self.meta,self.on,self.off_meta,self.off)

    def test_changed_runtime_gain_rejected(self):
        self.off_meta["actual_damping"] = [25, .1]
        with self.assertRaisesRegex(ValueError,"actual_damping"):
            verify_pair(self.meta,self.on,self.off_meta,self.off)

    def test_changed_timestamp_rejected(self):
        self.off[0]["source_physics_step"] += 1
        with self.assertRaisesRegex(ValueError,"source_physics_step"):
            verify_pair(self.meta,self.on,self.off_meta,self.off)
