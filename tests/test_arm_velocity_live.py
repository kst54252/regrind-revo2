"""Exercise actual action methods without importing/starting Isaac extensions."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from tools.rb3_revo2_ik.paired_arm_states import STATE_KEYS, check_state


class LiveVelocityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/rb3_revo2_actions.py"
        tree = ast.parse(path.read_text())
        action = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RB3WristIKAction")
        methods = [n for n in action.body if isinstance(n, ast.FunctionDef) and n.name in ("apply_actions", "reset_from_reference")]
        scope = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
        cls.methods = scope

    def make_action(self, mode="v_path"):
        sent = []
        asset = NS(set_joint_position_target_index=lambda **kw:sent.append(("p", kw["target"].clone(), kw["joint_ids"])),
                   set_joint_velocity_target_index=lambda **kw:sent.append(("v", kw["target"].clone(), kw["joint_ids"])),
                   write_joint_state_to_sim=lambda *a, **kw:None,
                   data=NS(joint_pos=NS(torch=torch.zeros(1, 17))))
        cmd = NS(target_hand_wrist_pos=torch.zeros(1,3), target_hand_wrist_quat=torch.tensor([[0.,0.,0.,1.]]))
        a = NS(cfg=NS(velocity_target_mode=mode, interpolation_substeps=4, command_name="reference",
                      position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3, position_weight=10., max_nfev=300),
               _asset=asset, _joint_ids=list(range(6)), device="cpu", _env=NS(physics_dt=.01, command_manager=NS(get_term=lambda _:cmd)),
               _interpolation_step=0, _interpolation_start_target=torch.zeros(1,6), _last_joint_target=torch.ones(1,6)*.4,
               _applied_joint_target=torch.zeros(1,6), target_pos=torch.zeros(1,3), target_quat=torch.zeros(1,4),
               _ik_success=torch.zeros(1,dtype=torch.bool), _warm_start_valid=torch.zeros(1,dtype=torch.bool),
               _ik_position_error=torch.zeros(1), _ik_orientation_error=torch.zeros(1))
        return a, sent

    def test_each_substep_uses_final_position_difference(self):
        a,sent = self.make_action()
        for _ in range(4): self.methods["apply_actions"](a)
        velocities = [x[1] for x in sent if x[0]=="v"]
        torch.testing.assert_close(torch.cat(velocities), torch.ones(4,6)*10.)
        self.assertTrue(all(x[2]==list(range(6)) for x in sent))

    def test_zero_mode_keeps_original_no_velocity_write(self):
        a,sent = self.make_action("zero")
        self.methods["apply_actions"](a)
        self.assertEqual([x[0] for x in sent], ["p"])

    def test_joint_winding_is_not_wrapped(self):
        a,sent = self.make_action()
        a.cfg.interpolation_substeps=1
        a._applied_joint_target[:]=3.13
        a._last_joint_target[:]=-3.13
        self.methods["apply_actions"](a)
        torch.testing.assert_close(sent[-1][1], torch.full((1,6),-626.))

    def test_reference_reset_reseeds_history_and_clears_velocity(self):
        a,sent = self.make_action()
        a._applied_joint_target[:]=10.
        a._kinematics=NS(inverse=lambda *args,**kw:NS(q=np.ones(6)*.2, success=True, finite=True,
                                                    joint_limit_violation=False, position_error_m=0., orientation_error_rad=0.))
        self.methods["reset_from_reference"](a,[0])
        torch.testing.assert_close(a._applied_joint_target, torch.ones(1,6)*.2)
        torch.testing.assert_close(sent[0][1], torch.zeros(1,6))
        a._last_joint_target[:]=.6
        a._interpolation_step=0
        self.methods["apply_actions"](a)
        torch.testing.assert_close(sent[-1][1], torch.ones(1,6)*10.)


class PairedStateTest(unittest.TestCase):
    def test_all_joints_and_velocities_checked(self):
        state={k:[0.] for k in STATE_KEYS};state["reference_frame"]=0
        self.assertTrue(all(v==0 for v in check_state(state,state).values()))
        for key in STATE_KEYS:
            bad=copy.deepcopy(state);bad[key][0]=.01
            with self.assertRaisesRegex(ValueError,key):check_state(bad,state)

    def test_reference_phase_checked(self):
        state={k:[0.] for k in STATE_KEYS};state["reference_frame"]=0
        with self.assertRaisesRegex(ValueError,"frame"):
            check_state(dict(state,reference_frame=1),state)
