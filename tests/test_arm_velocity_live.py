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
                      transfer_reset_sync=False,
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

    def test_transfer_rsi_reset_clears_only_selected_env_buffers(self):
        a, _ = self.make_action()
        a.cfg.transfer_reset_sync = True
        command = a._env.command_manager.get_term('reference')
        command.cfg = NS(rsi_enabled=True)
        command.time_steps = torch.tensor([2, 1])
        command.reference_joint_vel = torch.arange(36.).reshape(3, 12) / 100
        command.target_hand_joint_pos = torch.full((2, 6), .15)
        command.target_hand_wrist_pos = torch.zeros(2, 3)
        command.target_hand_wrist_quat = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
        a._asset.data.joint_pos.torch = torch.zeros(2, 17)
        for name in ('_interpolation_start_target', '_last_joint_target', '_applied_joint_target',
                     'target_pos', 'target_quat', '_ik_success', '_warm_start_valid',
                     '_ik_position_error', '_ik_orientation_error'):
            value = getattr(a, name)
            setattr(a, name, value.repeat((2,) + (1,) * (value.ndim - 1)))
        a._raw_actions = torch.ones(2, 6)
        a._processed_actions = torch.ones(2, 6)
        hand = NS(_raw_actions=torch.ones(2, 6), _processed_actions=torch.ones(2, 6),
                  _last_joint_target=torch.ones(2, 6))
        a._env.action_manager = NS(get_term=lambda _: hand)
        a._kinematics = NS(inverse=lambda *args, **kw: NS(q=np.ones(6)*.2, success=True, finite=True,
            joint_limit_violation=False, position_error_m=0., orientation_error_rad=0.))
        written = []
        a._asset.write_joint_state_to_sim = lambda q, v, **kw: written.append((q.clone(), v.clone(), kw))
        untouched = {name: getattr(a, name)[1].clone() for name in (
            '_applied_joint_target', '_interpolation_start_target', '_last_joint_target', '_raw_actions', '_processed_actions')}
        self.methods['reset_from_reference'](a, [0])
        torch.testing.assert_close(written[0][1], command.reference_joint_vel[2:3, :6])
        self.assertEqual(written[0][2]['env_ids'].tolist(), [0])
        for name, value in untouched.items():
            torch.testing.assert_close(getattr(a, name)[1], value)
        for term in (a, hand):
            torch.testing.assert_close(term._raw_actions[0], torch.zeros(6))
            torch.testing.assert_close(term._processed_actions[0], torch.zeros(6))
            torch.testing.assert_close(term._processed_actions[1], torch.ones(6))
        torch.testing.assert_close(hand._last_joint_target[0], command.target_hand_joint_pos[0])
        torch.testing.assert_close(hand._last_joint_target[1], torch.ones(6))


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
