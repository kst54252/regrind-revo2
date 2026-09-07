"""Execute the mounted scheduling methods without an Isaac runtime."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


class Decoder:
    def process_actions(self, actions):
        self.target_pos=actions[:, :3].clone()
        self.target_quat=torch.tensor([[0., 0., 0., 1.]])


class TestPolicyRateIK(unittest.TestCase):
    def make_action(self, policy_rate):
        path=Path(__file__).resolve().parents[1]/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/simple_mounted_interface.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name=='SimpleMountedWrist')
        cls.body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('process_actions','apply_actions','reset_from_reference')]
        def response(p,q,target,rotation,dt,tau):
            return p+(1-math.exp(-dt/tau))*(target-p),rotation.clone()
        scope={'torch':torch,'SE3ImpedanceActionTerm':Decoder,'response_step':response}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
        a=scope['SimpleMountedWrist']()
        a.goal=torch.zeros(1,6);a.applied=a.goal.clone();a.device='cpu';a.ids=list(range(6))
        a.response_at_physics=True;a.ik_policy_rate=policy_rate;a.velocity_path=True;a._ik_pending=False
        a.target_pos=torch.zeros(1,3);a.target_quat=torch.tensor([[0.,0.,0.,1.]])
        a.ik_target_pos=a.target_pos.clone();a.ik_target_quat=a.target_quat.clone()
        a.cfg=NS(response_tau=.1,command_name='reference')
        reference=NS(target_hand_wrist_pos=a.target_pos.clone(),target_hand_wrist_quat=a.target_quat.clone())
        a._env=NS(physics_dt=1/120,step_dt=1/30,command_manager=NS(get_term=lambda _:reference))
        a.limits=torch.tensor([[[-3.,3.]]*6]);a.speed_limit=torch.full((1,6),10.)
        a.sent=[];a.calls=[];a.fail=False
        a._asset=NS(data=NS(joint_pos=NS(torch=torch.zeros(1,6))),
            set_joint_position_target_index=lambda **kw:a.sent.append(kw['target'].clone()),
            set_joint_velocity_target_index=lambda **kw:None,write_joint_state_to_sim=lambda *args,**kw:None)
        def solve(pos,quat,previous):
            a.calls.append(pos.clone())
            return NS(success=not a.fail,finite=True,joint_limit_violation=False,q=[.01]*6)
        a.solve=solve
        return a

    def test_one_solve_per_policy_but_response_and_delivery_every_physics_step(self):
        a=self.make_action(True)
        for _ in range(3):
            a.process_actions(torch.ones(1,6))
            for step in range(4):
                a.apply_actions()
                self.assertEqual(a.ik_updated_this_step,step==0)
        self.assertEqual(len(a.calls),3)
        self.assertEqual(len(a.sent),12)
        self.assertEqual(a.tracking_ik_count,3)
        self.assertEqual(a.physics_apply_count,12)
        torch.testing.assert_close(a.ik_target_pos,torch.full((1,3),1-math.exp(-1)))

    def test_default_still_solves_every_physics_step(self):
        a=self.make_action(False);a.process_actions(torch.ones(1,6))
        for _ in range(4):a.apply_actions()
        self.assertEqual(len(a.calls),4)

    def test_failed_ik_holds_goal_without_retrying_each_physics_step(self):
        a=self.make_action(True);a.fail=True;a.process_actions(torch.ones(1,6))
        for _ in range(4):a.apply_actions()
        self.assertEqual(len(a.calls),1)
        torch.testing.assert_close(a.goal,torch.zeros(1,6))

    def test_reset_discards_pending_solve_and_old_response(self):
        a=self.make_action(True);a.process_actions(torch.ones(1,6))
        a.reset_from_reference([0]);a.calls.clear();a.sent.clear()
        self.assertFalse(a._ik_pending)
        a.apply_actions();self.assertEqual(len(a.calls),0)
        a.process_actions(torch.ones(1,6));a.apply_actions()
        self.assertEqual(len(a.calls),1)

    def test_velocity_bounded_reset_uses_exact_ik_and_clears_velocity_history(self):
        a=self.make_action(False)
        a.bounded_kin=object();a.bounded_velocity=torch.full((1,6),7.)
        original=a.solve;flags=[]
        def solve(*args,**kwargs):
            flags.append(kwargs.get('bounded'))
            return original(*args)
        a.solve=solve
        a.reset_from_reference([0])
        self.assertEqual(flags,[False])
        torch.testing.assert_close(a.bounded_velocity,torch.zeros(1,6))


if __name__=='__main__':unittest.main()
