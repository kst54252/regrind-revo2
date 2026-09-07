import ast
from pathlib import Path
import unittest
import numpy as np
import torch
from types import SimpleNamespace as NS
from tools.rb3_revo2_ik.recovery_replay import schedule,validate_unique_placements
from tools.rb3_revo2_ik.analyze_transfer_recovery import lag_samples


class TestTransferRecovery(unittest.TestCase):
    def test_zero_agent_bypasses_policy_with_twelve_zero_residuals(self):
        path=Path(__file__).resolve().parents[1]/'tools/rb3_revo2_ik/evaluate_mounted_interface.py'
        tree=ast.parse(path.read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.IfExp) and
                    isinstance(n.test,ast.Attribute) and n.test.attr=='zero_actions')
        self.assertEqual(branch.body.func.attr,'zeros')
        self.assertEqual(ast.literal_eval(branch.body.args[0]),(1,12))
        self.assertEqual(branch.orelse.func.id,'adapter')

    def test_lightweight_view_keeps_physics_updates_and_skips_diagnostics(self):
        path=Path(__file__).resolve().parents[1]/'tools/rb3_revo2_ik/evaluate_mounted_interface.py'
        tree=ast.parse(path.read_text())
        update=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update')
        self.assertEqual(update.body[0].value.func.id,'original_update')
        gate=next(n for n in update.body if isinstance(n,ast.If) and
                  isinstance(n.test,ast.Attribute) and n.test.attr=='realtime_view')
        self.assertIsInstance(gate.body[0],ast.Return)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('write_joint_state_to_sim','set_gravity') for n in ast.walk(update)))
        fps=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='RecordVideo')
        self.assertIn('env.step_dt',ast.unparse(fps))

    def test_heldout_bank_requires_distinct_actual_positions(self):
        with self.assertRaises(ValueError):validate_unique_placements([{'object_state':[.4,0,0]}]*20)
        xy=validate_unique_placements([{'object_state':[.4+i*.001,0,0]} for i in range(20)])
        self.assertEqual(xy.shape,(20,2))
    def test_velocity_matches_final_bounded_target_and_hold_is_zero(self):
        path=Path(__file__).resolve().parents[1]/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/simple_mounted_interface.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SimpleMountedWrist')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply_actions')
        scope={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
        sent={};a=NS(goal=torch.ones(1,6),applied=torch.zeros(1,6),speed_limit=torch.ones(1,6)*10,
            limits=torch.tensor([[[-3.,3.]]*6]),_env=NS(physics_dt=1/120),ids=list(range(6)),velocity_path=True,
            _asset=NS(set_joint_position_target_index=lambda **kw:sent.update(q=kw['target'].clone()),
                      set_joint_velocity_target_index=lambda **kw:sent.update(v=kw['target'].clone())))
        scope['apply_actions'](a)
        torch.testing.assert_close(sent['v'],torch.ones(1,6)*10)
        self.assertTrue(a.rate_limited)
        a.goal.copy_(a.applied)  # Reset/held accepted target has no old-episode derivative.
        scope['apply_actions'](a)
        torch.testing.assert_close(sent['v'],torch.zeros(1,6))

    def test_original_clock_is_not_shifted(self):
        rows=[dict(time_s=(i+1)/120) for i in range(8)]
        actual=schedule(rows)
        self.assertEqual([r for r,_ in actual],rows)
        self.assertTrue(all(hold is None for _,hold in actual))

    def test_physics_response_is_not_also_applied_at_policy_boundary(self):
        path=Path(__file__).resolve().parents[1]/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/simple_mounted_interface.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SimpleMountedWrist')
        process=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_actions')
        guard=next(n for n in process.body if isinstance(n,ast.If) and
                   isinstance(n.test,ast.Attribute) and n.test.attr=='response_at_physics')
        self.assertIsInstance(guard.body[0],ast.Return)
        apply=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply_actions')
        calls=[n for n in ast.walk(apply) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='response_step']
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].args[-2].attr,'physics_dt')

    def test_slow_diagnostic_only_repeats_samples(self):
        rows=[dict(index=i) for i in range(8)]
        actual=schedule(rows,speed=4)
        self.assertEqual([r['index'] for r,_ in actual],[i for i in range(8) for _ in range(4)])

    def test_five_static_holds_fixed_before_tuning(self):
        rows=[dict(index=i) for i in range(152)]
        actual=schedule(rows,holds=True)
        self.assertEqual(len(actual),5*4*120)
        for index in range(5):
            subset=[r['index'] for r,k in actual if k==index]
            self.assertEqual(len(set(subset)),1);self.assertEqual(len(subset),480)

    def test_lag_is_diagnostic_and_stationary_undefined(self):
        x=np.sin(np.linspace(0,8,100))[:,None];y=np.r_[np.repeat(x[:1],3,axis=0),x[:-3]]
        self.assertEqual(lag_samples(x,y,.01)['delay_s'],.03)
        self.assertIsNone(lag_samples(np.zeros((100,1)),np.zeros((100,1)),.01))

    def test_replay_does_not_call_policy_or_write_states(self):
        path=Path(__file__).resolve().parents[1]/'tools/rb3_revo2_ik/recovery_replay.py'
        tree=ast.parse(path.read_text())
        attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertNotIn('process_actions',attrs)
        self.assertFalse(any('write_root' in k or 'write_joint_state' in k for k in attrs))
        self.assertNotIn('policy',[n.id for n in ast.walk(tree) if isinstance(n,ast.Name)])


if __name__=='__main__':unittest.main()
