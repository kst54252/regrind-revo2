import ast
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
import torch
from tools.rb3_revo2_ik.frozen_policy_adapter import FrozenPolicyAdapter


class TestFrozenAdapter(unittest.TestCase):
    def make_policy(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.normalizer=torch.nn.BatchNorm1d(67)
                self.actor=torch.nn.Linear(67,12)
            def forward(self,obs):return self.actor(self.normalizer(obs['policy']))
        return Policy().eval()

    def test_same_observation_and_frozen_action(self):
        torch.manual_seed(42)
        policy=self.make_policy();adapter=FrozenPolicyAdapter(policy)
        obs={'policy':torch.randn(1,67)}
        before=obs['policy'].clone()
        for _ in range(5):
            torch.testing.assert_close(adapter(obs),policy(obs),rtol=0,atol=0)
        torch.testing.assert_close(obs['policy'],before,rtol=0,atol=0)
        adapter.assert_frozen()

    def test_actual_state_changes_are_not_replaced_by_targets(self):
        adapter=FrozenPolicyAdapter(self.make_policy())
        a={'policy':torch.zeros(1,67)};b={'policy':a['policy'].clone()}
        b['policy'][0,9:15]=1. # Actual wrist position history changes.
        self.assertFalse(torch.equal(adapter(a),adapter(b)))

    def test_nonfinite_and_shape_rejected(self):
        adapter=FrozenPolicyAdapter(self.make_policy())
        for obs in (torch.zeros(1,6),torch.full((1,67),float('nan'))):
            with self.assertRaises(ValueError):adapter({'policy':obs})

    def test_shared_decoder_and_no_step_teleport(self):
        path=Path(__file__).resolve().parents[1]/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/simple_mounted_interface.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SimpleMountedWrist')
        self.assertEqual(cls.bases[0].id,'SE3ImpedanceActionTerm')
        for method in [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('process_actions','apply_actions')]:
            names=[n.attr for n in ast.walk(method) if isinstance(n,ast.Attribute)]
            self.assertFalse(any('write_joint_state' in n or 'write_root' in n for n in names))
        process=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_actions')
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='process_actions' for n in ast.walk(process)),1)

    def test_physical_state_observation_terms_are_reused(self):
        # Execute the real maintained observation functions without booting Kit.
        root=Path(__file__).resolve().parents[1]
        path=root/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/observations.py'
        names={'_remove_command_translation','_maybe_apply_delay','object_pos','hand_wrist_pos','hand_joint_pos'}
        tree=ast.parse(path.read_text())
        methods=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        scope={'torch':torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+methods,type_ignores=[])),str(path),'exec'),scope)
        state=NS(current_object_pos=torch.tensor([[.45,.1,.02]]),current_hand_wrist_pos=torch.tensor([[.4,.1,.15]]),
                 current_hand_joint_pos=torch.arange(6,dtype=torch.float32)[None],default_hand_joint_pos=torch.ones(1,6)*.1,
                 observation_translation_offset=torch.tensor([[.05,.1,0.]]))
        env=NS(command_manager=NS(get_term=lambda _:state))
        def legacy_observation():
            obs=torch.zeros(1,67)
            obs[:,:3]=scope['object_pos'](env,'reference')
            wrist=scope['hand_wrist_pos'](env,'reference')
            hand=scope['hand_joint_pos'](env,'reference',relative_to_default=True)
            obs[:,9:15]=torch.cat([wrist,wrist],dim=1)
            obs[:,27:39]=torch.cat([hand,hand],dim=1)
            return {'policy':obs}
        policy=self.make_policy();adapter=FrozenPolicyAdapter(policy)
        old=legacy_observation();new=legacy_observation()
        torch.testing.assert_close(old['policy'],new['policy'],rtol=0,atol=0)
        torch.testing.assert_close(policy(old),adapter(new),rtol=0,atol=0)
        torch.testing.assert_close(new['policy'][0,:3],torch.tensor([.4,0.,.02]))
        adapter.assert_frozen()

    def test_target_slew_limit_without_velocity_feedforward(self):
        path=Path(__file__).resolve().parents[1]/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/simple_mounted_interface.py'
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SimpleMountedWrist')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply_actions')
        scope={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
        sent={}
        a=NS(goal=torch.ones(1,6),applied=torch.zeros(1,6),speed_limit=torch.ones(1,6)*10,
             limits=torch.tensor([[[-3.,3.]]*6]),_env=NS(physics_dt=1/120),ids=list(range(6)),
             _asset=NS(set_joint_position_target_index=lambda **kw:sent.update(q=kw['target'].clone()),
                       set_joint_velocity_target_index=lambda **kw:sent.update(v=kw['target'].clone())))
        scope['apply_actions'](a)
        torch.testing.assert_close(sent['q'],torch.ones(1,6)/12)
        torch.testing.assert_close(sent['v'],torch.zeros(1,6));self.assertTrue(a.rate_limited)
