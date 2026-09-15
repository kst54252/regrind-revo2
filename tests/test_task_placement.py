import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tools.rb3_revo2_ik.task_placement import TaskPlacement, place_reference, ShuffledPlacementCycle
from tools.arm_diagnostics.evaluate_semicircle import semicircle_grid, heldout_points

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp'
tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)


class TaskPlacementTests(unittest.TestCase):
    def test_live_cycle_covers_all_55_without_consuming_rng_on_probe(self):
        bank={i:dict(placement_id=i) for i in range(55)}
        cycle=ShuffledPlacementCycle(bank,55,42)
        orders=[]
        for k in range(3):
            order=[cycle[k*55+i]['placement_id'] for i in range(55)]
            self.assertEqual(sorted(order),list(range(55)))
            self.assertEqual(cycle[k*55+3],cycle[k*55+3])
            orders.append(order)
        self.assertNotEqual(orders[0],orders[1])
        self.assertNotEqual(orders[1],orders[2])

    def test_live_cycle_rejects_missing_states(self):
        with self.assertRaises(ValueError):ShuffledPlacementCycle({0:0},2,42)
        with self.assertRaises(ValueError):ShuffledPlacementCycle({},0,42)

    def setUp(self):
        self.frame = TaskPlacement(tensor([.4, 0, .012636]), 67.)
        self.offset = tensor([[.1, -.3, 0]])

    def test_point_history_and_fingertip_roundtrip(self):
        for shape in ((1, 3), (1, 2, 3), (1, 5, 3)):
            p = torch.randn(shape, dtype=torch.float64)
            torch.testing.assert_close(self.frame.canonical_position(
                self.frame.world_position(p, self.offset), self.offset), p)

    def test_orientation_left_rotation_xyzw(self):
        q = tensor(Rotation.from_euler('xyz', [[.2, .3, .4]]).as_quat())
        actual = self.frame.world_quat(q)
        expected = (Rotation.from_euler('z',67,degrees=True)*Rotation.from_quat(q.numpy())).as_quat()
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        torch.testing.assert_close(self.frame.canonical_quat(actual), q)

    def test_actual_observation_functions_return_same_values(self):
        tree = ast.parse((MDP/'observations.py').read_text())
        names = {'_remove_command_translation','_canonical_quat','object_pos','object_ori',
                 'hand_wrist_pos','hand_wrist_rot6d','action_base_wrist_pos_and_rot6d','fingertips_pos'}
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        scope = dict(torch=torch,ManagerBasedEnv=object,MotionCommand=object,
                     _maybe_apply_delay=lambda env,key,data:data,
                     matrix_from_quat=lambda q:tensor(Rotation.from_quat(q.numpy()).as_matrix()))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual observation functions>','exec'),scope)
        p=tensor([[.4,.05,.13]]);q=tensor(Rotation.from_euler('xyz',[[.2,.3,.4]]).as_quat())
        outputs=[]
        for placement in (None,self.frame):
            pp=p if placement is None else placement.world_position(p,self.offset)
            qq=q if placement is None else placement.world_quat(q)
            command=NS(current_object_pos=pp,current_object_quat=qq,current_hand_wrist_pos=pp,
                       current_hand_wrist_quat=qq,current_fingertips_pos=pp[:,None].repeat(1,5,1),
                       observation_translation_offset=torch.zeros_like(self.offset) if placement is None else self.offset,
                       task_placement=placement)
            env=NS(command_manager=NS(get_term=lambda name:command),num_envs=1,
                   action_manager=NS(get_term=lambda name:NS(get_base_pose=lambda:(pp,qq))))
            result=[scope[n](env,command_name='reference') for n in
                    ('object_pos','object_ori','hand_wrist_pos','hand_wrist_rot6d',
                     'action_base_wrist_pos_and_rot6d','fingertips_pos')]
            scope['hand_wrist_pos'](env,'reference',delay_key='wrist')
            torch.testing.assert_close(env.delayed_hand_wrist_pos,pp)
            outputs.append(result)
        for a,b in zip(*outputs):torch.testing.assert_close(a,b)

    def test_real_action_decoder_clips_before_rotating(self):
        tree=ast.parse((MDP/'actions.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SE3ImpedanceActionTerm')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_actions')
        scope=dict(torch=torch,_rotvec_to_quat=lambda r:tensor(Rotation.from_rotvec(r.numpy()).as_quat()),
                   quat_mul=lambda a,b:tensor((Rotation.from_quat(a.numpy())*Rotation.from_quat(b.numpy())).as_quat()))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual decoder>','exec'),scope)
        p=tensor([[.3,.02,.2]]);q=tensor([[0,0,0,1]]);results=[]
        action=tensor([[2,-.7,.3,-.3,1.8,.4]])
        for placement in (None,self.frame):
            base=(p,q) if placement is None else (placement.world_position(p,self.offset),placement.world_quat(q))
            obj=NS(cfg=NS(raw_clip=(-1,1),scale_pos=1/30,scale_rot=3.2/30,command_name='reference'),
                   _raw_actions=torch.zeros_like(action),_processed_actions=torch.zeros_like(action),
                   _get_base_pose=lambda:base,task_placement=placement)
            scope['process_actions'](obj,action)
            results.append(obj)
        torch.testing.assert_close(results[1].target_pos,self.frame.world_position(results[0].target_pos,self.offset))
        torch.testing.assert_close(results[1].target_quat,self.frame.world_quat(results[0].target_quat))
        torch.testing.assert_close(results[1]._raw_actions,results[0]._raw_actions)
        torch.testing.assert_close(results[1]._processed_actions,results[0]._processed_actions)

    def test_reset_never_accumulates_yaw(self):
        c=NS(num_envs=1,cfg=NS(joint_reference='revo2'),reference_joint_pos=torch.ones((2,6)))
        for body in ('object','wrist'):
            for field,value in [('pos',tensor([[.4,0,.01],[.4,.1,.1]])),
                                ('quat',tensor([[0,0,0,1],[0,0,0,1]])),
                                ('lin_vel',torch.ones((2,3),dtype=torch.float64)),
                                ('ang_vel',torch.ones((2,3),dtype=torch.float64))]:
                setattr(c,'reference_'+body+'_'+field,value.clone())
        before=c.reference_wrist_pos.clone()
        place_reference(c,90);place_reference(c,-35);place_reference(c,0)
        torch.testing.assert_close(c.reference_wrist_pos,before)

    def test_zero_yaw_keeps_old_translation(self):
        f=TaskPlacement(self.frame.center,0.)
        p=tensor([[.5,.1,.3]])
        torch.testing.assert_close(f.canonical_position(p,self.offset),p-self.offset)

    def test_fixed_spatial_denominator_and_holdout(self):
        p=semicircle_grid(.75)
        self.assertEqual(p.shape,(35,2))
        self.assertTrue(np.all(np.linalg.norm(p,axis=1)<=.75+1e-10))
        self.assertTrue(np.all(p[:,0]>=.3-1e-10))
        h=heldout_points(.75)
        self.assertEqual(h.shape,(20,2))
        np.testing.assert_array_equal(h,heldout_points(.75))
        self.assertTrue(np.all(np.linalg.norm(h,axis=1)<=.75))


if __name__=='__main__':unittest.main()
