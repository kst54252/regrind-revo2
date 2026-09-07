"""Response recurrence contracts; contact efficacy requires the Isaac comparison."""
import ast
import math
from pathlib import Path
import unittest
import torch
import numpy as np
from scipy.spatial.transform import Rotation


class TestMountedResponse(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'regrind/source/regrind/regrind/tasks/manager_based/dexterous/mdp/simple_mounted_interface.py'
        self.tree=ast.parse(path.read_text())
        method=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='response_step')
        # Independent quaternion oracle; runtime uses existing Isaac helpers.
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
        scope=dict(math=math,
            quat_mul=lambda a,b:tensor((Rotation.from_quat(a.numpy())*Rotation.from_quat(b.numpy())).as_quat()),
            quat_conjugate=lambda a:tensor(Rotation.from_quat(a.numpy()).inv().as_quat()),
            _quat_positive_real=lambda a:torch.where(a[...,3:]<0,-a,a),
            axis_angle_from_quat=lambda a:tensor(Rotation.from_quat(a.numpy()).as_rotvec()),
            _rotvec_to_quat=lambda a:tensor(Rotation.from_rotvec(a.numpy()).as_quat()))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
        self.step=scope['response_step']
        self.p=torch.zeros((1,3),dtype=torch.float64)
        self.q=tensor([[0,0,0,1]])

    def test_disabled_preserves_target_exactly(self):
        target=torch.ones_like(self.p)
        p,q=self.step(None,None,target,-self.q,1/30,0)
        torch.testing.assert_close(p,target,rtol=0,atol=0)
        torch.testing.assert_close(q,-self.q,rtol=0,atol=0)
        self.assertNotEqual(p.data_ptr(),target.data_ptr())

    def test_zoh_response_and_shortest_rotation(self):
        target=torch.ones_like(self.p)
        rot=torch.tensor(Rotation.from_euler('z',350,degrees=True).as_quat()[None])
        p,q=self.step(self.p,self.q,target,rot,1/30,.1)
        alpha=1-math.exp(-1/3)
        torch.testing.assert_close(p,target*alpha)
        np.testing.assert_allclose(Rotation.from_quat(q.numpy()).as_rotvec()[0,2],math.radians(-10)*alpha,atol=1e-12)
        torch.testing.assert_close(target,torch.ones_like(target))

    def test_invalid_times(self):
        for dt,tau in [(0,.1),(1/30,-1),(1/30,float('nan'))]:
            with self.assertRaises(ValueError):self.step(self.p,self.q,self.p,self.q,dt,tau)

    def test_reset_history_and_observation_separation(self):
        cls=next(n for n in self.tree.body if isinstance(n,ast.ClassDef) and n.name=='SimpleMountedWrist')
        reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset_from_reference')
        assigned=[n.attr for n in ast.walk(reset) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)]
        self.assertIn('ik_target_pos',assigned);self.assertIn('ik_target_quat',assigned)
        # Shaping only changes IK input; the decoder equilibrium remains logged.
        process=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_actions')
        assigned=[n.attr for n in ast.walk(process) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)]
        self.assertNotIn('target_pos',assigned);self.assertNotIn('target_quat',assigned)


if __name__=='__main__':unittest.main()
