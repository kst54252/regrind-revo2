import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tools.arm_diagnostics.compare_analytic_sqp import sqp_path
from tools.arm_diagnostics.scan_tabletop_sqp import assess_path, table_grid, select_reset
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics

ROOT=Path(__file__).resolve().parents[1]


class TabletopSQPTest(unittest.TestCase):
    def setUp(self):
        self.kin=RB3730Kinematics(base_position=[0,0,-.02])
        self.q=np.array([.1,-.3,1.5,.7,.8,-.4])
        self.layout=json.loads((ROOT/'config/workcell/rb3_revo2_table.json').read_text())

    def test_grid_keeps_full_mesh_footprint_inside_table(self):
        v=np.array([[-.043,-.043,-.01],[.043,.043,.02]])
        xy,lo,hi,fmin,fmax=table_grid(self.layout,v,[0,0,0,1],.02)
        self.assertEqual(len(xy),2700)
        self.assertTrue(np.all(xy+fmin>=lo-1e-10))
        self.assertTrue(np.all(xy+fmax<=hi+1e-10))
        np.testing.assert_allclose(xy.min(0),[.30,-.74])
        np.testing.assert_allclose(xy.max(0),[1.,.74])

    def test_same_reset_is_selected_at_source_xy(self):
        q,info=select_reset(self.kin,*self.kin.forward(self.q),self.q,[],-100)
        np.testing.assert_allclose(q,self.q,atol=1e-9)
        self.assertEqual(info['status'],'reset_ready')

    def test_unreachable_reset_never_has_a_fake_configuration(self):
        q,info=select_reset(self.kin,[10,0,0],[0,0,0,1],self.q,[],-100)
        self.assertIsNone(q)
        self.assertEqual(info['status'],'reset_unsolved')

    def test_pose_feasible_but_nonconverged_is_not_green(self):
        q=np.tile(self.q,(3,1));p,r=self.kin.forward_batch(q)
        records=[dict(frame=i,feasible=True,accepted=False,status=8) for i in (1,2)]
        row,_=assess_path(self.kin,q,p,r,records,1/120,np.full(6,10.),[],-100,1e-4,1e-3)
        self.assertEqual(row['status'],'optimizer_failure')
        self.assertTrue(row['constraints_met'])

    def test_unmeasured_suffix_cannot_pass(self):
        q=np.tile(self.q,(3,1));p,r=self.kin.forward_batch(q);q[2]=np.nan
        records=[dict(frame=1,feasible=False,accepted=False,status=8)]
        row,errors=assess_path(self.kin,q,p,r,records,1/120,np.full(6,10.),[],-100,.005,.05)
        self.assertEqual(row['status'],'constraint_failure')
        self.assertFalse(row['full_path_evaluated'])
        self.assertTrue(np.isnan(errors[2]).all())

    def test_grid_stop_option_preserves_comparison_default(self):
        q=np.tile(self.q,(4,1));p,r=self.kin.forward_batch(q)
        data=dict(q=q,p=p,quat=r,dt=1/120,speed=np.full(6,10.),command_t=np.arange(3)/120)
        info=dict(feasible=False,accepted=False,box_feasible=True,status=8)
        with patch('tools.arm_diagnostics.compare_analytic_sqp.solve_sqp',return_value=(self.q,info.copy())):
            stopped,records=sqp_path(self.kin,data,self.q,stop_on_infeasible=True)
            self.assertEqual(len(records),1)
            self.assertTrue(np.isnan(stopped[2:]).all())
        with patch('tools.arm_diagnostics.compare_analytic_sqp.solve_sqp',side_effect=lambda *a,**k:(self.q,info.copy())):
            continued,records=sqp_path(self.kin,data,self.q)
            self.assertEqual(len(records),3)
            self.assertTrue(np.isfinite(continued).all())


if __name__=='__main__':
    unittest.main()
