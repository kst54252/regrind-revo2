import json
from pathlib import Path
import unittest
import numpy as np
from tools.rb3_revo2_ik.precision_trajectory import build, quintic, command_rows


class PrecisionTrajectoryTest(unittest.TestCase):
    def test_single_joint_diagnostic_is_smooth_and_does_not_mutate_source(self):
        from tools.rb3_revo2_ik.precision_trajectory import single_joint_config
        source=json.loads((Path(__file__).resolve().parents[1]/'config/experiments/rb3_precision_benchmark.json').read_text())
        original=json.dumps(source,sort_keys=True)
        c=single_joint_config(source)
        q0,rows,phases=build(c)
        self.assertEqual(json.dumps(source,sort_keys=True),original)
        for phase in phases:
            delta=np.asarray(phase['end_q'])-phase['start_q']
            self.assertLessEqual(np.count_nonzero(abs(delta)>1e-12),1)
        self.assertLessEqual(np.max(np.abs([r[2] for r in rows])),.094)
        self.assertLessEqual(np.max(np.abs([r[3] for r in rows])),.289)

    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config/experiments/rb3_precision_benchmark.json').read_text())

    def test_quintic_endpoint_conditions(self):
        q,v,a=quintic(np.zeros(6),np.ones(6),np.array([0.,6.]),6.)
        np.testing.assert_allclose(q,[[0.]*6,[1.]*6]);np.testing.assert_allclose(v,0.);np.testing.assert_allclose(a,0.)

    def test_analytic_velocity_acceleration(self):
        t=2.;h=1e-4
        qm,vm,_=quintic(np.zeros(6),np.ones(6),t-h,6.)
        qp,vp,_=quintic(np.zeros(6),np.ones(6),t+h,6.)
        q,v,a=quintic(np.zeros(6),np.ones(6),t,6.)
        np.testing.assert_allclose((qp-qm)/(2*h),v,atol=1e-8)
        np.testing.assert_allclose((vp-vm)/(2*h),a,atol=1e-8)

    def test_frozen_duration_windows_and_bounds(self):
        _,rows,phases=build(self.config)
        self.assertEqual(len(rows),3480)
        self.assertAlmostEqual(rows[-1][0],29.)
        self.assertLessEqual(np.max(abs(np.array([r[2] for r in rows]))),.5)
        self.assertLessEqual(np.max(abs(np.array([r[3] for r in rows]))),1.)
        self.assertNotIn('move_D_held_out',[p['name'] for p in phases])
        _,validation,_=build(self.config,True)
        self.assertEqual(len(validation),2400)

    def test_bounds_cannot_be_evaded_by_sampling(self):
        self.config['segment_duration_s']=.1
        with self.assertRaisesRegex(ValueError,'bounds'):build(self.config)

    def test_default_is_unchanged(self):
        q0,rows,_=build(self.config)
        self.assertIs(command_rows(rows,q0,'analytical'),rows)

    def test_common_waypoints_delay_and_zero_velocity(self):
        q0,rows,_=build(self.config)
        linear=command_rows(rows,q0,'linear_zero')
        smooth=command_rows(rows,q0,'smooth_pv')
        for i in range(3,len(rows),4):
            np.testing.assert_allclose(linear[i][1],smooth[i][1],atol=1e-14)
        for i in range(4,len(rows)):
            np.testing.assert_array_equal(smooth[i][1],rows[i-4][1])
            np.testing.assert_array_equal(smooth[i][2],rows[i-4][2])
        np.testing.assert_array_equal(np.array([r[2] for r in linear]),0.)

    def test_no_angle_wrapping_or_future_target(self):
        rows=[(i+1,np.array([i+1.]),np.ones(1),np.zeros(1),0) for i in range(12)]
        q0=np.zeros(1)
        lin=command_rows(rows,q0,'linear_zero');smooth=command_rows(rows,q0,'smooth_pv')
        self.assertEqual(lin[-1][1][0],8.)
        self.assertEqual(smooth[-1][1][0],8.)
        for row in lin[:4]+smooth[:4]:np.testing.assert_array_equal(row[1],q0)

    def test_low_benchmark_named_poses_and_modes(self):
        from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
        config=json.loads((Path(__file__).resolve().parents[1]/'config/experiments/rb3_low_precision_benchmark.json').read_text())
        kin=RB3730Kinematics(base_position=config['base_position'],base_quaternion_xyzw=config['base_quaternion_xyzw'])
        for name,q in config['poses'].items():
            np.testing.assert_allclose(kin.forward(q)[0],config['pose_xyz_m'][name],atol=1e-7)
        q0,rows,_=build(config)
        smooth=command_rows(rows,q0,'smooth_pv')
        dt=config['physics_dt']
        q=np.array([r[1] for r in smooth]);v=np.array([r[2] for r in smooth])
        # Interior finite differences independently check analytical derivatives.
        np.testing.assert_allclose((q[2:]-q[:-2])/(2*dt),v[1:-1],atol=2e-6)
        self.assertLessEqual(np.max(abs(v)),config['max_command_speed_rad_s'])
        self.assertLessEqual(np.max(abs(np.array([r[3] for r in smooth]))),config['max_command_acceleration_rad_s2'])
