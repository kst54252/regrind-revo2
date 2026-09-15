from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.warm_start_ik import WarmStartIK
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics
from tools.robot_execution.backends import MockBackend, SimulationCallbacks
from tools.robot_execution.contracts import DecodedTarget, Pose, RobotState, reorder
from tools.robot_execution.session import ExecutionFault, ExecutionLimits, ExecutionSession
from tools.robot_execution.isaac_state import IsaacMountedStateReader


class TestRobotExecution(unittest.TestCase):
    def setUp(self):
        self.kin = RB3730Kinematics()
        self.names = self.kin.joint_names + Revo2Kinematics.joint_names
        self.q = np.r_[[0., .5, -1., .5, .8, 0.], np.full(6, .5)]
        low, high = self.kin.get_joint_limits()
        self.limits = ExecutionLimits(np.r_[low, np.zeros(6)], np.r_[high, np.ones(6)*2],
                                      np.ones(12)*.5, np.ones(12)*2, np.ones(12)*.1, .02, .05)
        self.t = 10.
        self.backend = MockBackend(self.kin, self.names, self.q, lambda: self.t)
        self.session = ExecutionSession(self.backend, self.kin, self.names, self.limits,
            frame='mock_world', solver=WarmStartIK(self.kin), clock=lambda: self.t, require_object=False)

    def advance(self):
        self.t += self.limits.period
        self.backend.advance(self.limits.period)

    def target(self, state):
        return DecodedTarget(state.wrist, self.q[6:], self.names[6:], self.t+.02, 0)

    def start(self):
        self.session.start()
        self.advance()

    def rejected(self, expression, pattern):
        with self.assertRaisesRegex(ExecutionFault, pattern):
            expression()
        self.assertEqual(self.session.status, 'faulted')
        self.assertTrue(self.backend.stop_reasons)
        self.assertEqual(len(self.backend.sent), 0)

    def test_actual_state_and_existing_fk_ik_without_double_mount(self):
        self.start()
        seen = []
        def target(state):
            seen.append(state)
            return self.target(state)
        command = self.session.step(target)
        np.testing.assert_allclose(command.position, self.q, atol=1e-8)
        np.testing.assert_array_equal(command.velocity_target, np.zeros(12))
        self.assertEqual(command.state_sequence, seen[0].sequence)
        self.assertEqual(seen[0].wrist_source, 'mock_fk')
        self.assertEqual(self.session.last_result.position_error_m, 0.)
        self.session.close()
        self.assertIsNone(self.backend.pending)

    def test_names_not_array_positions(self):
        read = self.backend.read_state
        def shuffled():
            state = read()
            return replace(state, joint_names=state.joint_names[::-1],
                           position=state.position[::-1], velocity=state.velocity[::-1])
        self.backend.read_state = shuffled
        self.start()
        def target(state):
            return replace(self.target(state), hand_names=self.names[6:][::-1],
                           hand_position=self.q[6:][::-1])
        np.testing.assert_allclose(self.session.step(target).position, self.q, atol=1e-8)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            reorder(self.q, self.names[:-1]+('unknown',), self.names)

    def test_hardware_blocked_before_any_io(self):
        self.backend.kind = 'hardware'
        self.backend.read_state = lambda: self.fail('Hardware read was attempted')
        with self.assertRaisesRegex(ExecutionFault, 'no device I/O'):
            self.session.start()
        self.assertFalse(self.backend.stop_reasons)

    def test_stale_state(self):
        old = self.backend.read_state()
        self.start()
        self.backend.read_state = lambda: replace(old, sample_time=self.t-1)
        self.rejected(lambda: self.session.step(self.target), 'stale')

    def test_repeated_state_sequence(self):
        old = self.backend.read_state()
        self.start()
        self.backend.read_state = lambda: old
        self.rejected(lambda: self.session.step(self.target), 'Repeated')

    def test_object_measurement_required_by_default(self):
        self.session.require_object = True
        self.rejected(self.session.start, 'measured object')

    def test_stale_object_pose(self):
        read = self.backend.read_state
        self.backend.read_state = lambda: replace(read(), object_pose=read().wrist, object_sample_time=self.t-.2)
        self.rejected(self.session.start, 'object.*stale')

    def test_protective_stop(self):
        read = self.backend.read_state
        self.backend.read_state = lambda: replace(read(), protective_stop=True)
        self.rejected(self.session.start, 'protective stop')

    def test_wrong_target_frame(self):
        self.start()
        def target(state):
            pose = replace(state.wrist, frame='camera_uncalibrated')
            return replace(self.target(state), wrist=pose)
        self.rejected(lambda: self.session.step(target), 'frames differ')

    def test_expired_target(self):
        self.start()
        self.rejected(lambda: self.session.step(lambda s: replace(self.target(s), valid_until=self.t)), 'Expired')

    def test_ik_failure_is_not_silently_replayed(self):
        self.start()
        self.session.solver = SimpleNamespace(inverse=lambda *a, **k: SimpleNamespace(success=False, message='unreachable'))
        self.rejected(lambda: self.session.step(self.target), 'IK rejected')

    def test_large_branch_jump_is_not_wrapped(self):
        self.start()
        self.session.solver = SimpleNamespace(inverse=lambda *a, **k: SimpleNamespace(
            success=True, finite=True, joint_limit_violation=False, q=self.q[:6]+np.array([0,0,0,0,0,2*np.pi])))
        self.rejected(lambda: self.session.step(self.target), 'limit violation|tracking gap')

    def test_speed_and_acceleration_guards(self):
        for delta, message in [(.02, 'path speed'), (.002, 'acceleration')]:
            with self.subTest(delta=delta):
                self.setUp()
                self.start()
                self.rejected(lambda: self.session.step(lambda s: replace(
                    self.target(s), hand_position=self.q[6:]+delta)), message)

    def test_actual_limits_and_velocity_checked(self):
        for field, value, message in [('position', np.ones(12)*100, 'limit violation'),
                                       ('velocity', np.ones(12)*100, 'speed')]:
            with self.subTest(field=field):
                self.setUp()
                read = self.backend.read_state
                self.backend.read_state = lambda: replace(read(), **{field:value})
                self.rejected(self.session.start, message)

    def test_ik_overrun_does_not_send(self):
        self.start()
        original = self.session.solver.inverse
        def slow(*args, **kwargs):
            result = original(*args, **kwargs)
            self.t += .03
            return result
        self.session.solver = SimpleNamespace(inverse=slow)
        self.rejected(lambda: self.session.step(self.target), 'expired')

    def test_cadence_overrun_faults_not_catchup(self):
        self.start()
        self.t += .1
        self.rejected(lambda: self.session.step(self.target), 'cadence')

    def test_early_command_burst_is_rejected(self):
        self.session.start()
        self.t += .001
        self.backend.advance(.001)
        self.rejected(lambda: self.session.step(self.target), 'cadence')

    def test_partial_send_failure_stops_and_latches(self):
        self.start()
        def failed(command):
            raise OSError('hand ack failed after arm send')
        self.backend.send = failed
        self.rejected(lambda: self.session.step(self.target), 'hand ack')
        with self.assertRaisesRegex(ExecutionFault, 'not running'):
            self.session.step(self.target)
        with self.assertRaisesRegex(ExecutionFault, 'new session'):
            self.session.start()

    def test_stop_failure_is_reported(self):
        self.start()
        def failed(reason):
            raise OSError('stop connection lost')
        self.backend.stop = failed
        with self.assertRaisesRegex(ExecutionFault, 'STOP REQUEST FAILED'):
            self.session.step(lambda _: (_ for _ in ()).throw(ValueError('bad action')))
        self.assertIn('connection lost', self.session.stop_error)

    def test_no_nonfinite_or_nonunit_quaternion(self):
        for q in ([0,0,0,0], [0,0,0,2], [float('nan'),0,0,1]):
            with self.assertRaises(ValueError):
                Pose(np.zeros(3), q, 'world')
        with self.assertRaises(ValueError):
            replace(self.backend.read_state(), position=np.ones(12)*np.nan)
        with self.assertRaises(ValueError):
            replace(self.limits, period=float('nan'))

    def test_snapshots_copy_transport_buffers(self):
        state = self.backend.read_state()
        self.backend.q[:] = 0
        np.testing.assert_array_equal(state.position, self.q)
        with self.assertRaises(ValueError):
            state.position[0] = 123

    def test_simulation_callback_seam_uses_same_contract(self):
        self.session.backend = SimulationCallbacks(
            lambda: replace(self.backend.read_state(), wrist_source='physx_body_pose'),
            self.backend.send, self.backend.stop)
        self.start()
        self.session.step(self.target)
        self.assertEqual(len(self.backend.sent), 1)
        self.session.close()

    def test_isaac_reader_actual_path_names_and_env_origin(self):
        origin = np.array([[2., 3., 0.]])
        robot = SimpleNamespace(joint_names=self.names[::-1], data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=self.q[::-1][None]),
            joint_vel=SimpleNamespace(torch=np.zeros((1,12)))))
        command = SimpleNamespace(robot=robot, current_hand_wrist_pos=np.array([[.4,0,.2]]),
            current_hand_wrist_quat=np.array([[0,0,0,1]]), current_object_pos=np.array([[.5,0,.02]]),
            current_object_quat=np.array([[0,0,0,1]]), target_hand_wrist_pos=np.ones((1,3))*100)
        env = SimpleNamespace(scene=SimpleNamespace(env_origins=origin))
        reader = IsaacMountedStateReader(env, command, self.names)
        state = reader.sample(7, 12.)
        np.testing.assert_array_equal(state.position, self.q)
        np.testing.assert_allclose(state.wrist.position, [2.4,3.,.2])
        np.testing.assert_allclose(state.object_pose.position, [2.5,3.,.02])
        self.assertEqual(state.wrist_source, 'physx_body_pose')
        self.assertEqual(state.sample_time, 12.)
        self.assertEqual(state.object_sample_time, 12.)
        self.assertEqual(state.sequence, 7)

    def test_unknown_leader_names_rejected(self):
        with self.assertRaisesRegex(ValueError, 'leader order'):
            ExecutionSession(self.backend, self.kin, self.names[:6]+tuple(f'joint{i}' for i in range(6)),
                             self.limits, frame='world')


if __name__ == '__main__':
    unittest.main()
