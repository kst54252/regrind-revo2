import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from tools.arm_diagnostics.search_tabletop_yaw import angle_key, dense_reference, rotate_task


class TabletopYawTest(unittest.TestCase):
    def test_zero_yaw_is_existing_translation(self):
        p = np.array([[.4, .1, .2], [.5, .2, .3]])
        q = Rotation.from_euler('xyz', [[.1, .2, .3], [.4, .5, .6]]).as_quat()
        old = np.array([.4, 0, .01]); new = np.array([.6, -.2, .01])
        pos, quat = rotate_task(p, q, old, new, 0.)
        np.testing.assert_allclose(pos, p+new-old, atol=1e-14)
        np.testing.assert_allclose(quat, q, atol=1e-14)

    def test_yaw_uses_can_pivot_and_rotates_orientation_on_left(self):
        c = np.array([.5, .2, .01]); p = c+np.array([[.1, 0, .15]])
        q = Rotation.from_euler('x', 50, degrees=True).as_quat()[None]
        pos, quat = rotate_task(p, q, c, c, 90)
        np.testing.assert_allclose(pos, c+[[0, .1, .15]], atol=1e-14)
        expected = (Rotation.from_euler('z', 90, degrees=True)*Rotation.from_quat(q)).as_quat()
        np.testing.assert_allclose(quat, expected, atol=1e-14)

    def test_relative_hand_object_pose_is_preserved_for_entire_motion(self):
        rng = np.random.default_rng(3)
        obj = rng.normal(size=(6, 3)); hand = rng.normal(size=(6, 3))
        obj_q = Rotation.random(6, random_state=rng).as_quat()
        hand_q = Rotation.random(6, random_state=rng).as_quat()
        center = obj[0]; destination = np.array([.5, .1, center[2]])
        op, oq = rotate_task(obj, obj_q, center, destination, -65)
        hp, hq = rotate_task(hand, hand_q, center, destination, -65)
        np.testing.assert_allclose(op[0], destination, atol=1e-14)
        np.testing.assert_allclose(hp[:, 2], hand[:, 2], atol=1e-14)
        original_local = Rotation.from_quat(obj_q).inv().apply(hand-obj)
        new_local = Rotation.from_quat(oq).inv().apply(hp-op)
        np.testing.assert_allclose(new_local, original_local, atol=1e-14)
        before = Rotation.from_quat(obj_q).inv()*Rotation.from_quat(hand_q)
        after = Rotation.from_quat(oq).inv()*Rotation.from_quat(hq)
        np.testing.assert_allclose((before.inv()*after).magnitude(), 0, atol=1e-14)

    def test_dense_sampling_keeps_original_knots_and_duration(self):
        pos = np.array([[.4, 0, .2], [.5, .1, .3], [.4, .2, .4]])
        quat = Rotation.from_euler('xyz', [[.1, .2, .3], [.4, .3, .2], [.1, -.2, .3]]).as_quat()
        reference = SimpleNamespace(frames=3, dt=1/30, wrist_pos=pos, wrist_quat_xyzw=quat)
        t, p, q = dense_reference(reference)
        self.assertEqual(len(t), 9)
        self.assertAlmostEqual(t[-1], 2/30)
        np.testing.assert_allclose(p[::4], pos, atol=1e-14)
        np.testing.assert_allclose(abs(np.sum(q[::4]*quat, axis=1)), 1., atol=1e-14)

    def test_reject_invalid_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):
            rotate_task([1, 2, 3], [[0, 0, 0, 1]], [0]*3, [0]*3, 0)
        with self.assertRaises(ValueError):
            rotate_task([[1, 2, 3]], [[0, 0, 0, 1]], [0]*3, [0]*3, np.nan)

    def test_search_angle_labels_not_joint_coordinates(self):
        self.assertEqual(angle_key(190), -170.)
        self.assertEqual(angle_key(-180), -180.)


if __name__ == '__main__':
    unittest.main()
