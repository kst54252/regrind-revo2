"""Tests for object-relative DexYCB hand mirroring."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.dexycb_world_transform.mirror_hand_trajectory import (
    mirror_hand_in_object_frame,
)


class MirrorHandTrajectoryTest(unittest.TestCase):
    def test_object_local_x_reflection_and_double_mirror(self):
        rng = np.random.default_rng(3)
        hand_local = rng.normal(size=(2, 21, 3)) * 0.04
        rotation = Rotation.from_euler("XYZ", [[0.2, -0.4, 0.7], [-0.3, 0.1, 1.0]])
        matrix = rotation.as_matrix()
        position = np.asarray([[0.3, -0.2, 0.9], [0.4, 0.1, 1.0]])
        hand_world = np.einsum("tij,tkj->tki", matrix, hand_local) + position[:, None]
        quaternion_wxyz = rotation.as_quat()[:, [3, 0, 1, 2]]

        mirrored = mirror_hand_in_object_frame(
            hand_world, position, quaternion_wxyz, "wxyz", mirror_axis=0
        )
        mirrored_local = np.einsum(
            "tji,tkj->tki", matrix, mirrored - position[:, None]
        )
        expected = hand_local.copy()
        expected[:, :, 0] *= -1.0
        np.testing.assert_allclose(mirrored_local, expected, atol=1.0e-12)

        restored = mirror_hand_in_object_frame(
            mirrored, position, quaternion_wxyz, "wxyz", mirror_axis=0
        )
        np.testing.assert_allclose(restored, hand_world, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
