"""Tests for the GUI-independent RL reference loader."""

from __future__ import annotations

import tempfile
import unittest

import h5py
import numpy as np

from regrind.data.rb3_revo2_reference import (
    RB3_JOINT_NAMES,
    REVO2_JOINT_NAMES,
    load_rb3_revo2_reference,
)


class RB3Revo2RLReferenceTest(unittest.TestCase):
    def _write(self, path, *, mismatch=False, failed=False):
        rb3 = np.arange(18, dtype=float).reshape(3, 6) * 0.01
        revo2 = np.arange(18, 36, dtype=float).reshape(3, 6) * 0.01
        combined = np.concatenate((rb3, revo2), axis=1)
        if mismatch:
            combined[0, 0] += 1.0
        with h5py.File(path, "w") as h5_file:
            h5_file["rb3_joints"] = rb3
            h5_file["revo2_joints"] = revo2
            h5_file["reference_joints"] = combined
            h5_file["object_pos"] = np.zeros((3, 3))
            h5_file["object_quat"] = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3)
            h5_file["wrist_pos"] = np.ones((3, 3)) * 0.1
            h5_file["wrist_quat"] = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3)
            h5_file["quat_convention"] = "wxyz"
            h5_file["fps"] = 30.0
            h5_file["rb3_joint_names"] = np.asarray(RB3_JOINT_NAMES, dtype="S")
            h5_file["revo2_joint_names"] = np.asarray(REVO2_JOINT_NAMES, dtype="S")
            h5_file["ik_success"] = np.asarray([True, not failed, True])
            h5_file["joint_limit_violation"] = np.zeros(3, dtype=bool)

    def test_loads_12_dof_and_converts_wxyz(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            self._write(output.name)
            reference = load_rb3_revo2_reference(output.name)
        self.assertEqual(reference.reference_joints.shape, (3, 12))
        self.assertEqual(reference.joint_names, RB3_JOINT_NAMES + REVO2_JOINT_NAMES)
        np.testing.assert_allclose(reference.object_quat_xyzw[0], (0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(reference.wrist_quat_xyzw[0], (0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(reference.dt, 1.0 / 30.0)

    def test_rejects_inconsistent_combined_array(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            self._write(output.name, mismatch=True)
            with self.assertRaisesRegex(ValueError, "does not equal"):
                load_rb3_revo2_reference(output.name)

    def test_rejects_failed_ik(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            self._write(output.name, failed=True)
            with self.assertRaisesRegex(ValueError, "failed IK"):
                load_rb3_revo2_reference(output.name)

    def test_loads_and_reorders_optional_mano21(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            self._write(output.name)
            sequential = np.arange(3 * 21 * 3, dtype=float).reshape(3, 21, 3)
            with h5py.File(output.name, "a") as h5_file:
                h5_file["mano_joint_world"] = sequential
                h5_file["mano_joint_order"] = "mano21_sequential_thumb_index_middle_ring_little"
            reference = load_rb3_revo2_reference(output.name)
        self.assertEqual(reference.mano_joint_world_semantic.shape, (3, 21, 3))
        np.testing.assert_array_equal(
            reference.mano_joint_world_semantic[:, 0],
            sequential[:, 0],
        )
        np.testing.assert_array_equal(
            reference.mano_joint_world_semantic[:, 1],
            sequential[:, 5],
        )
