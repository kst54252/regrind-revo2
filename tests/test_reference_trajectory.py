"""Tests for trajectory loading and continuity calculations."""

from __future__ import annotations

import tempfile
import unittest

import h5py
import numpy as np

from tools.rb3_revo2_ik.reference_trajectory import (
    DEFAULT_REVO2_JOINT_NAMES,
    MANO21_SEQUENTIAL_TO_REVO_SEMANTIC,
    analyze_continuity,
    load_reference_trajectory,
)


class ReferenceTrajectoryTest(unittest.TestCase):
    def test_combined_only_and_dt_override(self):
        q = np.arange(48, dtype=float).reshape(4, 12) * 0.01
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            with h5py.File(output.name, "w") as h5_file:
                h5_file["reference_joints"] = q
                h5_file["fps"] = 30.0
            trajectory = load_reference_trajectory(output.name, dt_override=0.01)
        np.testing.assert_allclose(trajectory.rb3_joints, q[:, :6])
        np.testing.assert_allclose(trajectory.revo2_joints, q[:, 6:])
        self.assertEqual(trajectory.dt, 0.01)

    def test_continuity_units_and_discontinuity_frame(self):
        q = np.zeros((4, 12))
        q[1:, 0] = (0.1, 0.7, 0.9)
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            with h5py.File(output.name, "w") as h5_file:
                h5_file["reference_joints"] = q
            trajectory = load_reference_trajectory(output.name, dt_override=0.1)
        analysis = analyze_continuity(trajectory, 0.5)
        self.assertAlmostEqual(analysis.max_joint_step_norm, 0.6)
        self.assertAlmostEqual(analysis.max_abs_velocity_per_joint[0], 6.0)
        self.assertAlmostEqual(analysis.max_abs_acceleration_per_joint[0], 50.0)
        np.testing.assert_array_equal(analysis.discontinuity_frames, (2,))

    def test_legacy_revo2_labels_map_to_usd_joint_names(self):
        q = np.zeros((2, 12))
        legacy_names = np.asarray([f"revo2_joint_{index}" for index in range(6)])
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            with h5py.File(output.name, "w") as h5_file:
                h5_file["reference_joints"] = q
                h5_file.create_dataset(
                    "revo2_joint_names",
                    data=legacy_names.astype(object),
                    dtype=h5py.string_dtype("utf-8"),
                )
            trajectory = load_reference_trajectory(output.name)
        self.assertEqual(trajectory.revo2_joint_names, DEFAULT_REVO2_JOINT_NAMES)

    def test_object_pose_loads_and_converts_wxyz(self):
        q = np.zeros((2, 12))
        object_pos = np.asarray([[0.3, 0.3, 0.02], [0.31, 0.3, 0.02]])
        object_quat_wxyz = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2)
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            with h5py.File(output.name, "w") as h5_file:
                h5_file["reference_joints"] = q
                h5_file["object_pos"] = object_pos
                h5_file["object_quat"] = object_quat_wxyz
                h5_file["quat_convention"] = "wxyz"
            trajectory = load_reference_trajectory(output.name)
        np.testing.assert_allclose(trajectory.object_pos, object_pos)
        np.testing.assert_allclose(
            trajectory.object_quat_xyzw,
            np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2),
        )

    def test_mano_skeleton_loads_from_recorded_source(self):
        q = np.zeros((2, 12))
        mano = np.arange(2 * 21 * 3, dtype=float).reshape(2, 21, 3) * 0.001
        with tempfile.TemporaryDirectory() as directory:
            source_path = f"{directory}/world.h5"
            reference_path = f"{directory}/reference.h5"
            with h5py.File(source_path, "w") as h5_file:
                h5_file["mano_joint_world"] = mano
            with h5py.File(reference_path, "w") as h5_file:
                h5_file["reference_joints"] = q
                h5_file.create_dataset(
                    "source_retargeting_file",
                    data=source_path,
                    dtype=h5py.string_dtype("utf-8"),
                )
            trajectory = load_reference_trajectory(reference_path)
        np.testing.assert_allclose(trajectory.mano_joint_world, mano)

    def test_revo_semantic_mano_is_restored_to_sequential_order(self):
        q = np.zeros((2, 12))
        sequential = np.arange(2 * 21 * 3, dtype=float).reshape(2, 21, 3)
        semantic = sequential[:, MANO21_SEQUENTIAL_TO_REVO_SEMANTIC]
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            with h5py.File(output.name, "w") as h5_file:
                h5_file["reference_joints"] = q
                h5_file["mano_joint_world"] = semantic
                h5_file["mano_joint_order"] = "revo_semantic_kp00_to_kp20"
            trajectory = load_reference_trajectory(output.name)
        np.testing.assert_allclose(trajectory.mano_joint_world, sequential)

    def test_loads_optional_follower_state_and_policy_drive_target(self):
        q = np.zeros((3, 12))
        follower = np.arange(15, dtype=float).reshape(3, 5) * 0.01
        drive_target = np.arange(18, dtype=float).reshape(3, 6) * 0.02
        follower_names = (
            "right_thumb_distal_joint",
            "right_index_distal_joint",
            "right_middle_distal_joint",
            "right_ring_distal_joint",
            "right_pinky_distal_joint",
        )
        with tempfile.NamedTemporaryFile(suffix=".h5") as output:
            with h5py.File(output.name, "w") as h5_file:
                h5_file["reference_joints"] = q
                h5_file["revo2_follower_joints"] = follower
                h5_file["revo2_joint_drive_target"] = drive_target
                h5_file.create_dataset(
                    "revo2_follower_joint_names",
                    data=np.asarray(follower_names, dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )
            trajectory = load_reference_trajectory(output.name)
        np.testing.assert_allclose(trajectory.revo2_follower_joints, follower)
        np.testing.assert_allclose(
            trajectory.revo2_joint_drive_target, drive_target
        )


if __name__ == "__main__":
    unittest.main()
