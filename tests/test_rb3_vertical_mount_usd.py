"""Composition checks for the physical RB3-to-Revo2 vertical adapter mount."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = PROJECT_ROOT / "USD" / "rb3_revo2_vertical.usda"
LINK6_PATH = "/World/rb3_730es_u/Geometry/link0/link1/link2/link3/link4/link5/link6"
ARM_MOUNT_PATH = LINK6_PATH + "/revo2_mount"
ADAPTER_MESH_PATH = LINK6_PATH + "/revo2_vertical_adapter/geometry/mesh"
WRIST_PATH = "/World/revo2_right/Geometry/world/right_hand_base_link"
HAND_MOUNT_PATH = WRIST_PATH + "/rb3_mount"


class RB3VerticalMountUsdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = Usd.Stage.Open(str(STAGE_PATH))
        if cls.stage is None:
            raise RuntimeError(f"failed to open {STAGE_PATH}")
        cls.cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def _world_matrix(self, path: str) -> np.ndarray:
        prim = self.stage.GetPrimAtPath(path)
        self.assertTrue(prim.IsValid(), path)
        return np.asarray(self.cache.GetLocalToWorldTransform(prim), dtype=float)

    def test_link6_to_wrist_is_flange_plus_adapter_and_identity(self):
        link6 = self._world_matrix(LINK6_PATH)
        wrist = self._world_matrix(WRIST_PATH)
        # Gf matrices use row-vector convention; translation is the final row.
        relative = np.linalg.inv(link6) @ wrist
        np.testing.assert_allclose(relative[:3, :3], np.eye(3), atol=1.0e-12)
        np.testing.assert_allclose(
            relative[3, :3], (0.0, 0.0, 0.141304972), atol=1.0e-12
        )

    def test_fixed_joint_frames_coincide(self):
        arm_mount = self._world_matrix(ARM_MOUNT_PATH)
        hand_mount = self._world_matrix(HAND_MOUNT_PATH)
        np.testing.assert_allclose(arm_mount, hand_mount, atol=1.0e-12)

    def test_adapter_has_collision(self):
        prim = self.stage.GetPrimAtPath(ADAPTER_MESH_PATH)
        self.assertTrue(prim.IsValid())
        self.assertTrue(UsdPhysics.CollisionAPI(prim))
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        self.assertTrue(mesh_collision)
        self.assertEqual(mesh_collision.GetApproximationAttr().Get(), "convexHull")


if __name__ == "__main__":
    unittest.main()
