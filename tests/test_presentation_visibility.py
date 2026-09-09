"""Visual-only capture overrides must not alter robot/physics/source layers."""
import unittest

from pxr import Usd, UsdGeom, UsdPhysics
from regrind.utils.presentation_visibility import hide_revo2_keypoints


class PresentationVisibilityTest(unittest.TestCase):
    def test_only_marker_visibility_is_authored_in_session_layer(self):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, '/World/envs/env_0/Robot/link').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(root)
        marker = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild('kp_17_index_tip'))
        mesh = UsdGeom.Sphere.Define(stage, marker.GetPath().AppendChild('kp_17_index_tip'))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        visual = UsdGeom.Sphere.Define(stage, root.GetPath().AppendChild('finger_visual'))
        other = UsdGeom.Sphere.Define(stage, '/World/Object/kp_17_index_tip')
        before = stage.GetRootLayer().ExportToString()
        paths = hide_revo2_keypoints(stage)
        self.assertEqual(paths, [str(marker.GetPath())])
        self.assertEqual(mesh.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertEqual(visual.ComputeVisibility(), UsdGeom.Tokens.inherited)
        self.assertEqual(other.ComputeVisibility(), UsdGeom.Tokens.inherited)
        self.assertTrue(mesh.GetPrim().IsActive())
        self.assertTrue(UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get())
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        self.assertEqual(stage.GetEditTarget().GetLayer(), stage.GetRootLayer())

    def test_explicit_model_scope_preserves_comparison_skeleton(self):
        stage = Usd.Stage.CreateInMemory()
        model = UsdGeom.Sphere.Define(stage, '/Presentation/Hand/kp_00_wrist')
        other = UsdGeom.Sphere.Define(stage, '/Presentation/HandOther/kp_00_wrist')
        skeleton = UsdGeom.Sphere.Define(stage, '/Presentation/Skeleton/kp_00_wrist')
        before = stage.GetRootLayer().ExportToString()
        self.assertEqual(hide_revo2_keypoints(stage, '/Presentation/Hand'), [str(model.GetPath())])
        self.assertEqual(model.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertEqual(other.ComputeVisibility(), UsdGeom.Tokens.inherited)
        self.assertEqual(skeleton.ComputeVisibility(), UsdGeom.Tokens.inherited)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        for invalid in ('/', '/Missing', 'relative'):
            with self.assertRaises(ValueError):
                hide_revo2_keypoints(stage, invalid)
