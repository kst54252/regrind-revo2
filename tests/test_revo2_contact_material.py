"""Opt-in contact scope, validation and source-asset preservation."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from copy import deepcopy

from pxr import Usd, UsdPhysics, UsdShade
from regrind.utils.revo2_contact_material import (
    DISTAL_NAMES, bind_last_phalanges, configure_contact, load_contact_config, verify_runtime_materials,
)
from tools.arm_diagnostics.analyze_transfer_recovery import verify_pair_conditions

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/experiments/revo2_rubber_contact.json'


class ContactMaterialTest(unittest.TestCase):
    def test_physx_negative_restitution_is_verified_per_environment(self):
        import numpy as np
        values=np.tile([.8,.8,0.],(16,25,1))
        values[:,:10,2]=-10000.
        self.assertEqual(verify_runtime_materials(values,load_contact_config(CONFIG)),[10]*16)
        values[15,9,2]=0.
        with self.assertRaises(ValueError): verify_runtime_materials(values,load_contact_config(CONFIG))

    def test_opt_in_does_not_mutate_shared_spawn(self):
        original = SimpleNamespace(func='original', usd_path='unchanged')
        cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=original)))
        spec = configure_contact(cfg, CONFIG)
        self.assertEqual(original.func, 'original')
        self.assertFalse(hasattr(original, 'revo2_contact_spec'))
        self.assertEqual(cfg.scene.robot.spawn.usd_path, 'unchanged')
        self.assertEqual(spec['static_friction'], .8)
        self.assertFalse(spec['calibrated_to_hardware'])

    def test_rejects_invalid_material(self):
        spec = json.loads(CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for key, value in [('scope','whole_hand'), ('static_friction',float('nan')),
                               ('contact_stiffness_n_per_m',0), ('dynamic_friction',1.2)]:
                path.write_text(json.dumps(dict(spec, **{key:value})))
                with self.assertRaises(ValueError): load_contact_config(path)

    def test_actual_assets_only_ten_distal_collision_bindings(self):
        for relative in ('USD/revo2_floating.usda', 'USD/rb3_revo2_vertical.usda'):
            path = ROOT/relative
            if not path.exists(): self.skipTest('Local robot USD not installed')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stage = Usd.Stage.Open(str(path))
            before = {layer.identifier:layer.ExportToString() for layer in stage.GetUsedLayers()}
            # Session edits are not saved back to any source USD layer.
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                material = UsdShade.Material.Define(stage, '/TestRubber')
                UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                paths = bind_last_phalanges(stage.GetDefaultPrim(), material)
            self.assertEqual(len(paths),10)
            for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
                bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                self.assertEqual(bool(bound and bound.GetPath() == material.GetPath()), str(prim.GetPath()) in paths)
            session = stage.GetSessionLayer()
            for layer in stage.GetUsedLayers():
                if layer != session:
                    self.assertEqual(layer.ExportToString(), before[layer.identifier])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            # New opinions contain only material/binding, never drive or mass edits.
            text = session.ExportToString()
            for forbidden in ('physics:mass', 'physics:diagonalInertia', 'drive:', 'points ='):
                self.assertNotIn(forbidden,text)

    def test_missing_distal_links_fail_closed(self):
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim('/Robot','Xform')
        material = UsdShade.Material.Define(stage,'/Rubber')
        with self.assertRaisesRegex(ValueError,'exactly five'):
            bind_last_phalanges(root,material)


class ContactComparisonTest(unittest.TestCase):
    def pair(self):
        fields = ('all_q','all_v','robot_root','object_state','wrist_pos','wrist_quat',
                  'wrist_velocity','hand_q','hand_v','follower_q','follower_v','phase')
        left = dict(mode='floating',stage='grasp',ends=[{} for _ in range(20)],
                    frozen_policy_verified=True,checkpoint_sha256='same-checkpoint',
                    robot_spawn=dict(func='isaaclab.sim.spawners.from_files.from_files:spawn_from_usd',
                                     usd_path='same-asset'),
                    initial_states=[dict.fromkeys(fields,0.) for _ in range(20)])
        right = deepcopy(left)
        right['fingertip_contact'] = {'verified':True}
        right['robot_spawn']['func'] = 'regrind.utils.revo2_contact_material:spawn_with_last_phalanx_contact'
        right['robot_spawn']['revo2_contact_spec'] = {'contact_stiffness_n_per_m':10000.}
        return [left,right]

    def test_material_pair_accepts_only_contact_differences(self):
        pair=self.pair();before=deepcopy(pair)
        verify_pair_conditions(pair,material_only=True)
        self.assertEqual(pair,before)
        for key in ('gains','physics_dt','checkpoint_sha256','mode'):
            pair=self.pair();pair[1][key]='changed'
            with self.assertRaises(ValueError): verify_pair_conditions(pair,material_only=True)
        for key in ('usd_path','func'):
            pair=self.pair();pair[1]['robot_spawn'][key]='different'
            with self.assertRaises(ValueError): verify_pair_conditions(pair,material_only=True)

    def test_material_comparison_requires_actual_same_initial_state(self):
        pair=self.pair();pair[1]['initial_states'][4]['phase']=1
        with self.assertRaises(AssertionError): verify_pair_conditions(pair,material_only=True)
