"""Opt-in last-phalanx compliant contacts, shared by floating/arm train/eval.

Only spawned-stage material opinions are authored; referenced USD files are never
saved or regenerated. Rigid geometry and mass/inertia stay unchanged. Parameters
are an explicitly uncalibrated proxy, not measured Revo2 rubber properties.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
DISTAL_NAMES = tuple(f'right_{finger}_distal_link' for finger in FINGERS)


def load_contact_config(path):
    data = json.loads(Path(path).read_text())
    if data.get('scope') != 'five_distal_links_and_fixed_touch_children':
        raise ValueError('Unsupported rubber collision scope')
    for key in ('static_friction', 'dynamic_friction', 'contact_stiffness_n_per_m', 'contact_damping_ns_per_m'):
        if not isinstance(data.get(key), (int, float)) or not math.isfinite(data[key]) or data[key] < 0:
            raise ValueError(f'Invalid rubber parameter: {key}')
    if data['contact_stiffness_n_per_m'] <= 0 or data['dynamic_friction'] > data['static_friction']:
        raise ValueError('Rubber requires positive stiffness and dynamic friction <= static friction')
    data['sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return data


def bind_last_phalanges(root, material):
    """Bind five non-instance body prims; retain their instanced collision meshes."""
    from pxr import Usd, UsdPhysics, UsdShade
    bodies = [p for p in Usd.PrimRange(root)
              if p.GetName() in DISTAL_NAMES and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if sorted(p.GetName() for p in bodies) != sorted(DISTAL_NAMES):
        raise ValueError('Expected exactly five Revo2 distal rigid links')
    paths = []
    for body in bodies:
        colliders = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        # Current USD has one distal shell and one fixed touch shell per finger.
        if len(colliders) != 2:
            raise ValueError(f'Unexpected last-phalanx geometry at {body.GetPath()}: {len(colliders)} colliders')
        UsdShade.MaterialBindingAPI.Apply(body).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, 'physics')
        paths.extend(str(p.GetPath()) for p in colliders)
    return sorted(paths)


def _spawn_one(prim_path, cfg, *args, **kwargs):
    import isaaclab.sim as sim
    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
    from pxr import UsdShade
    root = sim.spawn_from_usd(prim_path, cfg, *args, **kwargs)
    spec = cfg.revo2_contact_spec
    material_cfg = PhysxRigidBodyMaterialCfg(
        static_friction=spec['static_friction'], dynamic_friction=spec['dynamic_friction'],
        restitution=0., compliant_contact_stiffness=spec['contact_stiffness_n_per_m'],
        compliant_contact_damping=spec['contact_damping_ns_per_m'],
        compliant_contact_acceleration_spring=False, friction_combine_mode='average')
    material = sim.spawn_rigid_body_material(str(root.GetPath())+'/LastPhalanxContactMaterial', material_cfg)
    bind_last_phalanges(root, UsdShade.Material(material))
    return root


def spawn_with_last_phalanx_contact(prim_path, cfg, *args, **kwargs):
    # Author material before the existing USD clone operation, not after env_0
    # has been copied to the other environments. No new asset/mesh generation.
    from isaaclab.sim.utils import clone
    return clone(_spawn_one)(prim_path, cfg, *args, **kwargs)


def configure_contact(cfg, path):
    spec = load_contact_config(path)
    cfg.scene.robot.spawn = deepcopy(cfg.scene.robot.spawn)
    cfg.scene.robot.spawn.revo2_contact_spec = spec
    cfg.scene.robot.spawn.func = spawn_with_last_phalanx_contact
    return spec


def verify_runtime_materials(materials, spec):
    """PhysX restitution < 0 encodes compliant spring stiffness, not bounce.

    PxMaterial::setRestitution documentation:
    https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxMaterial.html
    Damping/force mode are checked in USD; this API does not expose them.
    """
    import numpy as np
    values = np.asarray(materials)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Invalid runtime material properties')
    soft = values[:,:,2] < 0
    counts = soft.sum(axis=1)
    if not np.all(counts == 10):
        raise ValueError(f'Expected 10 PhysX compliant collision shapes per env, got {counts.tolist()}')
    np.testing.assert_allclose(values[soft], np.broadcast_to(
        [spec['static_friction'],spec['dynamic_friction'],-spec['contact_stiffness_n_per_m']],
        values[soft].shape),rtol=1e-6,atol=0)
    return counts.tolist()


def verify_contact(env):
    """Check resolved bindings on every env and log PhysX material readback.

USD spring values are authored settings, not an independent force/stiffness
measurement. get_material_properties supplies the runtime material triples.
"""
    import numpy as np
    import warp as wp
    from pxr import Usd, UsdPhysics, UsdShade
    from isaaclab.sim import get_current_stage
    robot = env.scene['robot']
    spec = getattr(robot.cfg.spawn, 'revo2_contact_spec', None)
    if spec is None:
        return None
    stage = get_current_stage()
    records = []
    counts = []
    for env_path in env.scene.env_prim_paths:
        root = stage.GetPrimAtPath(env_path+'/Robot')
        if not root: raise ValueError('Missing spawned Robot '+env_path)
        count = 0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
            parent = prim
            expected = False
            while parent and parent != root:
                if parent.GetName() in DISTAL_NAMES and parent.HasAPI(UsdPhysics.RigidBodyAPI): expected = True
                parent = parent.GetParent()
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            applied = bool(material and material.GetPrim().GetName() == 'LastPhalanxContactMaterial')
            if expected != applied: raise ValueError('Rubber material scope mismatch '+str(prim.GetPath()))
            if applied:
                p = material.GetPrim(); count += 1
                for attribute, key in (('physics:staticFriction','static_friction'),
                    ('physics:dynamicFriction','dynamic_friction'),
                    ('physxMaterial:compliantContactStiffness','contact_stiffness_n_per_m'),
                    ('physxMaterial:compliantContactDamping','contact_damping_ns_per_m')):
                    np.testing.assert_allclose(p.GetAttribute(attribute).Get(), spec[key], rtol=1e-6)
                if p.GetAttribute('physxMaterial:compliantContactAccelerationSpring').Get() is not False:
                    raise ValueError('Expected force-unit compliant contacts')
                if len(counts) == 0: records.append(str(prim.GetPath()))
        if count != 10: raise ValueError(f'Expected 10 compliant colliders in {env_path}, got {count}')
        counts.append(count)
    materials = wp.to_torch(robot.root_view.get_material_properties()).cpu().numpy()
    runtime_counts = verify_runtime_materials(materials, spec)
    result = dict(spec=spec, compliant_colliders_per_env=counts, first_env_colliders=records,
        runtime_material_shape=list(materials.shape), runtime_material_rows=np.unique(materials.reshape(-1,3),axis=0).tolist(),
        runtime_compliant_shapes_per_env=runtime_counts,
        runtime_columns=['static friction','dynamic friction','restitution; negative value = -contact stiffness [N/m] in force mode'],
        note='USD spring settings and resolved bindings verified; runtime triples read from PhysX. No calibrated pad force-displacement measurement.')
    print('[rubber contact verification]', json.dumps(result), flush=True)
    return result
