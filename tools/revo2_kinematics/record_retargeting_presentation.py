"""Synchronized skeleton / real USD-mesh presentation; no policy or physics stepping.

Uses the existing FK (including mimic), saved MANO21 data and Isaac RTX renderer.
All USD edits are session-only; input trajectories and assets are never saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', default='20200709_143747_left')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics
    from tools.dexycb_world_transform.visualize_retargeted_world_interactive import MANO21_CHAINS, REVO2_CHAINS

    world = ROOT / 'outputs/isaac/dexycb' / args.sequence / 'world_trajectory.h5'
    retarget = ROOT / 'outputs/retargeted/dexycb' / args.sequence / 'revo2_retargeted.h5'
    with h5py.File(world) as file:
        data = {key: value[()] for key, value in file.items()}
    with h5py.File(retarget) as file:
        saved = file['robot_keypoints'][:]
        np.testing.assert_array_equal(file['source_frame_indices'][:], data['source_frame_indices'])
        np.testing.assert_array_equal(file['robot_joints'][:], data['revo2_joints'])
        if not file['solver_success'][:].all():
            raise ValueError('Retargeting has failed frames; do not present these as valid FK')
    if data['quaternion_order'].decode() != 'wxyz':
        raise ValueError('Expected explicit world wxyz quaternion convention')
    human = data['mano_joint_world_mano21']
    q = data['revo2_joints']
    rotation = Rotation.from_quat(data['wrist_quat_world'][:, [1, 2, 3, 0]]).as_matrix()
    kin = Revo2Kinematics()
    robot = np.stack([kin.get_keypoints(joints) for joints in q])
    robot = np.einsum('tij,tkj->tki', rotation, robot) + data['wrist_pos_world'][:, None, :]
    transform = data['T_world_camera']
    expected = saved @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(robot, expected, atol=1e-6, rtol=0)
    if human.shape != robot.shape or not np.isfinite(human).all() or not np.isfinite(robot).all():
        raise ValueError('Invalid skeletons')

    # Import Isaac only after validating inputs; renderer only, timeline stays stopped.
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 960, 'height': 900,
                         'renderer': 'RaytracedLighting', 'anti_aliasing': 3})
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Sdf
    import imageio.v2 as imageio

    args.output.mkdir(parents=True)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)

    def pose_op(path):
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(True)
        return xform.AddTransformOp()

    def set_pose(op, rot, pos):
        matrix = np.eye(4)
        matrix[:3, :3] = rot
        matrix[:3, 3] = pos
        op.Set(Gf.Matrix4d(matrix.T.tolist()))

    hand = UsdGeom.Xform.Define(stage, '/Presentation/Hand')
    hand.GetPrim().GetReferences().AddReference(str(ROOT / 'USD/revo2_right/revo2_right.usda'))
    # Neutral presentation material: make the white hand readable on a white
    # slide. This overrides only this unsaved stage, not the robot USD asset.
    material = UsdShade.Material.Define(stage, '/Presentation/HandMaterial')
    shader = UsdShade.Shader.Define(stage, '/Presentation/HandMaterial/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.32, .40, .52))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(hand.GetPrim()).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    # Match unique skeletal link prims, not same-name referenced visual children.
    link_ops = {}
    for name in kin._forward_links(q[0]):
        matches = [p for p in Usd.PrimRange(hand.GetPrim()) if p.GetName() == name]
        if not matches:
            raise ValueError('USD link missing: ' + name)
        matches.sort(key=lambda p: len(str(p.GetPath())))
        if len(matches) > 1 and len(str(matches[0].GetPath())) == len(str(matches[1].GetPath())):
            raise ValueError('Ambiguous USD link: ' + name)
        link_ops[name] = pose_op(matches[0].GetPath())

    can = UsdGeom.Xform.Define(stage, '/Presentation/Can')
    can.GetPrim().GetReferences().AddReference(str(ROOT / '007_tuna_fish_can/textured_simple.usd'))
    for prim in Usd.PrimRange(can.GetPrim()):
        opacity = prim.GetAttribute('inputs:opacity')
        if opacity.IsValid():
            opacity.Set(1.)  # Same transient fix as the maintained Isaac replay.
    can_op = pose_op(can.GetPath())
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)

    skeleton = UsdGeom.Xform.Define(stage, '/Presentation/Skeleton')
    dots, curves = [], []
    for label, color, chains in [('MANO', (.04, .32, .85), MANO21_CHAINS),
                                  ('Revo2', (1., .24, .025), REVO2_CHAINS)]:
        spheres = []
        for index in range(21):
            sphere = UsdGeom.Sphere.Define(stage, f'{skeleton.GetPath()}/{label}_{index:02}')
            sphere.CreateRadiusAttr(.0023)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            spheres.append(sphere.AddTranslateOp())
        curve = UsdGeom.BasisCurves.Define(stage, f'{skeleton.GetPath()}/{label}_bones')
        curve.CreateTypeAttr('linear')
        curve.CreateCurveVertexCountsAttr([len(chain) for chain in chains])
        curve.CreateWidthsAttr([.0018])
        curve.SetWidthsInterpolation('constant')
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        dots.append(spheres)
        curves.append((curve, chains))

    floor = UsdGeom.Cube.Define(stage, '/Presentation/Floor')
    floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set(Gf.Vec3d(.36, 0., -.013))
    floor.AddScaleOp().Set(Gf.Vec3d(200., 200., .02))
    floor.CreateDisplayColorAttr([Gf.Vec3f(.93, .95, .98)])
    light = UsdLux.DomeLight.Define(stage, '/Presentation/Light')
    light.CreateIntensityAttr(1100.)
    light.CreateColorAttr(Gf.Vec3f(1., 1., 1.))
    key = UsdLux.DistantLight.Define(stage, '/Presentation/Key')
    key.CreateIntensityAttr(1800.)
    key.AddRotateXYZOp().Set(Gf.Vec3f(25., -35., 20.))

    points = np.concatenate([human.reshape(-1, 3), robot.reshape(-1, 3)])
    center = (points.min(0) + points.max(0)) / 2
    center[2] = (points[:, 2].max() + 0.) / 2
    eye = center + np.array([.7, .65, .47])
    camera = UsdGeom.Camera.Define(stage, '/Presentation/Camera')
    camera.CreateProjectionAttr('orthographic')
    camera.CreateHorizontalApertureAttr(4.7)
    camera.CreateVerticalApertureAttr(4.7 * 900 / 960)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0, 0, 1)).GetInverse())
    product = rep.create.render_product(str(camera.GetPath()), (960, 900))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    rep.orchestrator.set_capture_on_play(False)

    def apply_frame(index):
        for name, (r, p) in kin._forward_links(q[index]).items():
            set_pose(link_ops[name], rotation[index] @ r,
                     rotation[index] @ p + data['wrist_pos_world'][index])
        object_r = Rotation.from_quat(data['object_quat_world'][index, [1, 2, 3, 0]]).as_matrix()
        set_pose(can_op, object_r, data['object_pos_world'][index])
        for spheres, (curve, chains), coords in zip(dots, curves, [human[index], robot[index]]):
            for op, pos in zip(spheres, coords):
                op.Set(Gf.Vec3d(*pos))
            curve.CreatePointsAttr([Gf.Vec3f(*coords[j]) for chain in chains for j in chain])
        # USD composition must match the reused FK at every recorded frame.
        cache = UsdGeom.XformCache()
        for name, (r, p) in kin._forward_links(q[index]).items():
            matrix = np.asarray(cache.GetLocalToWorldTransform(link_ops[name].GetAttr().GetPrim())).T
            np.testing.assert_allclose(matrix[:3, 3], rotation[index] @ p + data['wrist_pos_world'][index], atol=1e-7)
            np.testing.assert_allclose(matrix[:3, :3], rotation[index] @ r, atol=1e-7)

    exit_code = 0
    try:
        apply_frame(0)
        for _ in range(60):
            app.update()
        for panel in ('skeleton', 'isaac'):
            UsdGeom.Imageable(hand).GetVisibilityAttr().Set('invisible' if panel == 'skeleton' else 'inherited')
            UsdGeom.Imageable(skeleton).GetVisibilityAttr().Set('inherited' if panel == 'skeleton' else 'invisible')
            with imageio.get_writer(str(args.output / f'{panel}.mp4'), fps=10, codec='libx264',
                                    quality=9, macro_block_size=1) as writer:
                for index in range(len(q)):
                    apply_frame(index)
                    for _ in range(8):
                        app.update()
                    # A stopped timeline needs an explicit capture; delta=0 does
                    # not advance physics or the recorded trajectory frame.
                    rep.orchestrator.step(rt_subframes=8, delta_time=0., pause_timeline=True)
                    pixels = np.asarray(rgb.get_data())
                    if pixels.size == 0:
                        raise RuntimeError('Isaac renderer returned an empty RGB frame')
                    frame = pixels[:, :, :3].astype(np.uint8)
                    writer.append_data(frame)
                    if index in (0, len(q)//2, len(q)-1):
                        imageio.imwrite(args.output / f'{panel}_{index:03}.png', frame)
                    print(f'[presentation] {panel} {index+1}/{len(q)}', flush=True)
        metadata = dict(sequence=args.sequence, frames=len(q), source_fps=float(data['fps']),
                        playback_fps=10, playback_scale=10/float(data['fps']), final_hold_s=1,
                        world_file=str(world), world_sha256=hashlib.sha256(world.read_bytes()).hexdigest(),
                        retarget_file=str(retarget), fk_saved_error_max_m=float(np.linalg.norm(robot-expected, axis=-1).max()),
                        mode='kinematic USD pose replay using existing FK; no RL, IK, physics or grasp-success claim',
                        camera_eye=eye.tolist(), camera_target=center.tolist(),
                        frame_indices=data['source_frame_indices'].tolist())
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        rgb.detach([product])
        app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()
