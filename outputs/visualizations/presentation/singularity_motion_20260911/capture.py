"""Two short Isaac RTX clips of recorded BASELINE actual state (not raw IK).

Renderer-only reconstruction; no new policy, IK, physics, or source-asset edits.
Reuse verified arm/hand transforms. Measured distal angles override ideal mimic
angles for the render, so hand targets are not substituted for actual states.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'outputs/diagnostics/ik120_improvement_20260907/old20_baseline'
if (OUT/'capture.json').exists():
    raise FileExistsError(OUT/'capture.json')
meta = json.loads((SOURCE/'metadata.json').read_text())
with Path(meta['state_bank']).open() as stream:
    bank = json.loads(next(stream))
episodes = {0: [], 11: []}
with (SOURCE/'physics.jsonl').open() as stream:
    for line in stream:
        row = json.loads(line)
        if row['episode'] in episodes and 88 <= row['episode_step'] <= 148:
            episodes[row['episode']].append(row)
times = np.arange(88,149)/120
for rows in episodes.values():
    np.testing.assert_allclose([r['time_s'] for r in rows], times, rtol=0, atol=1e-9)
assert meta['physics_dt'] == 1/120

from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics
kin = RB3730Kinematics(base_position=bank['base_position'], base_quaternion_xyzw=bank['base_quaternion_xyzw'])
hand = Revo2Kinematics()
arm_ids = [meta['joint_names'].index(n) for n in kin.joint_names]
hand_ids = [meta['joint_names'].index(n) for n in hand.joint_names]
np.testing.assert_array_equal(arm_ids, meta['arm_ids'])
np.testing.assert_array_equal(hand_ids, meta['hand_ids'])

def measured_hand_links(all_q):
    leader_q = np.array(all_q)[hand_ids]
    links = hand._forward_links(leader_q)
    for finger in ('thumb','index','middle','ring','pinky'):
        proximal = f'right_{finger}_proximal_link'
        distal = f'right_{finger}_distal_link'
        touch = f'right_{finger}_touch_link'
        value = all_q[meta['joint_names'].index(f'right_{finger}_distal_joint')]
        links[distal] = hand._compose(links[proximal], hand._moving_pose(distal, 'X' if finger=='thumb' else 'Y', value, leader_q))
        links[touch] = hand._compose(links[distal], hand._fixed_pose(touch, leader_q))
    return links

# Choose ONE camera from both complete windows, not from whichever clip looks best.
cloud = []
errors = []
for rows in episodes.values():
    for row in rows:
        q = np.array(row['state']['all_q'])[arm_ids]
        p, quat = kin.forward(q)
        pe = np.linalg.norm(p-row['state']['wrist_pos'])
        oe = (Rotation.from_quat(quat).inv()*Rotation.from_quat(row['state']['wrist_quat'])).magnitude()
        errors.append((pe,oe))
        if pe>1e-6 or oe>1e-5:
            raise ValueError('Recorded actual state does not match the verified FK')
        cloud.extend(kin.forward_chain_points(q))
        r = Rotation.from_quat(quat).as_matrix()
        cloud.extend(r@lp+p for _,lp in measured_hand_links(row['state']['all_q']).values())
normal = np.array([.1,-1.,.36]); normal /= np.linalg.norm(normal)
right = np.cross([0.,0.,1.],normal); right /= np.linalg.norm(right)
up = np.cross(normal,right)
cloud = np.array(cloud)
screen_x, screen_y = cloud@right, cloud@up
center = right*(screen_x.min()+screen_x.max())/2 + up*(screen_y.min()+screen_y.max())/2 + normal*np.mean(cloud@normal)
width = max(np.ptp(screen_x)+.20, 2*(np.ptp(screen_y)+.17))

from isaacsim import SimulationApp
app = SimulationApp({'headless':True,'width':1600,'height':900,
                     'renderer':'RaytracedLighting','anti_aliasing':3})
status = 0
try:
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade,UsdPhysics
    from PIL import Image,ImageDraw,ImageFont
    import imageio.v2 as imageio

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    robot = UsdGeom.Xform.Define(stage,'/Capture/Robot')
    robot.GetPrim().GetReferences().AddReference(str(kin.source_usd_path),'/World')

    def find(name):
        matches = sorted([p for p in Usd.PrimRange(robot.GetPrim()) if p.GetName()==name],key=lambda p:len(str(p.GetPath())))
        if not matches: raise ValueError('Missing robot link '+name)
        return matches[0]

    def op(prim, reset=False):
        x = UsdGeom.Xformable(prim); x.ClearXformOpOrder(); x.SetResetXformStack(reset)
        return x.AddTransformOp()

    def pose(transform, rotation, translation):
        m = np.eye(4); m[:3,:3] = rotation; m[:3,3] = translation
        transform.Set(Gf.Matrix4d(m.T.tolist()))

    link0 = op(find('link0'),True)
    pose(link0,kin.base_rotation,kin.base_position)
    arm_ops = [op(find(f'link{i+1}')) for i in range(6)]
    hand_ops = {name:op(find(name),True) for name in hand._forward_links(np.zeros(6))}
    for prim in Usd.PrimRange(robot.GetPrim()):
        if prim.GetName().startswith('kp_'): UsdGeom.Imageable(prim).MakeInvisible()

    can = UsdGeom.Xform.Define(stage,'/Capture/Can')
    can.GetPrim().GetReferences().AddReference(str(ROOT/'007_tuna_fish_can/textured_simple.usd'))
    can_op = op(can.GetPrim(),True)
    for prim in Usd.PrimRange(can.GetPrim()):
        opacity = prim.GetAttribute('inputs:opacity')
        if opacity.IsValid(): opacity.Set(1.)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)

    def material(name,color,emissive=False):
        m = UsdShade.Material.Define(stage,'/Capture/Materials/'+name)
        s = UsdShade.Shader.Define(stage,str(m.GetPath())+'/Shader'); s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
        if emissive:s.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
        return m
    tablemat = material('Table',(.60,.63,.67))
    basemat = material('Pedestal',(.25,.28,.32))
    floormat = material('Floor',(.85,.87,.89))
    handmat = material('Hand',(.19,.24,.30))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/Capture/Robot/revo2_right')).Bind(handmat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    layout = json.loads((ROOT/'config/workcell/rb3_revo2_table.json').read_text())
    def box(name,xyz,size,mat):
        cube = UsdGeom.Cube.Define(stage,'/Capture/Workcell/'+name); cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*xyz)); cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    box('Pedestal',layout['robot_base']['center'],layout['robot_base']['size'],basemat)
    t = layout['table']
    box('Table',t['center_xy']+[t['top_z']-t['top_thickness']/2],t['size_xy']+[t['top_thickness']],tablemat)
    for i,xy in enumerate(t['leg_centers_xy']):
        z1,z2 = layout['floor_z'],t['top_z']-t['top_thickness']
        box('Leg'+str(i),xy+[(z1+z2)/2],t['leg_size_xy']+[z2-z1],basemat)
    box('Floor',[0,0,layout['floor_z']-.01],[20,20,.02],floormat)
    ticks = []
    for j,color in [(3,(.03,.22,.9)),(5,(.9,.04,.09))]:
        curve = UsdGeom.BasisCurves.Define(stage,f'/Capture/WristTick{j}')
        curve.CreateTypeAttr('linear'); curve.CreateCurveVertexCountsAttr([2]); curve.CreateWidthsAttr([.004])
        curve.SetWidthsInterpolation('constant')
        UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material(f'Tick{j}',color,True))
        ticks.append((j,curve))
    dome = UsdLux.DomeLight.Define(stage,'/Capture/Dome'); dome.CreateIntensityAttr(900.)
    sun = UsdLux.DistantLight.Define(stage,'/Capture/Sun'); sun.CreateIntensityAttr(1500.)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(25.,-30.,-25.))
    cam = UsdGeom.Camera.Define(stage,'/Capture/Camera'); cam.CreateProjectionAttr('orthographic')
    cam.CreateHorizontalApertureAttr(float(width*10)); cam.CreateVerticalApertureAttr(float(width*5))
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01,30.))
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(center+normal*3)),Gf.Vec3d(*center),Gf.Vec3d(*up)).GetInverse())
    product = rep.create.render_product(str(cam.GetPath()),(1600,800))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
    rep.orchestrator.set_capture_on_play(False)
    fontfile = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(fontfile,22)
    titlefont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',30)

    def apply(row):
        all_q = np.array(row['state']['all_q']); q = all_q[arm_ids]
        for transform,offset,axis,angle in zip(arm_ops,kin.joint_offsets,kin.joint_axes,q):
            pose(transform,Rotation.from_rotvec(axis*angle).as_matrix(),offset)
        p,quat = kin.forward(q); r = Rotation.from_quat(quat).as_matrix()
        for name,(local_r,local_p) in measured_hand_links(all_q).items():
            pose(hand_ops[name],r@local_r,r@local_p+p)
        state = row['state']['object_state']
        pose(can_op,Rotation.from_quat(state[3:7]).as_matrix(),state[:3])
        cache = UsdGeom.XformCache()
        for name in ('right_hand_base_link','revo2_mount'):
            m = np.asarray(cache.GetLocalToWorldTransform(find(name))).T
            np.testing.assert_allclose(m[:3,3],p,rtol=0,atol=1e-8)
            np.testing.assert_allclose(m[:3,:3],r,rtol=0,atol=1e-8)
        for j,curve in ticks:
            m = np.asarray(cache.GetLocalToWorldTransform(find(f'link{j+1}'))).T
            length = .066 if j==3 else .045
            curve.CreatePointsAttr([Gf.Vec3f(*m[:3,3]),Gf.Vec3f(*(m[:3,3]+length*m[:3,0]))])
        return q

    apply(episodes[0][0])
    for _ in range(60):app.update()
    captures = []
    for episode,rows in episodes.items():
        name = 'baseline_regular' if episode==0 else 'baseline_singularity'
        path = OUT/f'{name}.mp4'
        if path.exists():raise FileExistsError(path)
        with imageio.get_writer(str(path),fps=30,codec='libx264',macro_block_size=1,
                                ffmpeg_params=['-crf','17','-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
            for i,row in enumerate(rows):
                q = apply(row)
                for _ in range(3):app.update()
                rep.orchestrator.step(rt_subframes=8 if i else 32,delta_time=0.,pause_timeline=True)
                pixels = np.asarray(rgb.get_data())[:,:,:3].astype(np.uint8)
                if pixels.shape!=(800,1600,3) or pixels.std()<5:raise ValueError('Bad/blank render')
                frame = Image.new('RGB',(1600,900),'white'); frame.paste(Image.fromarray(pixels),(0,64))
                draw = ImageDraw.Draw(frame)
                draw.text((25,6),'REGULAR WRIST  |  BASELINE' if episode==0 else 'NEAR-SINGULAR WRIST  |  BASELINE',font=titlefont,fill=(25,35,50))
                draw.text((950,10),'Recorded actual state  |  0.25x',font=font,fill=(50,60,75))
                angle = np.rad2deg(np.arcsin(abs(np.sin(q[4]))))
                draw.text((25,869),f't = {row["time_s"]:.4f} s   |   wrist axis angle = {angle:.2f} deg',font=font,fill=(30,40,55))
                draw.text((825,869),'Blue / red ticks: wrist1 / wrist3   |   Isaac RTX replay',font=font,fill=(50,60,75))
                frame = np.asarray(frame)
                writer.append_data(frame)
                if i in (0,37,60):imageio.imwrite(OUT/f'{name}_{i:03}.png',frame)
                if i%10==0:print(f'[capture] {name} {i+1}/{len(rows)}',flush=True)
            for _ in range(30):writer.append_data(frame)
        captures.append(dict(file=path.name,episode=episode,rendered_frames=len(rows),final_hold_frames=30,
                             source_steps=[88,148],source_time_s=times.tolist(),actual_arm_q=[np.array(x['state']['all_q'])[arm_ids].tolist() for x in rows]))
    result = dict(source=str(SOURCE),source_metadata_sha256=hashlib.sha256((SOURCE/'metadata.json').read_bytes()).hexdigest(),
        source_checkpoint=meta['checkpoint'],physics_dt=meta['physics_dt'],video_fps=30,playback_scale=.25,
        duration_s=91/30,mode='Recorded actual physics states reconstructed in Isaac RTX; NO new policy, IK or physics stepping',
        actual_state_fields=['state.all_q (leaders AND followers)','state.object_state'],
        source_assets_saved=False,original_defaults_changed=False,quaternion_order='xyzw',
        max_fk_vs_recorded_actual_position_m=float(np.max(errors,axis=0)[0]),
        max_fk_vs_recorded_actual_orientation_rad=float(np.max(errors,axis=0)[1]),
        same_camera=True,camera_center=center.tolist(),view_normal=normal.tolist(),view_width_m=width,clips=captures)
    (OUT/'capture.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[complete]',OUT,flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    status=1
finally:
    app.close(exit_code=status)
