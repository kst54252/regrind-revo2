"""Render the exact logged raw IK solution at the 0.8-rad wrist jump.

This is a geometric reconstruction, NOT an actual-state screenshot or replay.
Recorded hand targets accompany raw arm IK; no object is synthesized or moved.
Existing FK/model and USD are reused; no simulation steps or asset saves.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
IMAGE = OUT/'rb3_wrist_singularity.png'
if IMAGE.exists(): raise FileExistsError(IMAGE)
SOURCE = ROOT/'outputs/diagnostics/ik120_improvement_20260907/old20_baseline'
meta = json.loads((SOURCE/'metadata.json').read_text())
with Path(meta['state_bank']).open() as f: bank = json.loads(next(f))
previous = None
with (SOURCE/'physics.jsonl').open() as f:
    for line in f:
        row = json.loads(line)
        if row['episode'] == 11 and abs(row['time_s']-1.0416666666666667)<1e-9: break
        if row['episode'] == 11: previous = row
    else: raise ValueError('Exact diagnosed physics sample is absent')
assert previous is not None
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics
kin = RB3730Kinematics(base_position=bank['base_position'],base_quaternion_xyzw=bank['base_quaternion_xyzw'])
q = np.asarray(row['raw_solve']['q'])
assert row['raw_solve']['success'] and np.isfinite(q).all()
lower, upper = kin.get_joint_limits()
assert np.all(q>=lower) and np.all(q<=upper)
wrist_p, wrist_q = kin.forward(q)
np.testing.assert_allclose(wrist_p,row['ik_input_pos'],atol=1e-5,rtol=0)
delta = q-np.asarray(previous['raw_solve']['q'])

from isaacsim import SimulationApp
app = SimulationApp({'headless':True,'width':2400,'height':1600,
                     'renderer':'RaytracedLighting','anti_aliasing':3})
try:
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    import imageio.v2 as imageio
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from shapely.geometry import Polygon
    from shapely import constrained_delaunay_triangles

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    robot = UsdGeom.Xform.Define(stage,'/Photo/Robot')
    robot.GetPrim().GetReferences().AddReference(str(kin.source_usd_path),'/World')
    def find(name):
        matches=sorted([p for p in Usd.PrimRange(robot.GetPrim()) if p.GetName()==name],
                       key=lambda p:len(str(p.GetPath())))
        if not matches: raise ValueError('Missing model prim: '+name)
        return matches[0]
    def pose(prim,r,p,reset=False):
        x=UsdGeom.Xformable(prim);x.ClearXformOpOrder();x.SetResetXformStack(reset)
        m=np.eye(4);m[:3,:3]=r;m[:3,3]=p
        x.AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))
    pose(find('link0'),kin.base_rotation,kin.base_position,True)
    for i,(offset,axis,angle) in enumerate(zip(kin.joint_offsets,kin.joint_axes,q)):
        pose(find(f'link{i+1}'),Rotation.from_rotvec(axis*angle).as_matrix(),offset)
    bt=np.eye(4);bt[:3,:3]=Rotation.from_quat(wrist_q).as_matrix();bt[:3,3]=wrist_p
    hand=Revo2Kinematics(base_transform=bt)
    np.testing.assert_array_equal([meta['joint_names'][i] for i in meta['hand_ids']],hand.joint_names)
    hand_q=np.asarray(row['hand_target'])
    for name,(r,p) in hand._forward_links(hand_q).items(): pose(find(name),r,p,True)
    for prim in Usd.PrimRange(robot.GetPrim()):
        if prim.GetName().startswith('kp_'): UsdGeom.Imageable(prim).MakeInvisible()

    cache=UsdGeom.XformCache()
    matrix=np.asarray(cache.GetLocalToWorldTransform(find('right_hand_base_link'))).T
    np.testing.assert_allclose(matrix,bt,atol=1e-8,rtol=0)
    mount=np.asarray(cache.GetLocalToWorldTransform(find('revo2_mount'))).T
    np.testing.assert_allclose(mount,bt,atol=1e-8,rtol=0)
    # Read axis directions and origins from the posed original USD link frames.
    origins=[];axes=[]
    for i,axis in enumerate(kin.joint_axes):
        t=np.asarray(cache.GetLocalToWorldTransform(find(f'link{i+1}'))).T
        origins.append(t[:3,3]);axes.append(t[:3,:3]@axis)
    origins=np.array(origins);axes=np.array(axes)
    angle=float(np.rad2deg(np.arccos(np.clip(axes[3]@axes[5],-1,1))))
    np.testing.assert_allclose(angle,np.rad2deg(q[4]),atol=1e-7)
    print('[photo] raw IK mounted-base FK/USD verified; wrist-axis angle deg:',angle,flush=True)

    center=np.array([.115,.005,.185])
    normal=np.array([.1,-1.,.32]);normal/=np.linalg.norm(normal)
    right=np.cross([0.,0.,1.],normal);right/=np.linalg.norm(right)
    up=np.cross(normal,right)
    overlay=center+normal*.65
    def xy(p): return np.array([(p-center)@right,(p-center)@up])
    def plane(x,y,d=0.): return overlay+right*x+up*y+normal*d
    def mat(name,c,flat=False):
        m=UsdShade.Material.Define(stage,'/Photo/Materials/'+name)
        s=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Shader');s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*((0,0,0) if flat else c)))
        s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
        if flat:s.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*c))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface');return m
    def bind(prim,m): UsdShade.MaterialBindingAPI.Apply(prim).Bind(m,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    def mesh(name,pts,counts,ids,m):
        s=UsdGeom.Mesh.Define(stage,'/Photo/'+name);s.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        s.CreateFaceVertexCountsAttr(counts);s.CreateFaceVertexIndicesAttr(ids)
        s.CreateSubdivisionSchemeAttr('none');s.CreateDoubleSidedAttr(True);bind(s.GetPrim(),m);return s
    white=mat('White',(1,1,1),True);dark=mat('Dark',(.02,.035,.05),True)
    muted=mat('Muted',(.16,.20,.24),True)
    blue=mat('Wrist1',(.015,.25,.85),True);red=mat('Wrist3',(.85,.03,.12),True)
    # Keep arm materials, use neutral hand material so the silhouettes read clearly.
    bind(stage.GetPrimAtPath('/Photo/Robot/revo2_right'),mat('Hand',(.23,.29,.36)))
    font=FontProperties(family='DejaVu Sans',weight='bold')
    def text(name,value,x,y,size,m=dark):
        geometry=Polygon()
        for poly in TextPath((0,0),value,size=size,prop=font).to_polygons():
            if len(poly)>=3:geometry=geometry.symmetric_difference(Polygon(poly))
        tris=list(constrained_delaunay_triangles(geometry).geoms)
        pts=[plane(x+a,y+b,.02) for t in tris for a,b in list(t.exterior.coords)[:3]]
        mesh('Text/'+name,pts,[3]*len(tris),list(range(len(pts))),m)
    def line(name,pts,m,width=.0012):
        c=UsdGeom.BasisCurves.Define(stage,'/Photo/Lines/'+name)
        c.CreateTypeAttr('linear');c.CreateCurveVertexCountsAttr([len(pts)])
        c.CreatePointsAttr([Gf.Vec3f(*p) for p in pts]);c.CreateWidthsAttr([width])
        c.SetWidthsInterpolation('constant');bind(c.GetPrim(),m)
    def dot(name,x,y,r,m):
        pts=[plane(x,y,.012)]+[plane(x+r*np.cos(a),y+r*np.sin(a),.012) for a in np.linspace(0,2*np.pi,48,endpoint=False)]
        mesh(name,pts,[3]*48,[j for i in range(48) for j in (0,i+1,(i+1)%48+1)],m)
    def arrow(name,p0,p1,m,width=.0024):
        a=xy(p0);b=xy(p1)
        line(name,[plane(*a),plane(*b)],m,width)
        v=(b-a)/np.linalg.norm(b-a);s=np.array([-v[1],v[0]])
        mesh('Arrows/'+name,[plane(*b,.005),plane(*(b-v*.014+s*.005),.005),plane(*(b-v*.014-s*.005),.005)],
             [3],[0,1,2],m)
    arrow('AxisWrist1',origins[3]-axes[3]*.025,origins[3]+axes[3]*.26,blue)
    arrow('AxisWrist3',origins[5]-axes[5]*.015,origins[5]+axes[5]*.20,red)
    for i,name,m in [(3,'Wrist1',blue),(5,'Wrist3',red)]:
        x,y=xy(origins[i]);dot(name+'Rim',x,y,.0075,white);dot(name+'Dot',x,y,.005,m)
    a=xy(origins[3]);b=xy(origins[5])
    text('Wrist1','wrist1 axis',a[0]-.038,a[1]+.061,.015,blue)
    line('LabelWrist1',[plane(a[0]-.006,a[1]+.051),plane(*a)],blue,.0008)
    text('Wrist3','wrist3 axis',b[0]-.012,b[1]+.076,.015,red)
    line('LabelWrist3',[plane(b[0]+.013,b[1]+.061),plane(*b)],red,.0008)

    text('Title','RB3 WRIST SINGULARITY',-.43,.271,.023)
    text('Sub','Recorded IK solution  |  placement 11  |  t = 1.041667 s (post-step record)',-.43,.246,.009)
    text('Angle',f'Axis separation: {angle:.3f} deg',-.005,-.085,.016)
    text('Wrist2',f'wrist2 = {np.rad2deg(q[4]):.3f} deg',-.005,-.112,.012)
    text('Jump',f'One IK step: wrist1 {np.rad2deg(delta[3]):+.1f} deg / wrist3 {np.rad2deg(delta[5]):+.1f} deg',-.005,-.142,.0088)
    text('Small','Wrist target moves only 2.317 mm / 0.154 deg',-.005,-.165,.009)
    text('Legend','Blue: wrist1 rotation axis    Red: wrist3 rotation axis',-.43,-.235,.010)
    text('Footer','Geometric reconstruction of raw IK, not actual robot tracking. No physics step or policy call.',-.43,-.265,.008)
    text('Timing','Command applied at 1.033333 s; state record at 1.041667 s. Overlay uses exact projected axes.',-.43,-.282,.0075)

    mesh('Backdrop',[center-normal*.4+right*x+up*y for x,y in [(-3,-3),(3,-3),(3,3),(-3,3)]],
         [4],[0,1,2,3],white)
    dome=UsdLux.DomeLight.Define(stage,'/Photo/Dome');dome.CreateIntensityAttr(900.)
    key=UsdLux.DistantLight.Define(stage,'/Photo/Key');key.CreateIntensityAttr(1800.)
    key.AddRotateXYZOp().Set(Gf.Vec3f(25.,-35.,-30.))
    camera=UsdGeom.Camera.Define(stage,'/Photo/Camera');camera.CreateProjectionAttr('orthographic')
    camera.CreateHorizontalApertureAttr(9.6);camera.CreateVerticalApertureAttr(6.4)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
    eye=center+normal*2.
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*center),Gf.Vec3d(*up)).GetInverse())
    product=rep.create.render_product(str(camera.GetPath()),(2400,1600))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
    rep.orchestrator.set_capture_on_play(False)
    for _ in range(60):app.update()
    rep.orchestrator.step(rt_subframes=32,delta_time=0.,pause_timeline=True)
    pixels=np.asarray(rgb.get_data())[:,:,:3].astype(np.uint8)
    if pixels.shape!=(1600,2400,3) or pixels.std()<5:raise ValueError('Invalid capture')
    imageio.imwrite(IMAGE,pixels)
    metadata=dict(source=str(SOURCE.relative_to(ROOT)),source_metadata_sha256=hashlib.sha256((SOURCE/'metadata.json').read_bytes()).hexdigest(),
        episode=row['episode'],physics_step=row['physics_step'],episode_step=row['episode_step'],
        record_state_time_s=row['time_s'],command_time_s=row['controller']['command_time_s'],
        pose_source='raw_solve.q, NOT q_cmd or q_actual',q_ik=q.tolist(),
        q_cmd=row['q_cmd'],q_actual=np.asarray(row['state']['all_q'])[meta['arm_ids']].tolist(),
        hand_pose_source='recorded hand_target in documented leader order',hand_q=hand_q.tolist(),
        axis_origins_m=origins.tolist(),axis_directions=axes.tolist(),wrist_axis_angle_deg=angle,
        joint_step_rad=delta.tolist(),wrist_fk_pos=wrist_p.tolist(),wrist_fk_quat_xyzw=wrist_q.tolist(),
        fk_usd_verified=True,physics_stepped=False,original_assets_saved=False,
        image=str(IMAGE.relative_to(ROOT)))
    (OUT/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('[photo complete]',IMAGE,flush=True)
finally:
    app.close()
