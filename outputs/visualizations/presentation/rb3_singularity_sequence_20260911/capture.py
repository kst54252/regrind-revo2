"""Three consecutive raw-IK poses from the recorded wrist singularity.

Identical orthographic view/scale; isolated USD presentation translations only.
No policy, physics steps, new IK solves, or edits to source assets. Times are
post-step log times, with pre-step command times saved separately. Compare
raw IK, NOT the rate-limited commands or actual robot motion.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
IMAGE=OUT/'before_peak_after.png'
if (OUT/'capture.json').exists(): raise FileExistsError(OUT/'capture.json')
SOURCE=ROOT/'outputs/diagnostics/ik120_improvement_20260907/old20_baseline'
meta=json.loads((SOURCE/'metadata.json').read_text())
with Path(meta['state_bank']).open() as f:bank=json.loads(next(f))
times=np.array([124,125,126])/120
rows=[]
with (SOURCE/'physics.jsonl').open() as f:
    for line in f:
        row=json.loads(line)
        if row['episode']==11 and np.min(abs(times-row['time_s']))<1e-9:rows.append(row)
        if row['episode']>11:break
np.testing.assert_allclose([r['time_s'] for r in rows],times,atol=1e-10,rtol=0)
np.testing.assert_allclose(np.diff(times),meta['physics_dt'],atol=1e-10,rtol=0)
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics
kin=RB3730Kinematics(base_position=bank['base_position'],base_quaternion_xyzw=bank['base_quaternion_xyzw'])
q=np.array([r['raw_solve']['q'] for r in rows])
targets=np.array([r['ik_input_pos'] for r in rows])
quats=np.array([r['ik_input_quat'] for r in rows])
dp=np.diff(targets,axis=0)
dr=(Rotation.from_quat(quats[1:])*Rotation.from_quat(quats[:-1]).inv()).magnitude()
dq=np.diff(q,axis=0)
lower,upper=kin.get_joint_limits()
assert np.isfinite(q).all() and (q>=lower).all() and (q<=upper).all()
assert all(r['raw_solve']['success'] for r in rows)

from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':4800,'height':2000,
                   'renderer':'RaytracedLighting','anti_aliasing':3})
exit_code=0
try:
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade
    import imageio.v2 as imageio
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from shapely.geometry import Polygon
    from shapely import constrained_delaunay_triangles

    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage,'/Photo/Global')
    center=np.array([.115,.005,.185])
    normal=np.array([.1,-1.,.32]);normal/=np.linalg.norm(normal)
    right=np.cross([0.,0.,1.],normal);right/=np.linalg.norm(right)
    up=np.cross(normal,right);overlay=center+normal*.65
    def plane(x,y,d=0.):return overlay+right*x+up*y+normal*d
    def xy(p):return np.array([(p-center)@right,(p-center)@up])
    def pose(prim,r,p,reset=False):
        x=UsdGeom.Xformable(prim);x.ClearXformOpOrder();x.SetResetXformStack(reset)
        t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;x.AddTransformOp().Set(Gf.Matrix4d(t.T.tolist()))
    def mat(name,c,flat=False):
        m=UsdShade.Material.Define(stage,'/Photo/Materials/'+name)
        s=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Shader');s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*((0,0,0) if flat else c)))
        s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
        if flat:s.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*c))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface');return m
    def bind(prim,m):UsdShade.MaterialBindingAPI.Apply(prim).Bind(m,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    def mesh(name,pts,counts,ids,m):
        p=UsdGeom.Mesh.Define(stage,'/Photo/'+name);p.CreatePointsAttr([Gf.Vec3f(*v) for v in pts])
        p.CreateFaceVertexCountsAttr(counts);p.CreateFaceVertexIndicesAttr(ids)
        p.CreateSubdivisionSchemeAttr('none');p.CreateDoubleSidedAttr(True);bind(p.GetPrim(),m);return p
    white=mat('White',(1,1,1),True);dark=mat('Dark',(.02,.03,.05),True)
    muted=mat('Muted',(.15,.18,.22),True);light=mat('Light',(.7,.75,.8),True)
    blue=mat('Wrist1',(.015,.25,.85),True);red=mat('Wrist3',(.85,.03,.12),True)
    green=mat('WristPoint',(.015,.5,.18),True);handmat=mat('Hand',(.23,.29,.36))
    font=FontProperties(family='DejaVu Sans',weight='bold')
    def text(name,value,x,y,size,m=dark):
        g=Polygon()
        for p in TextPath((0,0),value,size=size,prop=font).to_polygons():
            if len(p)>=3:g=g.symmetric_difference(Polygon(p))
        tris=list(constrained_delaunay_triangles(g).geoms)
        vertices=[plane(x+a,y+b,.03) for tri in tris for a,b in list(tri.exterior.coords)[:3]]
        mesh(name,vertices,[3]*len(tris),list(range(len(vertices))),m)
    def line(name,pts,m,width=.001):
        c=UsdGeom.BasisCurves.Define(stage,'/Photo/'+name);c.CreateTypeAttr('linear')
        c.CreateCurveVertexCountsAttr([len(pts)]);c.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        c.CreateWidthsAttr([width]);c.SetWidthsInterpolation('constant');bind(c.GetPrim(),m)
    def dot(name,x,y,r,m,depth=.012):
        vertices=[plane(x,y,depth)]+[plane(x+r*np.cos(a),y+r*np.sin(a),depth) for a in np.linspace(0,2*np.pi,48,endpoint=False)]
        mesh(name,vertices,[3]*48,[j for i in range(48) for j in (0,i+1,(i+1)%48+1)],m)
    def arrow(name,p0,p1,m):
        a=xy(p0);b=xy(p1);v=(b-a)/np.linalg.norm(b-a);side=np.array([-v[1],v[0]])
        line(name+'Line',[plane(*a),plane(*b)],m,.002)
        mesh(name+'Head',[plane(*b,.004),plane(*(b-v*.012+side*.004),.004),plane(*(b-v*.012-side*.004),.004)],
             [3],[0,1,2],m)
    captures=[]
    for i,(row,qi) in enumerate(zip(rows,q)):
        shift_x=(i-1)*1.0;shift=right*shift_x;prefix=f'Panel{i}'
        UsdGeom.Xform.Define(stage,'/Photo/'+prefix)
        robot=UsdGeom.Xform.Define(stage,'/Photo/'+prefix+'/Robot')
        robot.GetPrim().GetReferences().AddReference(str(kin.source_usd_path),'/World')
        def find(name):
            matches=sorted([p for p in Usd.PrimRange(robot.GetPrim()) if p.GetName()==name],key=lambda p:len(str(p.GetPath())))
            if not matches:raise ValueError('Missing model prim: '+name)
            return matches[0]
        pose(find('link0'),kin.base_rotation,kin.base_position+shift,True)
        for j,(off,axis,angle) in enumerate(zip(kin.joint_offsets,kin.joint_axes,qi)):
            pose(find(f'link{j+1}'),Rotation.from_rotvec(axis*angle).as_matrix(),off)
        p,rot=kin.forward(qi);bt=np.eye(4);bt[:3,:3]=Rotation.from_quat(rot).as_matrix();bt[:3,3]=p+shift
        np.testing.assert_allclose(p,targets[i],atol=1e-5,rtol=0)
        assert (Rotation.from_quat(rot)*Rotation.from_quat(quats[i]).inv()).magnitude()<1e-3
        hand=Revo2Kinematics(base_transform=bt)
        np.testing.assert_array_equal([meta['joint_names'][j] for j in meta['hand_ids']],hand.joint_names)
        for name,(r,pt) in hand._forward_links(row['hand_target']).items():pose(find(name),r,pt,True)
        for prim in Usd.PrimRange(robot.GetPrim()):
            if prim.GetName().startswith('kp_'):UsdGeom.Imageable(prim).MakeInvisible()
        bind(stage.GetPrimAtPath(str(robot.GetPath())+'/revo2_right'),handmat)
        cache=UsdGeom.XformCache()
        for name in ('right_hand_base_link','revo2_mount'):
            np.testing.assert_allclose(np.asarray(cache.GetLocalToWorldTransform(find(name))).T,bt,atol=1e-8,rtol=0)
        transforms=[np.asarray(cache.GetLocalToWorldTransform(find(f'link{j+1}'))).T for j in range(6)]
        origins=np.array([t[:3,3] for t in transforms]);axes=np.array([t[:3,:3]@a for t,a in zip(transforms,kin.joint_axes)])
        separation=float(np.rad2deg(np.arccos(np.clip(axes[3]@axes[5],-1,1))))
        np.testing.assert_allclose(separation,np.rad2deg(qi[4]),atol=1e-7)
        arrow(prefix+'/Axes/W1',origins[3]-.025*axes[3],origins[3]+.25*axes[3],blue)
        arrow(prefix+'/Axes/W3',origins[5]-.01*axes[5],origins[5]+.19*axes[5],red)
        # Radial ticks attached to link-local X rotate with each joint; unlike
        # the rotation axes, these reveal the large opposing wrist rotations.
        for j,name,m,length in [(3,'W1',blue,.063),(5,'W3',red,.047)]:
            a=xy(origins[j]);b=xy(origins[j]+transforms[j][:3,0]*length)
            line(prefix+'/Radial/'+name,[plane(*a,.008),plane(*b,.008)],m,.004)
            dot(prefix+'/Radial/'+name+'Tip',*b,.005,m,.016)
            dot(prefix+'/Radial/'+name+'Rim',*a,.006,white,.010)
            dot(prefix+'/Radial/'+name+'Center',*a,.0039,m,.015)
        a=xy(p+shift);dot(prefix+'/Wrist/Rim',*a,.006,white,.015);dot(prefix+'/Wrist/Point',*a,.004,green,.02)
        title=['A  BEFORE','B  LARGEST IK JUMP','C  NEXT STEP'][i]
        text(prefix+'/Title',title,shift_x-.44,.393,.026)
        text(prefix+'/Time',f'Log t = {row["time_s"]:.6f} s  /  step {row["episode_step"]}',shift_x-.44,.358,.014)
        text(prefix+'/CommandTime',f'IK command t = {row["controller"]["command_time_s"]:.6f} s',shift_x-.44,.332,.011,muted)
        text(prefix+'/W1',f'wrist1 = {np.rad2deg(qi[3]):+.2f} deg',shift_x-.44,-.27,.020,blue)
        text(prefix+'/W3',f'wrist3 = {np.rad2deg(qi[5]):+.2f} deg',shift_x-.44,-.305,.020,red)
        text(prefix+'/Angle',f'Axis angle / wrist2 = {separation:.3f} deg',shift_x-.44,-.341,.015)
        text(prefix+'/XYZ','Wrist target XYZ [m]',shift_x-.44,-.375,.012,muted)
        text(prefix+'/XYZValue',f'[{targets[i,0]:.6f}, {targets[i,1]:.6f}, {targets[i,2]:.6f}]',shift_x-.44,-.402,.014)
        if i:
            text(prefix+'/Delta',f'{chr(64+i)} -> {chr(65+i)}: wrist {np.linalg.norm(dp[i-1])*1000:.3f} mm / {np.rad2deg(dr[i-1]):.3f} deg',shift_x-.44,-.446,.014)
            text(prefix+'/JointDelta',f'wrist1 {np.rad2deg(dq[i-1,3]):+.2f} deg / wrist3 {np.rad2deg(dq[i-1,5]):+.2f} deg',shift_x-.44,-.476,.014)
        else:
            text(prefix+'/Delta','Same camera and scale in all three views.',shift_x-.44,-.446,.013,muted)
            text(prefix+'/JointDelta','Radial ticks show link-local rotation.',shift_x-.44,-.476,.013,muted)
        captures.append(dict(label=title,record_state_time_s=row['time_s'],command_time_s=row['controller']['command_time_s'],
            episode_step=row['episode_step'],physics_step=row['physics_step'],command_frame=row['command_frame'],
            q_ik=qi.tolist(),q_cmd=row['q_cmd'],q_actual=np.asarray(row['state']['all_q'])[meta['arm_ids']].tolist(),
            hand_target=row['hand_target'],ik_target_pos=targets[i].tolist(),ik_target_quat_xyzw=quats[i].tolist(),
            joint_axis_angle_deg=separation,source_pose='raw_solve.q, not q_cmd or actual state',
            presentation_shift_world_m=shift.tolist(),fk_usd_mount_verified=True))
        print('[verified]',title,'time',row['time_s'],'axis angle',separation,flush=True)

    text('Global/Title','SMALL WRIST MOTION, LARGE OPPOSING JOINT ROTATIONS',-1.44,.55,.034)
    text('Global/Sub','Recorded raw IK  |  placement 11  |  120 Hz: 8.33 ms between views  |  reconstructed USD geometry, not actual tracking',-1.44,.506,.016)
    for x in (-.5,.5):line('Global/Separator'+str(int(x*10)).replace('-','n'),[plane(x,.41),plane(x,-.49)],light,.001)
    text('Global/Footer','Blue: wrist1 axis/tick     Red: wrist3 axis/tick     Green: Revo2 wrist/base target     No IK recomputation, physics stepping, or asset edits.',-1.44,-.555,.014)
    mesh('Backdrop',[center-normal*.4+right*x+up*y for x,y in [(-5,-5),(5,-5),(5,5),(-5,5)]],[4],[0,1,2,3],white)
    dome=UsdLux.DomeLight.Define(stage,'/Photo/Dome');dome.CreateIntensityAttr(900.)
    key=UsdLux.DistantLight.Define(stage,'/Photo/Key');key.CreateIntensityAttr(1800.)
    key.AddRotateXYZOp().Set(Gf.Vec3f(25.,-35.,-30.))
    cam=UsdGeom.Camera.Define(stage,'/Photo/Camera');cam.CreateProjectionAttr('orthographic')
    cam.CreateHorizontalApertureAttr(30.);cam.CreateVerticalApertureAttr(12.5)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01,20.))
    pose(cam.GetPrim(),np.eye(3),np.zeros(3))
    camx=UsdGeom.Xformable(cam.GetPrim());camx.ClearXformOpOrder();camop=camx.AddTransformOp()
    def camera_at(c):camop.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(c+normal*3.)),Gf.Vec3d(*c),Gf.Vec3d(*up)).GetInverse())
    camera_at(center)
    product=rep.create.render_product(str(cam.GetPath()),(4800,2000))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);rep.orchestrator.set_capture_on_play(False)
    for _ in range(60):app.update()
    def capture(path,shape):
        if path.exists():
            print('[preserve existing capture]',path,flush=True)
            return
        rep.orchestrator.step(rt_subframes=32,delta_time=0.,pause_timeline=True)
        pixels=np.asarray(rgb.get_data())[:,:,:3].astype(np.uint8)
        if pixels.shape!=shape or pixels.std()<5:raise ValueError('Invalid capture')
        imageio.imwrite(path,pixels);print('[photo complete]',path,flush=True)
    capture(IMAGE,(2000,4800,3))
    rgb.detach([product]);product.destroy()
    UsdGeom.Imageable(stage.GetPrimAtPath('/Photo/Global')).MakeInvisible()
    cam.CreateHorizontalApertureAttr(10.);cam.CreateVerticalApertureAttr(10.)
    product=rep.create.render_product(str(cam.GetPath()),(2000,2000));rgb.attach([product])
    for i in range(3):
        camera_at(center+right*(i-1)+up*(-.035))
        # Other panels lie outside the camera frustum. Keep all robot references
        # visible: toggling instance/prototype visibility can hide shared meshes.
        for _ in range(8):app.update()
        capture(OUT/f'{i+1:02d}_{["before","peak","after"][i]}.png',(2000,2000,3))
    metadata=dict(source=str(SOURCE.relative_to(ROOT)),source_metadata_sha256=hashlib.sha256((SOURCE/'metadata.json').read_bytes()).hexdigest(),
        episode=11,physics_dt=meta['physics_dt'],frames=captures,
        between_frame_wrist_target_xyz_delta_mm=(dp*1000).tolist(),
        between_frame_wrist_target_distance_mm=(np.linalg.norm(dp,axis=1)*1000).tolist(),
        between_frame_wrist_target_angle_deg=np.rad2deg(dr).tolist(),between_frame_joint_delta_deg=np.rad2deg(dq).tolist(),
        no_new_ik=True,physics_stepped=False,original_assets_saved=False,
        view_normal=normal.tolist(),view_up=up.tolist(),same_camera_and_scale=True,
        joint_order=kin.joint_names,quaternion_order='xyzw')
    (OUT/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
except Exception:
    import traceback
    traceback.print_exc()
    exit_code=1
finally:
    app.close(exit_code=exit_code)
