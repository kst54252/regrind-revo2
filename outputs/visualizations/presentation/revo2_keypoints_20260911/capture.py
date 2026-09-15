"""One still: actual Revo2 USD plus camera-projected, FK-checked keypoints.

Render-only presentation, not a physics test. Original USDs remain untouched.
Number labels are kp_XX identifiers, NOT MANO indices. Zero leader angles;
existing FK supplies all mimic motion. Occlusion-free annotations are projected
onto a camera-parallel plane without changing their image-space coordinates.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').is_file())
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
IMAGE = OUT / 'revo2_semantic_keypoints_labeled.png'
if IMAGE.exists():
    raise FileExistsError(IMAGE)

import numpy as np
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics

kin = Revo2Kinematics()
q = np.zeros(6)
points = kin.get_keypoints(q)
assert points.shape == (21, 3) and np.isfinite(points).all()
lower, upper = kin.get_joint_limits()
assert np.all(q >= lower) and np.all(q <= upper)

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 2400, 'height': 2400,
                     'renderer': 'RaytracedLighting', 'anti_aliasing': 3})
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
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    hand = UsdGeom.Xform.Define(stage, '/Photo/Hand')
    asset = ROOT / 'USD/revo2_right/revo2_right.usda'
    hand.GetPrim().GetReferences().AddReference(str(asset))

    def material(name, color, flat=False):
        mat = UsdShade.Material.Define(stage, '/Photo/Materials/' + name)
        s = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Surface')
        s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*((0., 0., 0.) if flat else color)))
        s.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.65)
        if flat:
            s.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        mat.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), 'surface')
        return mat

    def bind(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    bind(hand.GetPrim(), material('Hand', (.25, .30, .36)))
    for name, (r, p) in kin._forward_links(q).items():
        candidates = sorted([prim for prim in Usd.PrimRange(hand.GetPrim()) if prim.GetName() == name],
                            key=lambda prim: len(str(prim.GetPath())))
        if not candidates:
            raise ValueError('Missing model link: ' + name)
        x = UsdGeom.Xformable(candidates[0])
        x.ClearXformOpOrder(); x.SetResetXformStack(True)
        t = np.eye(4); t[:3, :3] = r; t[:3, 3] = p
        x.AddTransformOp().Set(Gf.Matrix4d(t.T.tolist()))

    # Verify original USD kp_* transform origins, not the presentation overlay.
    cache = UsdGeom.XformCache()
    errors = []
    for name, p in zip(kin.keypoint_names, points):
        matches = sorted([prim for prim in Usd.PrimRange(hand.GetPrim()) if prim.GetName() == name],
                         key=lambda prim: len(str(prim.GetPath())))
        if not matches:
            raise ValueError('Missing USD keypoint: ' + name)
        actual = np.array(cache.GetLocalToWorldTransform(matches[0]).ExtractTranslation())
        errors.append(float(np.linalg.norm(actual-p)))
        np.testing.assert_allclose(actual, p, atol=1e-5, rtol=0)
        UsdGeom.Imageable(matches[0]).MakeInvisible()
    print('[photo] original USD/FK: 21/21, maximum error m =', max(errors), flush=True)

    center = np.array([0., .034, .075])
    normal = np.array([1., -.035, .08]); normal /= np.linalg.norm(normal)
    right = np.cross([0., 0., 1.], normal); right /= np.linalg.norm(right)
    up = np.cross(normal, right)
    overlay = center + normal * .20

    def on_plane(x, y, depth=0.):
        return overlay + right*x + up*y + normal*depth

    def project(p):
        rel = p-center
        return np.array([np.dot(rel, right), np.dot(rel, up)])

    def mesh(name, vertices, counts, indices, mat):
        prim = UsdGeom.Mesh.Define(stage, '/Photo/' + name)
        prim.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        prim.CreateFaceVertexCountsAttr(counts)
        prim.CreateFaceVertexIndicesAttr(indices)
        prim.CreateSubdivisionSchemeAttr('none')
        prim.CreateDoubleSidedAttr(True)
        bind(prim.GetPrim(), mat)
        return prim

    dark = material('Text', (.025, .04, .065), True)
    muted = material('Muted', (.14, .19, .25), True)
    white = material('White', (1., 1., 1.), True)
    font = FontProperties(family='DejaVu Sans', weight='bold')

    def text(name, value, x, y, size, mat=dark):
        path = TextPath((0, 0), value, size=size, prop=font)
        geometry = Polygon()
        for poly in path.to_polygons():
            if len(poly) >= 3:
                geometry = geometry.symmetric_difference(Polygon(poly))
        tris = list(constrained_delaunay_triangles(geometry).geoms)
        vertices = [on_plane(x+a, y+b, .010) for tri in tris for a, b in list(tri.exterior.coords)[:3]]
        if vertices:
            mesh('Text/'+name, vertices, [3]*len(tris), list(range(len(vertices))), mat)

    def disc(name, x, y, radius, mat, depth):
        vertices = [on_plane(x,y,depth)] + [on_plane(x+radius*np.cos(a),y+radius*np.sin(a),depth)
                                              for a in np.linspace(0,2*np.pi,48,endpoint=False)]
        indices = [v for i in range(48) for v in (0,i+1,(i+1)%48+1)]
        mesh(name, vertices, [3]*48, indices, mat)

    # White backing plane and studio lights. No floor/contact/physics simulation.
    mesh('Backdrop', [center-normal*.25+right*x+up*y for x,y in [(-2,-2),(2,-2),(2,2),(-2,2)]],
         [4], [0,1,2,3], white)
    dome = UsdLux.DomeLight.Define(stage, '/Photo/Dome'); dome.CreateIntensityAttr(850.)
    key = UsdLux.DistantLight.Define(stage, '/Photo/Key'); key.CreateIntensityAttr(1500.)
    key.AddRotateXYZOp().Set(Gf.Vec3f(30., -60., -30.))

    groups = [('Thumb', (0,13,14,15,16), (.80,.06,.08)),
              ('Index', (0,1,2,3,17), (.015,.30,.85)),
              ('Middle', (0,4,5,6,18), (.01,.46,.23)),
              ('Ring', (0,10,11,12,19), (.94,.30,.015)),
              ('Little', (0,7,8,9,20), (.49,.11,.72))]
    mats = {}; colors = {0: dark}
    projected = np.array([project(p) for p in points])
    for name, chain, color in groups:
        mat = mats[name] = material(name, color, True)
        colors.update({i: mat for i in chain[1:]})
        line = UsdGeom.BasisCurves.Define(stage, '/Photo/Lines/'+name)
        line.CreateTypeAttr('linear'); line.CreateCurveVertexCountsAttr([len(chain)])
        line.CreatePointsAttr([Gf.Vec3f(*on_plane(*projected[i])) for i in chain])
        line.CreateWidthsAttr([.00045]); line.SetWidthsInterpolation('constant')
        bind(line.GetPrim(), mat)
    for i, (x,y) in enumerate(projected):
        disc(f'Points/p{i:02}_rim', x, y, .00205, white, .002)
        disc(f'Points/p{i:02}', x, y, .00150, colors[i], .003)
        dx,dy = (.0033,.0017) if i not in (13,14,15,16) else (-.002,.005)
        text(f'Label{i:02}', f'{i:02}', x+dx, y+dy, .0031, colors[i])

    text('Title', 'REVO2', -.104, .103, .008)
    text('Subtitle', '21 SEMANTIC KEYPOINTS', -.104, .094, .0034, muted)
    text('Note', 'Right hand  /  Open pose  /  6 actuated joints', -.104, -.092, .0026, muted)
    for n, (name, chain, color) in enumerate(groups):
        x = -.099+n*.041
        disc('Legend/'+name, x, -.102, .0015, mats[name], .004)
        text('Legend'+name, name, x+.004, -.1032, .003, mats[name])
    text('Footer', 'Labels: kp_00 - kp_20   |   Projected FK overlay on the original USD model',
         -.104, -.113, .00215, muted)

    camera = UsdGeom.Camera.Define(stage, '/Photo/Camera')
    camera.CreateProjectionAttr('orthographic')
    camera.CreateHorizontalApertureAttr(2.4); camera.CreateVerticalApertureAttr(2.4)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
    eye = center+normal*.8
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(*up)).GetInverse())
    product = rep.create.render_product(str(camera.GetPath()), (2400,2400))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
    rep.orchestrator.set_capture_on_play(False)
    for _ in range(60): app.update()
    rep.orchestrator.step(rt_subframes=32,delta_time=0.,pause_timeline=True)
    pixels = np.asarray(rgb.get_data())[:,:,:3].astype(np.uint8)
    if pixels.shape != (2400,2400,3) or pixels.std() < 5:
        raise ValueError('Invalid RGB render')
    imageio.imwrite(IMAGE,pixels)
    metadata = dict(asset=str(asset.relative_to(ROOT)), asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        keypoints_json=str(kin.keypoints_path.relative_to(ROOT)), keypoint_names=kin.keypoint_names,
        keypoints_xyz_m=points.tolist(), joint_names=kin.joint_names, q_rad=q.tolist(),
        usd_fk_errors_m=errors, all_21_verified=True, image_size=[2400,2400],
        camera_eye=eye.tolist(), camera_target=center.tolist(),
        annotation='Orthographic FK projection, occlusion-free; labels are kp IDs, not MANO indices.',
        physics_stepped=False, input_assets_saved=False)
    (OUT/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('[photo complete]',IMAGE,flush=True)
finally:
    app.close()
