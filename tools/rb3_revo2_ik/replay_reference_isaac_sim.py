"""RB3+Revo2 trajectory replay with kinematic or dynamic-object modes.

Kinematic mode teleports joints and the reference object while physics is
paused.  Dynamic-object mode position-controls the robot, places the can only
at reset, and lets contact, gravity, and friction determine its motion.

After running this file, use the Script Editor console:
    REPLAY.play()
    REPLAY.pause()
    REPLAY.reset()
    REPLAY.seek(25)
    REPLAY.summary()
"""

import asyncio
import json
import os
import sys
import time

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in ("0", "false", "no", "off")


# User configuration ---------------------------------------------------------
PROJECT_ROOT = os.environ.get(
    "REVO2_PROJECT_ROOT", "/home/wanjunkim/ARSL/regrind-revo2"
)
TRAJECTORY_PATH = os.environ.get(
    "REVO2_TRAJECTORY_PATH",
    os.path.join(
        PROJECT_ROOT,
        "outputs",
        "trajectories",
        "dexycb",
        "20200928_144714",
        "rb3_revo2_reference_world_x040_y000_lift_z_yawm120_strict.h5",
    ),
)
OBJECT_MESH_PATH = os.environ.get(
    "REVO2_OBJECT_MESH_PATH",
    os.path.join(PROJECT_ROOT, "007_tuna_fish_can", "textured_simple.obj"),
)
MODEL_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "tools", "rb3_revo2_ik", "rb3_model.json"
)
VALIDATION_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "diagnostics",
    "rb3_kinematic_replay_validation.npz",
)

# None uses the stored FPS. Set the real frame period for interpolated data.
_dt_environment = os.environ.get("REVO2_REPLAY_DT")
DT_OVERRIDE = None if _dt_environment is None else float(_dt_environment)
PLAYBACK_RATE = float(os.environ.get("REVO2_REPLAY_SPEED", "1.0"))
LOOP = _env_bool("REVO2_REPLAY_LOOP", False)
AUTO_PLAY = _env_bool("REVO2_REPLAY_AUTO_PLAY", True)
START_FRAME = int(os.environ.get("REVO2_REPLAY_START_FRAME", "0"))
PRINT_EVERY = 10

JOINT_READBACK_TOLERANCE_RAD = 1.0e-5
WRIST_POSITION_TOLERANCE_M = 1.0e-4
WRIST_ORIENTATION_TOLERANCE_RAD = 1.0e-3
VIEWPORT_SYNC_TOLERANCE_M = 1.0e-5
VIEWPORT_SYNC_TOLERANCE_RAD = 1.0e-5
JOINT_LIMIT_TOLERANCE_RAD = 1.0e-8
DISCONTINUITY_STEP_THRESHOLD_RAD = 0.5
DEBUG_ROOT_PATH = "/World/RB3KinematicReplayDebug"
OBJECT_ROOT_PATH = "/World/RB3KinematicReplayObject"
SHOW_DEMO_SKELETON = _env_bool("REVO2_SHOW_DEMO_SKELETON", True)
PHYSICS_OBJECT = _env_bool("REVO2_PHYSICS_OBJECT", False)
OBJECT_MASS_KG = float(os.environ.get("REVO2_OBJECT_MASS_KG", "0.15"))
OBJECT_FRICTION = float(os.environ.get("REVO2_OBJECT_FRICTION", "0.8"))
if PHYSICS_OBJECT:
    VALIDATION_OUTPUT_PATH = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "diagnostics",
        "rb3_physics_object_replay_validation.npz",
    )
SHOW_OBJECT_REFERENCE = _env_bool(
    "REVO2_SHOW_OBJECT_REFERENCE", not PHYSICS_OBJECT
)
PHYSICS_TABLE_PATH = "/World/RB3PhysicsTable"
PHYSICS_MATERIAL_PATH = "/World/RB3PhysicsMaterial"

# DexYCB/MANO joint order: wrist, then four joints for thumb through pinky.
MANO_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)

REVO2_FOLLOWERS = {
    "right_thumb_distal_joint": ("right_thumb_proximal_joint", 1.0, 0.0),
    "right_index_distal_joint": ("right_index_proximal_joint", 1.155, 0.0),
    "right_middle_distal_joint": ("right_middle_proximal_joint", 1.155, 0.0),
    "right_ring_distal_joint": ("right_ring_proximal_joint", 1.155, 0.0),
    "right_pinky_distal_joint": ("right_pinky_proximal_joint", 1.155, 0.0),
}


TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools", "rb3_revo2_ik")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
from reference_trajectory import (  # noqa: E402
    analyze_continuity,
    joint_limit_violations,
    load_reference_trajectory,
    quaternion_angular_error_xyzw,
)


def _load_model_config():
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _indices(articulation, names):
    missing = [name for name in names if name not in articulation.dof_names]
    if missing:
        raise RuntimeError(
            f"Articulation {articulation.prim_path} is missing DOFs {missing}; "
            f"available={articulation.dof_names}"
        )
    return np.asarray(
        [articulation.get_dof_index(name) for name in names], dtype=np.int32
    )


def _stage_joint_limits(stage, names):
    joints = {
        prim.GetName(): UsdPhysics.RevoluteJoint(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    }
    missing = [name for name in names if name not in joints]
    if missing:
        raise RuntimeError(f"Stage has no revolute joints named {missing}")
    lower = np.deg2rad(
        np.asarray([joints[name].GetLowerLimitAttr().Get() for name in names], dtype=float)
    )
    upper = np.deg2rad(
        np.asarray([joints[name].GetUpperLimitAttr().Get() for name in names], dtype=float)
    )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("Stage joint limits contain NaN/Inf")
    return lower, upper


def _set_sphere_position(sphere, position):
    UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )


def _set_demo_skeleton(skeleton, points):
    if skeleton is None:
        return
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"MANO skeleton frame must have shape (21,3), got {points.shape}")
    curve_points = np.concatenate(
        [points[np.asarray(chain, dtype=int)] for chain in MANO_FINGER_CHAINS], axis=0
    )
    skeleton["curve"].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(curve_points))
    for sphere, position in zip(skeleton["joints"], points):
        _set_sphere_position(sphere, position)


def _create_demo_skeleton(stage, trajectory):
    if not SHOW_DEMO_SKELETON:
        print("[replay] pre-retargeting MANO skeleton disabled")
        return None
    if trajectory.mano_joint_world is None:
        print(
            "[replay] WARNING: no pre-retargeting MANO skeleton was found in "
            "the reference or its source trajectory"
        )
        return None

    root_path = DEBUG_ROOT_PATH + "/DemoManoSkeleton"
    UsdGeom.Xform.Define(stage, root_path)
    curve = UsdGeom.BasisCurves.Define(stage, root_path + "/Bones")
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr(
        Vt.IntArray([len(chain) for chain in MANO_FINGER_CHAINS])
    )
    curve.CreateWidthsAttr(Vt.FloatArray([0.0045]))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.15, 0.85)]))

    joints = []
    for index in range(21):
        sphere = UsdGeom.Sphere.Define(stage, f"{root_path}/Joint_{index:02d}")
        sphere.CreateRadiusAttr(0.006)
        sphere.CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(1.0, 0.15, 0.85)])
        )
        joints.append(sphere)
    skeleton = {"curve": curve, "joints": tuple(joints)}
    _set_demo_skeleton(skeleton, trajectory.mano_joint_world[0])
    print("[replay] pre-retargeting MANO skeleton loaded: 21 joints, 5 chains")
    return skeleton


def _matrix_pose_xyzw(matrix):
    position = np.asarray(matrix.ExtractTranslation(), dtype=float)
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    xyzw = np.asarray(
        (imaginary[0], imaginary[1], imaginary[2], quaternion.GetReal()), dtype=float
    )
    xyzw /= np.linalg.norm(xyzw)
    return position, xyzw


def _create_object_mesh(stage, mesh_path):
    """Create an object visual rooted at a poseable Xform.

    Prefer the pre-converted sibling USD so OBJ textures are preserved. If it
    is absent, load the requested OBJ with trimesh and author its triangles
    directly into the current Stage.
    """
    mesh_path = os.path.abspath(os.path.expanduser(mesh_path))
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"object mesh not found: {mesh_path}")
    if stage.GetPrimAtPath(OBJECT_ROOT_PATH).IsValid():
        stage.RemovePrim(OBJECT_ROOT_PATH)
    root = UsdGeom.Xform.Define(stage, OBJECT_ROOT_PATH)
    converted_usd = os.path.splitext(mesh_path)[0] + ".usd"
    if os.path.isfile(converted_usd):
        root.GetPrim().GetReferences().AddReference(converted_usd)
        # The YCB MTL uses ``Tr 1``. The converted USD interpreted that as
        # opacity 0, making a correctly loaded can completely invisible.
        # Override only this transient replay composition, not the asset file.
        for prim in Usd.PrimRange(root.GetPrim()):
            opacity = prim.GetAttribute("inputs:opacity")
            if opacity.IsValid():
                opacity.Set(1.0)
        source = converted_usd
    else:
        import trimesh

        loaded = trimesh.load(mesh_path, force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = [geometry for geometry in loaded.geometry.values()]
            if not geometries:
                raise ValueError(f"object OBJ has no mesh geometry: {mesh_path}")
            loaded = trimesh.util.concatenate(geometries)
        mesh = UsdGeom.Mesh.Define(stage, OBJECT_ROOT_PATH + "/Geometry")
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.75, 0.62, 0.32)]))
        source = mesh_path
    xform = UsdGeom.Xformable(root.GetPrim())
    ordered_ops = xform.GetOrderedXformOps()
    translate_op = next(
        (op for op in ordered_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    orient_op = next(
        (op for op in ordered_ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient),
        None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if orient_op is None:
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    print(f"[replay] object visual loaded: {source}")
    return translate_op, orient_op


def _bind_physics_material(prim, material_path):
    relationship = prim.CreateRelationship("material:binding:physics", False)
    relationship.SetTargets([material_path])


def _configure_dynamic_can(stage, mesh_path):
    """Add a robust cylinder collider, mass, gravity, and a Z=0 table."""
    if OBJECT_MASS_KG <= 0.0:
        raise ValueError("REVO2_OBJECT_MASS_KG must be > 0")
    if OBJECT_FRICTION < 0.0:
        raise ValueError("REVO2_OBJECT_FRICTION must be >= 0")

    import trimesh

    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"object mesh contains no geometry: {mesh_path}")
        loaded = trimesh.util.concatenate(geometries)
    bounds = np.asarray(loaded.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"invalid object mesh bounds: {bounds}")
    radius = float(np.max(np.abs(bounds[:, :2])))
    height = float(bounds[1, 2] - bounds[0, 2])
    center_z = float(0.5 * (bounds[0, 2] + bounds[1, 2]))
    if radius <= 0.0 or height <= 0.0:
        raise ValueError(f"invalid can collider radius/height: {radius}, {height}")

    root_prim = stage.GetPrimAtPath(OBJECT_ROOT_PATH)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(root_prim)
    mass.CreateMassAttr(OBJECT_MASS_KG)

    collider = UsdGeom.Cylinder.Define(stage, OBJECT_ROOT_PATH + "/PhysicsCollision")
    collider.CreateAxisAttr(UsdGeom.Tokens.z)
    collider.CreateRadiusAttr(radius)
    collider.CreateHeightAttr(height)
    UsdGeom.XformCommonAPI(collider.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, center_z))
    collider.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)

    material_prim = stage.DefinePrim(PHYSICS_MATERIAL_PATH, "Material")
    material = UsdPhysics.MaterialAPI.Apply(material_prim)
    material.CreateStaticFrictionAttr(OBJECT_FRICTION)
    material.CreateDynamicFrictionAttr(OBJECT_FRICTION)
    material.CreateRestitutionAttr(0.02)
    _bind_physics_material(collider.GetPrim(), material_prim.GetPath())

    if stage.GetPrimAtPath(PHYSICS_TABLE_PATH).IsValid():
        stage.RemovePrim(PHYSICS_TABLE_PATH)
    table = UsdGeom.Cube.Define(stage, PHYSICS_TABLE_PATH)
    table.CreateSizeAttr(1.0)
    table.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.18, 0.18, 0.20)]))
    table_xform = UsdGeom.XformCommonAPI(table.GetPrim())
    table_xform.SetScale(Gf.Vec3f(2.0, 2.0, 0.02))
    table_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.01))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim()).CreateCollisionEnabledAttr(True)
    _bind_physics_material(table.GetPrim(), material_prim.GetPath())

    scenes = [
        UsdPhysics.Scene(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    ]
    scene = scenes[0] if scenes else UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    print(
        "[physics] dynamic tuna can enabled: "
        f"mass={OBJECT_MASS_KG:g} kg friction={OBJECT_FRICTION:g} "
        f"cylinder(radius={radius:.5f}, height={height:.5f}) m"
    )
    print("[physics] static collision table top: world Z=0 m; gravity: -Z, 9.81 m/s^2")


def _set_object_pose(object_xform_ops, position, quaternion_xyzw):
    if object_xform_ops is None:
        return
    translate_op, orient_op = object_xform_ops
    position = np.asarray(position, dtype=float)
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=float)
    quaternion_xyzw /= np.linalg.norm(quaternion_xyzw)
    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        translate_value = Gf.Vec3f(*position.tolist())
    else:
        translate_value = Gf.Vec3d(*position.tolist())
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_value = Gf.Quatf(
            float(quaternion_xyzw[3]), Gf.Vec3f(*quaternion_xyzw[:3].tolist())
        )
    else:
        orient_value = Gf.Quatd(
            float(quaternion_xyzw[3]), Gf.Vec3d(*quaternion_xyzw[:3].tolist())
        )
    translate_op.Set(translate_value)
    orient_op.Set(orient_value)


def _create_debug_visuals(stage, trajectory):
    if stage.GetPrimAtPath(DEBUG_ROOT_PATH).IsValid():
        stage.RemovePrim(DEBUG_ROOT_PATH)
    UsdGeom.Xform.Define(stage, DEBUG_ROOT_PATH)
    if trajectory.wrist_pos is not None:
        curve = UsdGeom.BasisCurves.Define(stage, DEBUG_ROOT_PATH + "/WristReferencePath")
        curve.CreateTypeAttr("linear")
        curve.CreateWrapAttr("nonperiodic")
        curve.CreateCurveVertexCountsAttr(Vt.IntArray([trajectory.frames]))
        curve.CreatePointsAttr(
            Vt.Vec3fArray.FromNumpy(np.asarray(trajectory.wrist_pos, dtype=np.float32))
        )
        curve.CreateWidthsAttr(Vt.FloatArray([0.004]))
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.75, 1.0)]))

    actual_marker = UsdGeom.Sphere.Define(stage, DEBUG_ROOT_PATH + "/ActualWrist")
    actual_marker.CreateRadiusAttr(0.012)
    actual_marker.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.1, 0.1)]))
    reference_marker = UsdGeom.Sphere.Define(stage, DEBUG_ROOT_PATH + "/ReferenceWrist")
    reference_marker.CreateRadiusAttr(0.009)
    reference_marker.CreateDisplayColorAttr(
        Vt.Vec3fArray([Gf.Vec3f(0.1, 1.0, 0.2)])
    )
    object_marker = None
    if trajectory.object_pos is not None and SHOW_OBJECT_REFERENCE:
        object_curve = UsdGeom.BasisCurves.Define(
            stage, DEBUG_ROOT_PATH + "/ObjectReferencePath"
        )
        object_curve.CreateTypeAttr("linear")
        object_curve.CreateWrapAttr("nonperiodic")
        object_curve.CreateCurveVertexCountsAttr(Vt.IntArray([trajectory.frames]))
        object_curve.CreatePointsAttr(
            Vt.Vec3fArray.FromNumpy(np.asarray(trajectory.object_pos, dtype=np.float32))
        )
        object_curve.CreateWidthsAttr(Vt.FloatArray([0.003]))
        object_curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        object_curve.CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(1.0, 0.7, 0.0)])
        )
        object_marker = UsdGeom.Sphere.Define(
            stage, DEBUG_ROOT_PATH + "/ObjectReference"
        )
        object_marker.CreateRadiusAttr(0.012)
        object_marker.CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(1.0, 0.7, 0.0)])
        )
        _set_sphere_position(object_marker, trajectory.object_pos[0])
    demo_skeleton = _create_demo_skeleton(stage, trajectory)
    return actual_marker, reference_marker, object_marker, demo_skeleton


class ReplayBackend:
    """Boundary between the kinematic and future physics replay paths."""

    def apply(self, rb3_q, revo2_q):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def teleport(self, rb3_q, revo2_q):
        self.apply(rb3_q, revo2_q)


class KinematicTeleportBackend(ReplayBackend):
    """Direct joint-state teleport backend."""

    def __init__(self, arm, hand, rb3_names, revo2_names):
        self.arm = arm
        self.hand = hand
        self.rb3_names = tuple(rb3_names)
        self.revo2_names = tuple(revo2_names)
        self.arm_indices = _indices(arm, self.rb3_names)
        self.hand_names = self.revo2_names + tuple(REVO2_FOLLOWERS)
        self.hand_indices = _indices(hand, self.hand_names)

    def _expanded_hand(self, leader_q):
        values = dict(zip(self.revo2_names, np.asarray(leader_q, dtype=float)))
        for follower, (leader, multiplier, offset) in REVO2_FOLLOWERS.items():
            if leader not in values:
                raise RuntimeError(
                    f"Mimic leader {leader!r} is absent from trajectory joint order"
                )
            values[follower] = offset + multiplier * values[leader]
        return np.asarray([values[name] for name in self.hand_names], dtype=float)

    def apply(self, rb3_q, revo2_q):
        self.arm.set_joint_positions(
            np.asarray(rb3_q, dtype=float), joint_indices=self.arm_indices
        )
        self.hand.set_joint_positions(
            self._expanded_hand(revo2_q), joint_indices=self.hand_indices
        )

    def read(self):
        rb3 = np.asarray(
            self.arm.get_joint_positions(joint_indices=self.arm_indices), dtype=float
        )
        expanded_hand = np.asarray(
            self.hand.get_joint_positions(joint_indices=self.hand_indices), dtype=float
        )
        return rb3, expanded_hand[:6]


class PhysicsPositionControllerBackend(ReplayBackend):
    """Drive the same joints toward the trajectory while physics advances."""

    def __init__(self, arm, hand, rb3_names, revo2_names):
        self.teleport_backend = KinematicTeleportBackend(
            arm, hand, rb3_names, revo2_names
        )
        self.arm = arm
        self.hand = hand
        self.arm_indices = self.teleport_backend.arm_indices
        self.hand_indices = self.teleport_backend.hand_indices

    def apply(self, rb3_q, revo2_q):
        self.arm.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(rb3_q, dtype=float),
                joint_indices=self.arm_indices,
            )
        )
        self.hand.apply_action(
            ArticulationAction(
                joint_positions=self.teleport_backend._expanded_hand(revo2_q),
                joint_indices=self.hand_indices,
            )
        )

    def teleport(self, rb3_q, revo2_q):
        self.teleport_backend.apply(rb3_q, revo2_q)
        self.apply(rb3_q, revo2_q)

    def read(self):
        return self.teleport_backend.read()


class KinematicReplayController:
    def __init__(self):
        self.trajectory = load_reference_trajectory(
            TRAJECTORY_PATH, dt_override=DT_OVERRIDE
        )
        self.continuity = analyze_continuity(
            self.trajectory, DISCONTINUITY_STEP_THRESHOLD_RAD
        )
        self.model = _load_model_config()
        self.stage = None
        self.timeline = None
        self.physx_interface = None
        self.backend = None
        self.wrist_prim = None
        self.wrist_body = None
        self.actual_marker = None
        self.reference_marker = None
        self.object_marker = None
        self.demo_skeleton = None
        self.object_xform_ops = None
        self.object_body = None
        self.initialized = False
        self.playing = False
        self.current_frame = int(np.clip(START_FRAME, 0, self.trajectory.frames - 1))
        self._play_task = None

        frames = self.trajectory.frames
        self.rb3_readback_error = np.full(frames, np.nan)
        self.revo2_readback_error = np.full(frames, np.nan)
        self.rb3_actual = np.full((frames, 6), np.nan)
        self.revo2_actual = np.full((frames, 6), np.nan)
        self.stage_wrist_pos = np.full((frames, 3), np.nan)
        self.stage_wrist_quat_xyzw = np.full((frames, 4), np.nan)
        self.viewport_wrist_pos = np.full((frames, 3), np.nan)
        self.viewport_wrist_quat_xyzw = np.full((frames, 4), np.nan)
        self.viewport_sync_position_error = np.full(frames, np.nan)
        self.viewport_sync_orientation_error = np.full(frames, np.nan)
        self.wrist_position_error = np.full(frames, np.nan)
        self.wrist_orientation_error = np.full(frames, np.nan)
        self.object_actual_pos = np.full((frames, 3), np.nan)
        self.object_actual_quat_xyzw = np.full((frames, 4), np.nan)
        self.object_reference_position_error = np.full(frames, np.nan)
        self.stage_finite = np.zeros(frames, dtype=bool)
        self.validated = np.zeros(frames, dtype=bool)
        self.readback_problem_frames = set()
        self.wrist_problem_frames = set()
        self.viewport_sync_problem_frames = set()
        self.limit_violation_frames = np.empty(0, dtype=int)
        self.joint_lower = np.full(12, np.nan)
        self.joint_upper = np.full(12, np.nan)
        self._print_preflight()

    @property
    def dt(self):
        return self.trajectory.dt / PLAYBACK_RATE

    def _print_preflight(self):
        names = self.trajectory.rb3_joint_names + self.trajectory.revo2_joint_names
        print("[12-DoF reference continuity preflight]")
        print(f"  file:                 {self.trajectory.source_path}")
        print(f"  frames:               {self.trajectory.frames}")
        print(f"  stored FPS:           {self.trajectory.fps:.6g}")
        print(f"  analysis dt:          {self.trajectory.dt:.9g} s")
        print(
            f"  max joint-step norm:  {self.continuity.max_joint_step_norm:.9g} rad"
        )
        print(
            f"  discontinuity frames: {self.continuity.discontinuity_frames.tolist()} "
            f"(threshold={DISCONTINUITY_STEP_THRESHOLD_RAD} rad)"
        )
        print(f"  non-finite frames:    {self.continuity.nonfinite_frames.tolist()}")
        print("  per-joint max |velocity| [rad/s] / |acceleration| [rad/s^2]:")
        for index, name in enumerate(names):
            print(
                f"    {name:<36} "
                f"{self.continuity.max_abs_velocity_per_joint[index]: .6f} / "
                f"{self.continuity.max_abs_acceleration_per_joint[index]: .6f}"
            )

    def _reset_dynamic_object(self):
        if self.object_body is None:
            return
        position = np.asarray(self.trajectory.object_pos[0], dtype=float)
        quaternion_xyzw = np.asarray(
            self.trajectory.object_quat_xyzw[0], dtype=float
        )
        quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
        self.object_body.set_world_pose(position, quaternion_wxyz)
        self.object_body.set_linear_velocity(np.zeros(3, dtype=float))
        self.object_body.set_angular_velocity(np.zeros(3, dtype=float))
        print(f"[physics] can reset once at {position.tolist()}; trajectory pose driving is OFF")

    async def initialize(self):
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("No USD Stage is currently open")
        arm_path = self.model["arm_articulation_prim"]
        hand_path = self.model["hand_articulation_prim"]
        wrist_path = self.model["mounted_wrist_frame"]
        for description, path in (
            ("RB3 articulation", arm_path),
            ("mounted wrist", wrist_path),
        ):
            if not self.stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"{description} prim not found in current Stage: {path}")

        # Create physics tensor views once, then pause. Replay itself never
        # advances physics time and only teleports joint positions.
        self.timeline = omni.timeline.get_timeline_interface()
        self.physx_interface = omni.physx.get_physx_interface()
        self.timeline.play()
        await omni.kit.app.get_app().next_update_async()
        arm = SingleArticulation(prim_path=arm_path, name="rb3_kinematic_replay")
        arm.initialize()
        if set(self.trajectory.revo2_joint_names).issubset(set(arm.dof_names)):
            hand = arm
            print("[replay] assembled RB3+Revo2 uses one articulation handle")
        else:
            if not self.stage.GetPrimAtPath(hand_path).IsValid():
                raise RuntimeError(f"Revo2 articulation prim not found: {hand_path}")
            hand = SingleArticulation(
                prim_path=hand_path, name="revo2_kinematic_replay"
            )
            hand.initialize()
            print("[replay] using separate RB3 and Revo2 articulation handles")
        self.wrist_body = SingleRigidPrim(
            prim_path=wrist_path,
            name="mounted_wrist_pose_reader",
            reset_xform_properties=False,
        )
        self.wrist_body.initialize()
        self.timeline.pause()

        backend_type = (
            PhysicsPositionControllerBackend
            if PHYSICS_OBJECT
            else KinematicTeleportBackend
        )
        self.backend = backend_type(
            arm,
            hand,
            self.trajectory.rb3_joint_names,
            self.trajectory.revo2_joint_names,
        )
        rb3_lower, rb3_upper = _stage_joint_limits(
            self.stage, self.trajectory.rb3_joint_names
        )
        revo_lower, revo_upper = _stage_joint_limits(
            self.stage, self.trajectory.revo2_joint_names
        )
        all_lower = np.concatenate((rb3_lower, revo_lower))
        all_upper = np.concatenate((rb3_upper, revo_upper))
        self.joint_lower = all_lower
        self.joint_upper = all_upper
        _, self.limit_violation_frames = joint_limit_violations(
            self.trajectory.reference_joints,
            all_lower,
            all_upper,
            JOINT_LIMIT_TOLERANCE_RAD,
        )
        self.wrist_prim = self.stage.GetPrimAtPath(wrist_path)
        (
            self.actual_marker,
            self.reference_marker,
            self.object_marker,
            self.demo_skeleton,
        ) = _create_debug_visuals(self.stage, self.trajectory)
        if self.trajectory.object_pos is not None:
            self.object_xform_ops = _create_object_mesh(
                self.stage, OBJECT_MESH_PATH
            )
            _set_object_pose(
                self.object_xform_ops,
                self.trajectory.object_pos[0],
                self.trajectory.object_quat_xyzw[0],
            )
            if PHYSICS_OBJECT:
                _configure_dynamic_can(self.stage, OBJECT_MESH_PATH)
                self.timeline.play()
                await omni.kit.app.get_app().next_update_async()
                self.object_body = SingleRigidPrim(
                    prim_path=OBJECT_ROOT_PATH,
                    name="dynamic_tuna_can",
                    reset_xform_properties=False,
                )
                self.object_body.initialize()
                self.timeline.pause()
                self.backend.teleport(
                    self.trajectory.rb3_joints[0], self.trajectory.revo2_joints[0]
                )
                self._reset_dynamic_object()
        self.initialized = True
        await self._apply_frame(self.current_frame)
        if PHYSICS_OBJECT:
            print(
                "[replay] initialized in DYNAMIC-OBJECT mode; robot uses position "
                "targets and the can moves only through gravity/contact"
            )
        else:
            print("[replay] initialized in KINEMATIC mode; physics is paused")
        print("[controls] REPLAY.play() | pause() | reset() | seek(frame) | summary()")
        if AUTO_PLAY:
            self.play()

    async def _apply_frame(self, frame):
        if not self.initialized:
            return
        frame = int(frame)
        if not 0 <= frame < self.trajectory.frames:
            raise IndexError(f"frame must be in [0,{self.trajectory.frames - 1}]")
        rb3_ref = self.trajectory.rb3_joints[frame]
        revo_ref = self.trajectory.revo2_joints[frame]
        if not np.isfinite(rb3_ref).all() or not np.isfinite(revo_ref).all():
            raise ValueError(f"reference frame {frame} contains NaN/Inf")
        physics_advancing = PHYSICS_OBJECT and self.playing
        if physics_advancing:
            self.backend.apply(rb3_ref, revo_ref)
        else:
            self.timeline.pause()
            self.backend.teleport(rb3_ref, revo_ref)
            # A paused timeline does not publish teleported link poses to
            # USD/Fabric. Synchronize without advancing gravity or contacts.
            try:
                self.physx_interface.update_transformations(True, True, False, False)
            except TypeError:
                self.physx_interface.update_transformations(True, True, False)
        await omni.kit.app.get_app().next_update_async()

        rb3_actual, revo_actual = self.backend.read()
        self.rb3_actual[frame] = rb3_actual
        self.revo2_actual[frame] = revo_actual
        self.rb3_readback_error[frame] = float(np.max(np.abs(rb3_actual - rb3_ref)))
        self.revo2_readback_error[frame] = float(
            np.max(np.abs(revo_actual - revo_ref))
        )
        if (
            self.rb3_readback_error[frame] > JOINT_READBACK_TOLERANCE_RAD
            or self.revo2_readback_error[frame] > JOINT_READBACK_TOLERANCE_RAD
        ):
            self.readback_problem_frames.add(frame)

        # Validate against the PhysX rigid-body pose, then separately confirm
        # that the synchronized USD/Fabric pose seen by the viewport agrees.
        wrist_position, wrist_quaternion_wxyz = self.wrist_body.get_world_pose()
        wrist_position = np.asarray(wrist_position, dtype=float)
        wrist_quaternion_wxyz = np.asarray(wrist_quaternion_wxyz, dtype=float)
        wrist_quaternion = wrist_quaternion_wxyz[[1, 2, 3, 0]]
        wrist_quaternion /= np.linalg.norm(wrist_quaternion)
        self.stage_wrist_pos[frame] = wrist_position
        self.stage_wrist_quat_xyzw[frame] = wrist_quaternion
        viewport_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
            self.wrist_prim
        )
        viewport_position, viewport_quaternion = _matrix_pose_xyzw(viewport_matrix)
        self.viewport_wrist_pos[frame] = viewport_position
        self.viewport_wrist_quat_xyzw[frame] = viewport_quaternion
        self.viewport_sync_position_error[frame] = float(
            np.linalg.norm(viewport_position - wrist_position)
        )
        self.viewport_sync_orientation_error[frame] = (
            quaternion_angular_error_xyzw(viewport_quaternion, wrist_quaternion)
        )
        if (
            self.viewport_sync_position_error[frame] > VIEWPORT_SYNC_TOLERANCE_M
            or self.viewport_sync_orientation_error[frame]
            > VIEWPORT_SYNC_TOLERANCE_RAD
        ):
            self.viewport_sync_problem_frames.add(frame)
        self.stage_finite[frame] = bool(
            np.isfinite(rb3_actual).all()
            and np.isfinite(revo_actual).all()
            and np.isfinite(wrist_position).all()
            and np.isfinite(wrist_quaternion).all()
            and np.isfinite(viewport_position).all()
            and np.isfinite(viewport_quaternion).all()
        )
        _set_sphere_position(self.actual_marker, wrist_position)
        if self.trajectory.wrist_pos is not None:
            reference_position = self.trajectory.wrist_pos[frame]
            self.wrist_position_error[frame] = float(
                np.linalg.norm(wrist_position - reference_position)
            )
            self.wrist_orientation_error[frame] = quaternion_angular_error_xyzw(
                wrist_quaternion, self.trajectory.wrist_quat_xyzw[frame]
            )
            _set_sphere_position(self.reference_marker, reference_position)
            if (
                self.wrist_position_error[frame] > WRIST_POSITION_TOLERANCE_M
                or self.wrist_orientation_error[frame]
                > WRIST_ORIENTATION_TOLERANCE_RAD
            ):
                self.wrist_problem_frames.add(frame)
        if self.object_marker is not None:
            _set_sphere_position(self.object_marker, self.trajectory.object_pos[frame])
        if self.object_xform_ops is not None and not PHYSICS_OBJECT:
            _set_object_pose(
                self.object_xform_ops,
                self.trajectory.object_pos[frame],
                self.trajectory.object_quat_xyzw[frame],
            )
        if self.object_body is not None:
            object_position, object_quaternion_wxyz = self.object_body.get_world_pose()
            object_position = np.asarray(object_position, dtype=float)
            object_quaternion_wxyz = np.asarray(object_quaternion_wxyz, dtype=float)
            self.object_actual_pos[frame] = object_position
            self.object_actual_quat_xyzw[frame] = object_quaternion_wxyz[[1, 2, 3, 0]]
            self.object_reference_position_error[frame] = float(
                np.linalg.norm(object_position - self.trajectory.object_pos[frame])
            )
        if self.demo_skeleton is not None:
            _set_demo_skeleton(
                self.demo_skeleton, self.trajectory.mano_joint_world[frame]
            )
        self.validated[frame] = True
        self.current_frame = frame
        if (
            frame % PRINT_EVERY == 0
            or frame == self.trajectory.frames - 1
            or (
                not PHYSICS_OBJECT
                and (
                    frame in self.readback_problem_frames
                    or frame in self.wrist_problem_frames
                    or frame in self.viewport_sync_problem_frames
                )
            )
        ):
            print(
                f"[frame {frame:04d}] rb3_readback={self.rb3_readback_error[frame]:.3g} "
                f"revo2_readback={self.revo2_readback_error[frame]:.3g} rad "
                f"wrist_pos={self.wrist_position_error[frame]:.3g} m "
                f"wrist_ori={self.wrist_orientation_error[frame]:.3g} rad "
                f"viewport_sync={self.viewport_sync_position_error[frame]:.3g} m"
                + (
                    f" can_z={self.object_actual_pos[frame, 2]:.3g} m"
                    if self.object_body is not None
                    else ""
                )
            )

    async def _run(self):
        try:
            if PHYSICS_OBJECT:
                self.timeline.play()
            while self.playing:
                frame = self.current_frame
                start = time.perf_counter()
                await self._apply_frame(frame)
                next_frame = frame + 1
                if next_frame >= self.trajectory.frames:
                    if LOOP:
                        next_frame = 0
                        if PHYSICS_OBJECT:
                            self.timeline.pause()
                            self.backend.teleport(
                                self.trajectory.rb3_joints[0],
                                self.trajectory.revo2_joints[0],
                            )
                            self._reset_dynamic_object()
                            self.timeline.play()
                    else:
                        self.playing = False
                        self.summary()
                        break
                self.current_frame = next_frame
                remaining = self.dt - (time.perf_counter() - start)
                if remaining > 0:
                    await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            pass
        finally:
            if self.timeline is not None:
                self.timeline.pause()

    def play(self):
        if not self.initialized:
            print("[replay] initialization is not complete yet")
            return
        if self.playing:
            return
        if self.current_frame >= self.trajectory.frames - 1:
            self.current_frame = 0
        self.playing = True
        self._play_task = asyncio.ensure_future(self._run())
        print(f"[replay] play from frame {self.current_frame}")

    def pause(self):
        self.playing = False
        if self.timeline is not None:
            self.timeline.pause()
        print(f"[replay] paused at frame {self.current_frame}")

    def reset(self):
        self.pause()
        asyncio.ensure_future(self._reset())

    async def _reset(self):
        self.current_frame = 0
        self.backend.teleport(
            self.trajectory.rb3_joints[0], self.trajectory.revo2_joints[0]
        )
        self._reset_dynamic_object()
        await self._apply_frame(0)

    def seek(self, frame):
        frame = int(frame)
        if not 0 <= frame < self.trajectory.frames:
            raise IndexError(f"frame must be in [0,{self.trajectory.frames - 1}]")
        self.playing = False
        asyncio.ensure_future(self._apply_frame(frame))

    def summary(self):
        checked = self.validated
        checked_indices = np.flatnonzero(checked)
        finite_problem_frames = np.flatnonzero(checked & ~self.stage_finite)
        issue_frames = set(self.continuity.nonfinite_frames.tolist())
        issue_frames.update(self.continuity.discontinuity_frames.tolist())
        issue_frames.update(self.limit_violation_frames.tolist())
        issue_frames.update(self.readback_problem_frames)
        issue_frames.update(self.wrist_problem_frames)
        issue_frames.update(self.viewport_sync_problem_frames)
        issue_frames.update(finite_problem_frames.tolist())
        mode = "dynamic-object physics" if PHYSICS_OBJECT else "kinematic"
        print(f"\n[{mode} replay validation summary]")
        print(f"  total frames:                  {self.trajectory.frames}")
        print(f"  Stage-validated frames:        {len(checked_indices)}")
        print(f"  joint-limit violation frames: {self.limit_violation_frames.tolist()}")
        print(f"  reference NaN/Inf frames:      {self.continuity.nonfinite_frames.tolist()}")
        print(f"  Stage NaN/Inf frames:          {finite_problem_frames.tolist()}")
        print(
            f"  max joint-step norm:           "
            f"{self.continuity.max_joint_step_norm:.9g} rad"
        )
        print(
            f"  max joint velocity:            "
            f"{np.max(self.continuity.max_abs_velocity_per_joint):.9g} rad/s"
        )
        print(
            f"  max joint acceleration:        "
            f"{np.max(self.continuity.max_abs_acceleration_per_joint):.9g} rad/s^2"
        )
        print(
            f"  discontinuity frames:          "
            f"{self.continuity.discontinuity_frames.tolist()}"
        )
        print(f"  joint readback problems:       {sorted(self.readback_problem_frames)}")
        if len(checked_indices):
            print(
                f"  max RB3/Revo2 readback error:  "
                f"{np.nanmax(self.rb3_readback_error[checked]):.9g} / "
                f"{np.nanmax(self.revo2_readback_error[checked]):.9g} rad"
            )
        if self.trajectory.wrist_pos is not None and len(checked_indices):
            print(
                f"  wrist position mean/max:       "
                f"{np.nanmean(self.wrist_position_error[checked]):.9g} / "
                f"{np.nanmax(self.wrist_position_error[checked]):.9g} m"
            )
            print(
                f"  wrist orientation mean/max:    "
                f"{np.nanmean(self.wrist_orientation_error[checked]):.9g} / "
                f"{np.nanmax(self.wrist_orientation_error[checked]):.9g} rad"
            )
            print(f"  wrist FK problem frames:       {sorted(self.wrist_problem_frames)}")
        object_checked = checked & np.isfinite(self.object_reference_position_error)
        if self.object_body is not None and object_checked.any():
            print(
                f"  physical can final position:    "
                f"{self.object_actual_pos[checked_indices[-1]].tolist()}"
            )
            print(
                f"  can/reference pos mean/max:     "
                f"{np.nanmean(self.object_reference_position_error[object_checked]):.9g} / "
                f"{np.nanmax(self.object_reference_position_error[object_checked]):.9g} m"
            )
        if len(checked_indices):
            print(
                f"  viewport sync pos mean/max:     "
                f"{np.nanmean(self.viewport_sync_position_error[checked]):.9g} / "
                f"{np.nanmax(self.viewport_sync_position_error[checked]):.9g} m"
            )
            print(
                f"  viewport sync ori mean/max:     "
                f"{np.nanmean(self.viewport_sync_orientation_error[checked]):.9g} / "
                f"{np.nanmax(self.viewport_sync_orientation_error[checked]):.9g} rad"
            )
            print(
                f"  viewport sync problems:        "
                f"{sorted(self.viewport_sync_problem_frames)}"
            )
        print(f"  all problem frames:            {sorted(issue_frames)}")
        os.makedirs(os.path.dirname(VALIDATION_OUTPUT_PATH), exist_ok=True)
        np.savez_compressed(
            VALIDATION_OUTPUT_PATH,
            rb3_reference=self.trajectory.rb3_joints,
            revo2_reference=self.trajectory.revo2_joints,
            reference_joints=self.trajectory.reference_joints,
            rb3_actual=self.rb3_actual,
            revo2_actual=self.revo2_actual,
            rb3_readback_error_rad=self.rb3_readback_error,
            revo2_readback_error_rad=self.revo2_readback_error,
            stage_wrist_pos=self.stage_wrist_pos,
            stage_wrist_quat_xyzw=self.stage_wrist_quat_xyzw,
            viewport_wrist_pos=self.viewport_wrist_pos,
            viewport_wrist_quat_xyzw=self.viewport_wrist_quat_xyzw,
            viewport_sync_position_error_m=self.viewport_sync_position_error,
            viewport_sync_orientation_error_rad=self.viewport_sync_orientation_error,
            wrist_position_error_m=self.wrist_position_error,
            wrist_orientation_error_rad=self.wrist_orientation_error,
            object_actual_pos=self.object_actual_pos,
            object_actual_quat_xyzw=self.object_actual_quat_xyzw,
            object_reference_position_error_m=self.object_reference_position_error,
            physics_object=np.asarray(PHYSICS_OBJECT),
            validated=self.validated,
            stage_finite=self.stage_finite,
            joint_lower_rad=self.joint_lower,
            joint_upper_rad=self.joint_upper,
            joint_step=self.continuity.joint_step,
            joint_step_norm=self.continuity.joint_step_norm,
            joint_velocity_rad_s=self.continuity.velocity,
            joint_acceleration_rad_s2=self.continuity.acceleration,
            discontinuity_frames=self.continuity.discontinuity_frames,
            joint_limit_violation_frames=self.limit_violation_frames,
            readback_problem_frames=np.asarray(
                sorted(self.readback_problem_frames), dtype=int
            ),
            wrist_problem_frames=np.asarray(sorted(self.wrist_problem_frames), dtype=int),
            viewport_sync_problem_frames=np.asarray(
                sorted(self.viewport_sync_problem_frames), dtype=int
            ),
            all_problem_frames=np.asarray(sorted(issue_frames), dtype=int),
            dt=self.trajectory.dt,
        )
        print(f"  saved validation arrays:       {VALIDATION_OUTPUT_PATH}")


# Re-running the Script Editor file stops the prior controller/task.
try:
    REPLAY.pause()
except (NameError, AttributeError):
    pass
try:
    if _INITIALIZE_TASK is not None and not _INITIALIZE_TASK.done():
        _INITIALIZE_TASK.cancel()
except NameError:
    pass

REPLAY = KinematicReplayController()
_INITIALIZE_TASK = asyncio.ensure_future(REPLAY.initialize())
