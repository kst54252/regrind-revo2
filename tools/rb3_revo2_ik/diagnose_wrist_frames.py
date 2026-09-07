"""Snapshot wrist frames without stepping physics or running a controller/policy.

Default: solve reference wrist targets with existing strict IK and teleport both
assets. --telemetry: reconstruct recorded actual arm joints and compare to the
target from that SAME recorded sample (not the next reference command frame).
All reported matrices use column vectors; quaternions are explicitly XYZW.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", required=True)
parser.add_argument("--telemetry", help="play-arm --arm-tracking-path NPZ; env 0 only")
parser.add_argument("--output", required=True, help="New JSONL file; existing files are refused")
parser.add_argument("--offset-xy", type=float, nargs=2, default=(0.0, 0.0),
                    help="Rigid reference translation; forbidden with recorded telemetry")
parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
parser.add_argument("--position-tolerance", type=float, default=1e-5)
parser.add_argument("--orientation-tolerance", type=float, default=1e-4)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
output = Path(args.output).expanduser().resolve()
if output.exists():
    parser.error(f"output already exists: {output}")
if args.telemetry and any(args.offset_xy):
    parser.error("telemetry already contains placed targets; do not apply another offset")
launcher = AppLauncher(args)

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation
import omni.physx
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import regrind.tasks  # noqa: F401
from regrind.robots.free_revo2_right_hand import FREE_REVO2_RIGHT_HAND_CFG, FREE_REVO2_USD_PATH


def pose_matrix(pos, quat_xyzw):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    matrix[:3, 3] = pos
    return matrix


def usd_matrix(stage, cache, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"missing prim: {path}")
    # Gf uses row vectors: transpose ONCE before composing NumPy matrices.
    return np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T


def error(expected, actual):
    radians = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
    return {"position_m": float(np.linalg.norm(expected[:3, 3] - actual[:3, 3])),
            "orientation_rad": radians, "orientation_deg": float(np.rad2deg(radians))}


def rigid_pose(robot, body=None):
    if body is None:
        value = robot.data.root_link_pose_w.torch[0].detach().cpu().numpy()
        return pose_matrix(value[:3], value[3:])
    index = robot.body_names.index(body)
    return pose_matrix(robot.data.body_pos_w.torch[0, index].detach().cpu().numpy(),
                       robot.data.body_quat_w.torch[0, index].detach().cpu().numpy())


def run():
    task = "Regrind-RB3-Revo2-TunaCan-Online-Play-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1, use_fabric=False)
    cfg.seed = 42
    cfg.commands.reference.trajectory_path = str(Path(args.reference).resolve())
    cfg.commands.reference.randomize_object_xy = False
    cfg.commands.reference.debug_output = False
    cfg.actions.root_pose.debug_output = False
    # Separate floating asset used only as a geometric probe; no env.step or
    # policy calls occur. Overlap is intentional and no contact is simulated.
    cfg.scene.floating_probe = FREE_REVO2_RIGHT_HAND_CFG.replace(prim_path="/World/FloatingProbe")
    env = gym.make(task, cfg=cfg)
    try:
        env.reset()
        base = env.unwrapped
        command = base.command_manager.get_term("reference")
        action = base.action_manager.get_term("root_pose")
        arm, floating = command.robot, base.scene["floating_probe"]
        kin = action._kinematics
        stage = base.sim.stage
        model = kin.config
        root = "/World/envs/env_0/Robot"
        remap = lambda path: root + path.removeprefix("/World")
        paths = {
            "floating_root": "/World/FloatingProbe",
            "floating_base": "/World/FloatingProbe/Geometry/world/right_hand_base_link",
            "revo2_container": remap(model["hand_root_prim"]),
            "revo2_geometry_world": remap(model["hand_articulation_prim"]),
            "mounted_base": remap(model["mounted_wrist_frame"]),
            "link6": remap(model["link6_prim"]),
            "revo2_mount": remap(model["mount_frame_prim"]),
            "hand_rb3_mount": remap(model["mounted_wrist_frame"]) + "/rb3_mount",
            "stock_tcp": remap(model["rb3_tcp_prim"]),
        }
        source = Usd.Stage.Open(str(FREE_REVO2_USD_PATH))
        source_root = str(source.GetDefaultPrim().GetPath())
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        t_f_b = np.linalg.inv(usd_matrix(source, cache, source_root)) @ usd_matrix(
            source, cache, source_root + "/Geometry/world/right_hand_base_link")
        t_l_b = pose_matrix(kin.link6_to_wrist_position,
                           Rotation.from_matrix(kin.link6_to_wrist_rotation).as_quat())
        reference = command.reference
        positions = reference.wrist_pos.copy()
        positions[:, :2] += args.offset_xy
        quats = reference.wrist_quat_xyzw
        q_recorded = None
        recorded_base_poses = None
        indices = np.arange(reference.frames)
        if args.telemetry:
            with np.load(args.telemetry, allow_pickle=False) as data:
                positions = data["target_wrist_pos"].copy()
                quats = data["target_wrist_quat_xyzw"].copy()
                q_recorded = data["actual_rb3_joints"].copy()
                recorded_base_poses = np.stack([
                    pose_matrix(p, q) for p, q in zip(
                        data["actual_wrist_pos"], data["actual_wrist_quat_xyzw"]
                    )
                ])
                indices = data["frame_index"].copy()
                telemetry_dt = float(data["dt"])
                if not np.isclose(telemetry_dt, reference.dt):
                    raise ValueError("reference and telemetry dt differ")
        count = len(positions)
        if count == 0:
            raise ValueError("empty target trajectory")
        for name, value, shape in (("targets", positions, (count, 3)),
                                   ("quaternions", quats, (count, 4)),
                                   ("frame_index", indices, (count,))):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}: {value.shape}")
        if q_recorded is not None and (q_recorded.shape != (count, 6) or not np.isfinite(q_recorded).all()):
            raise ValueError("invalid recorded arm joints")
        if args.max_frames > 0:
            count = min(count, args.max_frames)
        header = {"type": "metadata", "mode": "recorded_joint_reconstruction" if args.telemetry else "strict_ik_teleport",
                  "reference": str(reference.path), "telemetry": args.telemetry,
                  "matrix_convention": "column vectors, translation in [:3,3]", "quaternion_order": "xyzw",
                  "dt": reference.dt, "paths": paths, "T_floating_root_base": t_f_b.tolist(),
                  "T_link6_base_model": t_l_b.tolist(), "ik_target_frame": paths["mounted_base"],
                  "notes": "No physics integration or controller/policy execution. Container prims are not rigid-body wrist frames."}
        output.parent.mkdir(parents=True, exist_ok=True)
        failures, tracking_failures, stale_usd, records = [], [], [], []
        warm = arm.data.joint_pos.torch[0, action._joint_ids].cpu().numpy().copy()
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=base.device)
        physx = omni.physx.get_physx_interface()
        with output.open("x", encoding="utf-8") as log:
            log.write(json.dumps(header) + "\n")
            print(json.dumps(header, indent=2))
            for sample in range(count):
                target = pose_matrix(positions[sample], quats[sample])
                expected = target @ t_f_b
                result = kin.inverse(expected[:3, 3], Rotation.from_matrix(expected[:3, :3]).as_quat(),
                                     initial_q=warm, neutral_q=warm)
                if not (result.success and result.finite and not result.joint_limit_violation):
                    raise RuntimeError(f"strict IK failed at sample {sample}: {result}")
                warm = result.q.copy()
                q = warm if q_recorded is None else q_recorded[sample]
                arm.write_joint_state_to_sim(tensor(q[None]), tensor(np.zeros((1, 6))), joint_ids=action._joint_ids)
                floating.write_root_link_pose_to_sim_index(root_pose=tensor(np.r_[positions[sample], quats[sample]][None]))
                floating.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1, 6))))
                # Request publication without integrating another timestep.
                # Backends may still leave USD at the authored pose: compare
                # independently to PhysX, never mistake USD lag for bad IK.
                physx.update_transformations(True, True, False, False)
                base.scene.update(0.0)
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                worlds = {key: usd_matrix(stage, cache, path) for key, path in paths.items()}
                actual = rigid_pose(arm, "right_hand_base_link")
                link6 = rigid_pose(arm, "link6")
                float_root, float_base = rigid_pose(floating), rigid_pose(floating, "right_hand_base_link")
                fk = pose_matrix(*kin.forward(q))
                fixed = {"floating_root_to_base": np.linalg.inv(float_root) @ float_base,
                         "link6_to_base": np.linalg.inv(link6) @ actual,
                         "link6_to_mount_usd": np.linalg.inv(worlds["link6"]) @ worlds["revo2_mount"],
                         "mount_to_base_usd": np.linalg.inv(worlds["revo2_mount"]) @ worlds["mounted_base"],
                         "link6_to_stock_tcp_usd": np.linalg.inv(worlds["link6"]) @ worlds["stock_tcp"]}
                runtime_mount = link6 @ fixed["link6_to_mount_usd"]
                runtime = {"floating_root": float_root, "floating_base": float_base,
                           "mounted_base": actual, "link6": link6,
                           "revo2_mount": runtime_mount,
                           "stock_tcp": link6 @ fixed["link6_to_stock_tcp_usd"],
                           "hand_rb3_mount": actual @ np.linalg.inv(worlds["mounted_base"]) @ worlds["hand_rb3_mount"]}
                checks = {"target_to_actual_base": error(expected, actual),
                          "fk_to_actual_base": error(fk, actual),
                          "floating_target_to_actual_root": error(target, float_root),
                          "floating_fixed_transform": error(t_f_b, fixed["floating_root_to_base"]),
                          "mount_model_to_actual": error(t_l_b, fixed["link6_to_base"]),
                          "usd_to_physx_base": error(actual, worlds["mounted_base"]),
                          "usd_to_physx_link6": error(link6, worlds["link6"]),
                          "usd_to_physx_floating_base": error(float_base, worlds["floating_base"]),
                          "mount_to_base": error(runtime_mount, actual)}
                if recorded_base_poses is not None:
                    checks["recorded_target_to_actual_base"] = error(expected, recorded_base_poses[sample])
                    checks["recorded_actual_to_reconstructed_base"] = error(recorded_base_poses[sample], actual)
                failed = lambda e: e["position_m"] > args.position_tolerance or e["orientation_rad"] > args.orientation_tolerance
                if any(failed(value) for key, value in checks.items()
                       if key != "target_to_actual_base" and not key.startswith(("usd_to_", "recorded_"))):
                    failures.append(sample)
                if any(failed(value) for key, value in checks.items() if key.startswith("usd_to_")):
                    stale_usd.append(sample)
                if failed(checks["target_to_actual_base"]):
                    tracking_failures.append(sample)
                record = {"type": "frame", "sample": sample, "source_frame_index": int(indices[sample]),
                          "sample_time_s": sample * reference.dt, "floating_wrist_target": target.tolist(),
                          "expected_revo2_base": expected.tolist(), "actual_mounted_base": actual.tolist(),
                          "ik_target": expected.tolist(), "link6_target": (expected @ np.linalg.inv(t_l_b)).tolist(),
                          "ik_fk_at_actual_q": fk.tolist(), "actual_rb3_joints": q.tolist(),
                          "world_runtime": {key: value.tolist() for key, value in runtime.items()},
                          "runtime_mount_source": "PhysX link6/base composed with USD local attachment transforms",
                          "world_usd": {key: value.tolist() for key, value in worlds.items()},
                          "fixed_transforms": {key: value.tolist() for key, value in fixed.items()},
                          "errors": checks, "ik_success": result.success}
                if recorded_base_poses is not None:
                    record["recorded_actual_mounted_base"] = recorded_base_poses[sample].tolist()
                if not np.isfinite(np.asarray(list(worlds.values()))).all():
                    raise RuntimeError(f"nonfinite transform at sample {sample}")
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record))
                records.append(checks)
            summary = {"type": "summary", "samples": count, "frame_mismatch_samples": failures,
                       "usd_readback_mismatch_samples": stale_usd,
                       "target_tracking_error_samples": tracking_failures,
                       "max_errors": {key: {metric: max(r[key][metric] for r in records)
                                            for metric in records[0][key]} for key in records[0]}}
            log.write(json.dumps(summary) + "\n")
            print(json.dumps(summary, indent=2))
        if failures or (q_recorded is None and tracking_failures):
            raise RuntimeError(f"frame verification failed; inspect {output}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        run()
    finally:
        launcher.app.close()
