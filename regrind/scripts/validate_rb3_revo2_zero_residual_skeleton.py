"""Replay zero residual actions while drawing the source MANO21 skeleton."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Regrind-RB3-Revo2-Tuna-Play-v0")
parser.add_argument("--reference", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--skeleton_scale", type=float, default=0.65)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import regrind.tasks  # noqa: F401
from regrind.utils.markers import ManoMarker


# Sequential MANO21 -> the semantic convention used by regrind.utils.ManoMarker.
MANO21_SEQUENTIAL_TO_REVO_SEMANTIC = np.asarray(
    (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20),
    dtype=np.int64,
)


def _text(value) -> str:
    value = np.asarray(value).item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_demo_skeleton(path: str, expected_frames: int) -> np.ndarray:
    with h5py.File(path, "r") as reference:
        if "mano_joint_world" in reference:
            points = np.asarray(reference["mano_joint_world"][()], dtype=np.float32)
        elif "mano_joint_world_mano21" in reference:
            points = np.asarray(reference["mano_joint_world_mano21"][()], dtype=np.float32)
        else:
            raise KeyError(
                "reference has no mano_joint_world; rebuild it from world_trajectory.h5"
            )
        order = _text(
            reference["mano_joint_order"][()]
            if "mano_joint_order" in reference
            else "mano21_sequential_thumb_index_middle_ring_little"
        )
    if points.shape != (expected_frames, 21, 3):
        raise ValueError(
            f"MANO skeleton must have shape {(expected_frames, 21, 3)}, got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("MANO skeleton contains NaN/Inf")
    if "sequential" in order:
        points = points[:, MANO21_SEQUENTIAL_TO_REVO_SEMANTIC]
    return points


def _object_reference_errors(command):
    position_error = torch.linalg.vector_norm(
        command.current_object_pos - command.target_object_pos, dim=-1
    ).max()
    quaternion_dot = torch.sum(
        command.current_object_quat * command.target_object_quat, dim=-1
    ).abs().clamp(max=1.0)
    orientation_error = (2.0 * torch.acos(quaternion_dot)).max()
    return float(position_error.item()), float(orientation_error.item())


def main() -> None:
    if args_cli.skeleton_scale <= 0.0:
        raise ValueError("--skeleton_scale must be positive")
    print("[zero-residual+skeleton] parsing configuration", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.commands.reference.trajectory_path = args_cli.reference
    env_cfg.commands.reference.rsi_enabled = False
    env_cfg.commands.reference.loop = True
    env_cfg.commands.reference.reset_object_on_loop = True
    env_cfg.commands.reference.debug_output = False
    env_cfg.terminations.demo_end_reached = None
    env_cfg.terminations.success = None
    env_cfg.terminations.object_deviation = None
    env_cfg.terminations.hand_far_from_object = None
    env_cfg.rewards.early_terminated_penalty = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("reference")
    action_term = base_env.action_manager.get_term("joint_pos")
    skeleton_numpy = _load_demo_skeleton(
        command.reference.path, command.reference.frames
    )
    skeleton = torch.as_tensor(
        skeleton_numpy, dtype=torch.float32, device=base_env.device
    )
    marker = ManoMarker(
        prim_path="/World/Visuals/DemoMANO21",
        device=base_env.device,
        scale=args_cli.skeleton_scale,
    )

    def visualize_skeleton() -> None:
        frame_points = skeleton[command.time_steps]
        world_points = frame_points + base_env.scene.env_origins.unsqueeze(1)
        marker.visualize(world_points)

    print("[zero-residual+skeleton] environment ready")
    print(f"  reference:       {command.reference.path}")
    print(f"  frames/fps:       {command.reference.frames}/{command.reference.fps:g}")
    print(f"  actions:          {env.action_space.shape[-1]} zero residuals")
    print("  blue/green/yellow: source MANO21 joints/fingertips/bones")
    print("  object loop reset: enabled")

    env.reset()
    visualize_skeleton()
    initial_object_error = _object_reference_errors(command)
    print(
        "  initial object error: "
        f"position={initial_object_error[0]:.3e} m, "
        f"orientation={initial_object_error[1]:.3e} rad"
    )
    actions = torch.zeros(env.action_space.shape, device=base_env.device)
    max_formula_error = 0.0
    max_tracking_error = 0.0
    step_count = 0
    previous_wrap_count = 0
    finite = True
    while simulation_app.is_running():
        with torch.inference_mode():
            target_before_step = command.target_joint_pos.clone()
            env.step(actions)
            visualize_skeleton()
            formula_error = torch.max(
                torch.abs(action_term.last_joint_target - target_before_step)
            ).item()
            actual = command.robot.data.joint_pos.torch[:, command.joint_ids]
            finite = bool(torch.isfinite(actual).all())
            if not finite:
                print(f"[ERROR] non-finite robot state at step {step_count + 1}")
                break
            tracking_error = torch.linalg.vector_norm(
                actual - target_before_step, dim=-1
            ).max().item()
            max_formula_error = max(max_formula_error, formula_error)
            max_tracking_error = max(max_tracking_error, tracking_error)
            step_count += 1
            wrap_count = int(command.wrap_count[0].item())
            if wrap_count != previous_wrap_count:
                print(
                    f"[zero-residual+skeleton] cycle {wrap_count}: "
                    f"formula={max_formula_error:.3e} rad, "
                    f"physics tracking={max_tracking_error:.3e} rad"
                )
                previous_wrap_count = wrap_count
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break

    print("[zero-residual+skeleton] summary")
    print(f"  steps: {step_count}")
    print(f"  max formula error: {max_formula_error:.9g} rad")
    print(f"  max physics tracking error: {max_tracking_error:.9g} rad")
    print(f"  finite: {finite}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
