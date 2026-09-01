"""Launch the RB3+Revo2+tuna task and replay it with zero residual actions."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Regrind-RB3-Revo2-Tuna-Play-v0")
parser.add_argument("--reference", required=True, help="Final RB3+Revo2 HDF5/NPZ reference.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit after N steps; 0 keeps the viewer open until it is closed.",
)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import regrind.tasks  # noqa: F401


def _object_reference_errors(command):
    position_error = torch.linalg.vector_norm(
        command.current_object_pos - command.target_object_pos,
        dim=-1,
    ).max()
    quaternion_dot = torch.sum(
        command.current_object_quat * command.target_object_quat,
        dim=-1,
    ).abs().clamp(max=1.0)
    orientation_error = (2.0 * torch.acos(quaternion_dot)).max()
    linear_velocity_error = torch.linalg.vector_norm(
        command.current_object_lin_vel - command.target_object_lin_vel,
        dim=-1,
    ).max()
    angular_velocity_error = torch.linalg.vector_norm(
        command.current_object_ang_vel - command.target_object_ang_vel,
        dim=-1,
    ).max()
    return tuple(
        float(value.item())
        for value in (
            position_error,
            orientation_error,
            linear_velocity_error,
            angular_velocity_error,
        )
    )


def main():
    print("[zero-residual] parsing environment configuration", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.commands.reference.trajectory_path = args_cli.reference
    # Keep this legacy validator deterministic. The MDP debug entrypoint tests
    # random RSI and reference-end termination separately.
    env_cfg.commands.reference.rsi_enabled = False
    env_cfg.commands.reference.loop = True
    env_cfg.commands.reference.reset_object_on_loop = True
    env_cfg.commands.reference.debug_output = False
    env_cfg.terminations.demo_end_reached = None
    env_cfg.terminations.success = None
    env_cfg.terminations.object_deviation = None
    env_cfg.terminations.hand_far_from_object = None
    env_cfg.rewards.early_terminated_penalty = None
    print("[zero-residual] creating IsaacLab environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[zero-residual] IsaacLab environment created", flush=True)
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("reference")
    action_term = base_env.action_manager.get_term("joint_pos")

    if env.action_space.shape != (1, 12):
        raise RuntimeError(f"expected action space (1,12), got {env.action_space.shape}")
    print("[zero-residual] environment ready")
    print(f"  reference:        {command.reference.path}")
    print(f"  frames/fps:        {command.reference.frames}/{command.reference.fps:g}")
    print(f"  action dimension:  {env.action_space.shape[-1]}")
    print(f"  arm scale:         {env_cfg.residual_scale.arm:g} rad")
    print(f"  hand scale:        {env_cfg.residual_scale.hand:g} rad")
    print("  mode:              q_target = q_ref + scale * 0")
    print("  loop object reset: reference frame 0 pose + velocity")

    env.reset()
    initial_object_errors = _object_reference_errors(command)
    print(
        "[zero-residual] initial object reset errors: "
        f"position={initial_object_errors[0]:.3e} m, "
        f"orientation={initial_object_errors[1]:.3e} rad, "
        f"linear_velocity={initial_object_errors[2]:.3e} m/s, "
        f"angular_velocity={initial_object_errors[3]:.3e} rad/s"
    )
    actions = torch.zeros(env.action_space.shape, device=base_env.device)
    max_formula_error = 0.0
    max_tracking_error = 0.0
    step_count = 0
    previous_wrap_count = 0
    first_nonfinite_step = None
    last_finite_actual = None
    first_out_of_range_step = None
    while simulation_app.is_running():
        with torch.inference_mode():
            target_before_step = command.target_joint_pos.clone()
            env.step(actions)
            formula_error = torch.max(
                torch.abs(action_term.last_joint_target - target_before_step)
            ).item()
            actual = command.robot.data.joint_pos.torch[:, command.joint_ids]
            if not bool(torch.isfinite(actual).all()):
                first_nonfinite_step = step_count + 1
                print(
                    "[zero-residual] ERROR non-finite robot state at "
                    f"step={first_nonfinite_step}, reference_frame={int(command.time_steps[0])}"
                )
                if last_finite_actual is not None:
                    print(f"  previous joint state: {last_finite_actual[0].tolist()}")
                break
            last_finite_actual = actual.clone()
            if first_out_of_range_step is None and float(torch.max(torch.abs(actual))) > 10.0:
                first_out_of_range_step = step_count + 1
                print(
                    "[zero-residual] WARNING robot joint state first exceeded 10 rad at "
                    f"step={first_out_of_range_step}, reference_frame={int(command.time_steps[0])}: "
                    f"{actual[0].tolist()}"
                )
            tracking_error = torch.linalg.vector_norm(actual - target_before_step, dim=-1).max().item()
            max_formula_error = max(max_formula_error, formula_error)
            max_tracking_error = max(max_tracking_error, tracking_error)
            step_count += 1

            wrap_count = int(command.wrap_count[0].item())
            if wrap_count != previous_wrap_count:
                object_reset_errors = _object_reference_errors(command)
                print(
                    f"[zero-residual] completed cycle {wrap_count}: "
                    f"formula max={max_formula_error:.3e} rad, "
                    f"physics tracking max={max_tracking_error:.3e} rad; "
                    "object reset to reference frame 0 "
                    f"(position={object_reset_errors[0]:.3e} m, "
                    f"orientation={object_reset_errors[1]:.3e} rad, "
                    f"linear_velocity={object_reset_errors[2]:.3e} m/s, "
                    f"angular_velocity={object_reset_errors[3]:.3e} rad/s)"
                )
                previous_wrap_count = wrap_count
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break

    print("[zero-residual] validation summary")
    print(f"  steps:                    {step_count}")
    print(f"  max q_target formula err: {max_formula_error:.9g} rad")
    print(f"  max physics tracking err: {max_tracking_error:.9g} rad")
    finite = bool(torch.isfinite(actual).all())
    print(f"  finite:                   {finite}")
    print(f"  first non-finite step:    {first_nonfinite_step}")
    print(f"  first >10 rad step:       {first_out_of_range_step}")
    env.close()
    if not finite:
        raise RuntimeError(f"robot state became non-finite at step {first_nonfinite_step}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
