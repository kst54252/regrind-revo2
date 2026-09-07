"""Inspect REGRIND observations, rewards, and RSI for RB3+Revo2+tuna."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Regrind-RB3-Revo2-TunaCan-Play-v0")
parser.add_argument("--reference", required=True)
parser.add_argument("--object-keypoints", required=True)
parser.add_argument("--max_steps", type=int, default=0, help="0 runs until the viewer closes")
parser.add_argument("--print_every", type=int, default=30)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--check_online_reset", action="store_true",
                    help="Assert post-RSI online arm FK and command-buffer agreement")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import regrind.tasks  # noqa: F401


def _all_finite(observations: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in observations.values())


def _check_online_reset(base_env, command, env_ids):
    """Check the selected reset frame, not the next advanced command frame."""
    action = base_env.action_manager.get_term("root_pose")
    measured = command.robot.data.joint_pos.torch[:, action._joint_ids]
    for index in env_ids.detach().cpu().tolist():
        frame = int(command.last_rsi_frame[index])
        position = command.reference.wrist_pos[frame] + command.placement_offset[index].cpu().numpy()
        quaternion = command.reference.wrist_quat_xyzw[frame]
        pos_error, rot_error, _, _ = action._kinematics.pose_error(
            measured[index].cpu().numpy(), position, quaternion
        )
        if not (pos_error <= action.cfg.position_tolerance_m
                and rot_error <= action.cfg.orientation_tolerance_rad):
            raise AssertionError(f"reset FK mismatch env={index} frame={frame}: "
                                 f"position={pos_error} rotation={rot_error}")
        for target in (action.last_joint_target, action.applied_joint_target,
                       action._interpolation_start_target):
            torch.testing.assert_close(target[index], measured[index])
        if not bool(action.ik_success[index]):
            raise AssertionError(f"reset IK failed for env={index}")
    print(f"[reset check] passed envs={env_ids.tolist()} RSI={command.last_rsi_frame[env_ids].tolist()}")


def _print_step(base_env, command, reward, step, reset):
    components = {
        name: float(value[0])
        for name, value in base_env.reward_manager.get_active_iterable_terms(0)
    }
    print(f"[MDP debug] step={step}")
    state_label = "post-autoreset state" if reset else "current state"
    reward_label = "pre-autoreset transition" if reset else "current transition"
    print(f"  state snapshot:         {state_label}")
    print(
        f"  reference frame/phase: {int(command.time_steps[0])}/"
        f"{float(command.phi[0]):.6f}"
    )
    print(f"  RSI selected frame:    {int(command.last_rsi_frame[0])}")
    print(
        "  object keypoint error: "
        f"{float(command.metrics['error_object_keypoints_pos'][0]):.9g} m"
    )
    print(
        "  wrist error:           "
        f"pos={float(command.metrics['error_hand_wrist_pos'][0]):.9g} m, "
        f"rot={float(command.metrics['error_hand_wrist_rot'][0]):.9g} rad"
    )
    print(f"  reward components ({reward_label}): {components}")
    print(f"  total reward ({reward_label}):      {float(reward[0]):.9g}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.reference.trajectory_path = args_cli.reference
    env_cfg.commands.reference.object_keypoints_path = args_cli.object_keypoints
    env_cfg.commands.reference.rsi_enabled = True
    env_cfg.commands.reference.loop = False
    env_cfg.commands.reference.enable_reset_perturbation = False
    env_cfg.commands.reference.debug_output = True
    if args_cli.check_online_reset and not env_cfg.commands.reference.reset_wrist_ik_action:
        raise ValueError("--check_online_reset requires the Online task")

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    observations, _ = env.reset()
    command = base_env.command_manager.get_term("reference")
    action_shape = env.action_space.shape
    policy_shape = tuple(observations["policy"].shape)
    critic_shape = tuple(observations["critic"].shape)

    print("[MDP debug] environment ready")
    print(f"  actor observation shape:  {policy_shape}")
    print(f"  critic observation shape: {critic_shape}")
    print(f"  action shape:             {action_shape}")
    print(f"  fingertip tensor shape:   {tuple(command.current_fingertips_pos.shape)}")
    print(f"  object keypoint shape:    {tuple(command.current_object_keypoints_pos.shape)}")
    print(f"  initial RSI frame:        {int(command.last_rsi_frame[0])}")
    n = args_cli.num_envs
    expected_dims = (67, 94) if env_cfg.commands.reference.expose_revo2_as_hand else (76, 109)
    if action_shape != (n, 12):
        raise RuntimeError(f"expected action shape ({n},12), got {action_shape}")
    if policy_shape != (n, expected_dims[0]) or critic_shape != (n, expected_dims[1]):
        raise RuntimeError(
            f"unexpected observation shapes: actor={policy_shape}, critic={critic_shape}"
        )
    if command.current_fingertips_pos.shape != (n, 5, 3):
        raise RuntimeError("critic must use exactly five real Revo2 fingertip links")
    if command.current_object_keypoints_pos.shape != (n, 50, 3):
        raise RuntimeError("object tracking must use 50 local surface keypoints")

    if args_cli.check_online_reset:
        _check_online_reset(base_env, command, torch.arange(n, device=base_env.device))

    actions = torch.zeros(action_shape, dtype=torch.float32, device=base_env.device)
    step = 0
    finite = _all_finite(observations)
    while simulation_app.is_running():
        with torch.inference_mode():
            observations, reward, terminated, truncated, _ = env.step(actions)
        step += 1
        state_finite = (
            _all_finite(observations)
            and bool(torch.isfinite(reward).all())
            and bool(torch.isfinite(command.current_hand_joint_pos).all())
            and bool(torch.isfinite(command.current_object_keypoints_pos).all())
        )
        finite = finite and state_finite
        reset = bool((terminated | truncated)[0])
        reset_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        if args_cli.check_online_reset and reset_ids.numel():
            _check_online_reset(base_env, command, reset_ids)
        if step == 1 or step % args_cli.print_every == 0 or reset:
            _print_step(base_env, command, reward, step, reset)
            print(f"  reset this step:        {reset}")
            print(f"  NaN/Inf free:           {state_finite}")
        if not state_finite:
            raise RuntimeError(f"NaN/Inf detected at debug step {step}")
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break

    print("[MDP debug] summary")
    print(f"  steps:       {step}")
    print(f"  finite:      {finite}")
    print(f"  actor shape: {policy_shape}")
    print(f"  critic shape:{critic_shape}")
    print(f"  action shape:{action_shape}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
