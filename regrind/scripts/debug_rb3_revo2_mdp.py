"""Inspect REGRIND observations, rewards, and RSI for RB3+Revo2+tuna."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Regrind-RB3-Revo2-Tuna-Play-v0")
parser.add_argument("--reference", required=True)
parser.add_argument("--object-keypoints", required=True)
parser.add_argument("--max_steps", type=int, default=0, help="0 runs until the viewer closes")
parser.add_argument("--print_every", type=int, default=30)
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
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.commands.reference.trajectory_path = args_cli.reference
    env_cfg.commands.reference.object_keypoints_path = args_cli.object_keypoints
    env_cfg.commands.reference.rsi_enabled = True
    env_cfg.commands.reference.loop = False
    env_cfg.commands.reference.enable_reset_perturbation = False
    env_cfg.commands.reference.debug_output = True

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
    if action_shape != (1, 12):
        raise RuntimeError(f"expected action shape (1,12), got {action_shape}")
    if policy_shape != (1, 76) or critic_shape != (1, 109):
        raise RuntimeError(
            f"unexpected observation shapes: actor={policy_shape}, critic={critic_shape}"
        )
    if command.current_fingertips_pos.shape != (1, 5, 3):
        raise RuntimeError("critic must use exactly five real Revo2 fingertip links")
    if command.current_object_keypoints_pos.shape != (1, 50, 3):
        raise RuntimeError("object tracking must use 50 local surface keypoints")

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
