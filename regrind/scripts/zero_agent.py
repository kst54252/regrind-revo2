# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--reference", type=str, default=None, help="Override reference trajectory path.")
parser.add_argument("--max_steps", type=int, default=0, help="Exit after N steps; 0 runs until the window closes.")
parser.add_argument("--real_time", action="store_true", help="Throttle simulation to environment step_dt.")
parser.add_argument(
    "--random-placement",
    action="store_true",
    help="Sample can/reference XY from the strict-IK-validated table rectangle on reset.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import regrind.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.reference is not None:
        if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "reference"):
            raise ValueError("--reference requires an environment command named 'reference'")
        env_cfg.commands.reference.trajectory_path = args_cli.reference
        env_cfg.commands.reference.rsi_enabled = False
        env_cfg.commands.reference.loop = True
        env_cfg.commands.reference.enable_reset_perturbation = False
    if args_cli.random_placement:
        if env_cfg.commands.reference.joint_reference != "revo2":
            raise ValueError(
                "--random-placement is for floating-hand replay; generate strict IK before "
                "using a sampled placement with --legacy-arm-rl"
            )
        env_cfg.commands.reference.randomize_object_xy = True
        # Demo-end termination causes an environment reset and a fresh sample.
        env_cfg.commands.reference.loop = False
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    step_count = 0
    finite = True
    while simulation_app.is_running():
        step_started = time.perf_counter()
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)
            finite = bool(torch.isfinite(env.unwrapped.scene["robot"].data.joint_pos.torch).all())
            step_count += 1
            if not finite or (args_cli.max_steps > 0 and step_count >= args_cli.max_steps):
                break
        if args_cli.real_time:
            remaining = env.unwrapped.step_dt - (time.perf_counter() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)

    print(f"[zero-agent] steps={step_count}, finite={finite}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
