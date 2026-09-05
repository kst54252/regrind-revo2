# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections import Counter
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=270, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--headless", dest="legacy_headless", action="store_true", default=False, help="Run Isaac Sim without a window."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--zero_actions", action="store_true", default=False, help="Use zero actions to step the environment.")
parser.add_argument(
    "--auto_gravity_from_ckpt",
    action="store_true",
    default=False,
    help=(
        "Infer the gravity the checkpoint was trained under from its iteration number and the run's "
        "gravity curriculum, and fix sim.gravity to that value for eval (overrides the cfg gravity)."
    ),
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after this many environment steps (0 keeps the interactive loop running).",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=0,
    help="Stop after this many completed episodes and print a deterministic success summary.",
)
parser.add_argument(
    "--rollout-path",
    "--rollout_path",
    type=str,
    default=None,
    help="Save environment 0 floating-hand rollout as HDF5 (one episode only).",
)
parser.add_argument(
    "--rollout-frames",
    "--rollout_frames",
    type=int,
    default=0,
    help="Frames to save; 0 uses the loaded reference length.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = args_cli.legacy_headless
del args_cli.legacy_headless
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# AppLauncher may append Kit arguments to sys.argv; keep only user Hydra overrides.
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

import gymnasium as gym
import h5py
import importlib.metadata as metadata
import numpy as np
import os
from pathlib import Path
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import regrind.tasks  # noqa: F401


def _tensor_row(value) -> np.ndarray:
    """Copy environment zero from an Isaac tensor to host NumPy."""

    return value[0].detach().cpu().numpy().copy()


def _save_floating_rollout(path: str, samples: list[dict[str, np.ndarray]], command, dt: float) -> None:
    """Write a policy rollout in the format consumed by the strict-IK bridge."""

    if not samples:
        raise RuntimeError("cannot save an empty floating-hand rollout")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = tuple(samples[0])
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in keys}
    with h5py.File(output, "w") as h5_file:
        for key, value in arrays.items():
            h5_file.create_dataset(key, data=value)
        h5_file.create_dataset("fps", data=1.0 / dt)
        h5_file.create_dataset("quat_convention", data="xyzw", dtype=h5py.string_dtype("utf-8"))
        h5_file.create_dataset(
            "revo2_joint_names",
            data=np.asarray(command.controlled_joint_names, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        if "revo2_follower_joints" in arrays:
            h5_file.create_dataset(
                "revo2_follower_joint_names",
                data=np.asarray(command.follower_names, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
        h5_file.create_dataset(
            "source_reference",
            data=str(command.reference.path),
            dtype=h5py.string_dtype("utf-8"),
        )
        if "mano_joint_world" in arrays:
            h5_file.create_dataset(
                "mano_joint_order",
                data="revo_semantic_kp00_to_kp20",
                dtype=h5py.string_dtype("utf-8"),
            )
        h5_file.create_dataset("rollout_complete", data=len(samples) == command.reference.frames)
    print(f"[ROLLOUT] saved {len(samples)} floating-hand frames to {output}")


def _floating_snapshot(
    command,
    action: torch.Tensor,
    joint_drive_target: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Capture the physical floating-hand/object state for downstream RB3 IK."""

    sample = {
        "frame_index": np.asarray(int(command.time_steps[0].item()), dtype=np.int64),
        "reference_phase": np.asarray(float(command.phi[0].item()), dtype=np.float32),
        "wrist_pos": _tensor_row(command.current_hand_wrist_pos),
        "wrist_quat": _tensor_row(command.current_hand_wrist_quat),
        "revo2_joints": _tensor_row(command.current_hand_joint_pos),
        "revo2_fingertip_pos": _tensor_row(command.current_fingertips_pos),
        "revo2_follower_joints": _tensor_row(
            command.robot.data.joint_pos.torch[:, command.follower_ids]
        ),
        # Unlike target_revo2_joints (the retargeting reference), this is the
        # clipped q_ref + policy residual that generated the physical grip.
        "revo2_joint_drive_target": _tensor_row(joint_drive_target),
        "object_pos": _tensor_row(command.current_object_pos),
        "object_quat": _tensor_row(command.current_object_quat),
        "floating_action": _tensor_row(action),
        "target_wrist_pos": _tensor_row(command.target_hand_wrist_pos),
        "target_wrist_quat": _tensor_row(command.target_hand_wrist_quat),
        "target_revo2_joints": _tensor_row(command.target_hand_joint_pos),
        "target_object_pos": _tensor_row(command.target_object_pos),
        "target_object_quat": _tensor_row(command.target_object_quat),
    }
    mano = command.reference.mano_joint_world_semantic
    if mano is not None:
        sample["mano_joint_world"] = (
            np.asarray(mano[int(command.time_steps[0].item())], dtype=np.float32)
            + _tensor_row(command.placement_offset).astype(np.float32)
        )
    return sample


# PLACEHOLDER: Extension template (do not remove this comment)

def _infer_gravity_from_checkpoint(resume_path: str, num_steps_per_env: int) -> tuple[float, float, float] | None:
    """Infer the gravity a checkpoint was trained under from its iteration and the run's curriculum.

    The gravity curriculum (``curriculum_gravity_with_randomization``) maps ``env.common_step_counter``
    to a gravity range, and ``common_step_counter == iteration * num_steps_per_env``. The iteration is
    encoded in the checkpoint filename (``model_<iter>.pt``) and the exact schedule (including any
    train-time Hydra overrides) is dumped to ``<run>/params/env.yaml``. We reproduce the same stage
    lookup the curriculum uses and collapse the resulting range to its midpoint.

    Returns the ``(gx, gy, gz)`` gravity, or ``None`` if it cannot be determined.
    """
    import re
    import yaml

    model_file = os.path.basename(resume_path)
    m = re.fullmatch(r"model_(\d+)\.pt", model_file)
    if m is None:
        print(f"[WARN] --auto_gravity_from_ckpt: cannot parse iteration from {model_file!r}; skipping.")
        return None
    iteration = int(m.group(1))

    env_yaml_path = os.path.join(os.path.dirname(resume_path), "params", "env.yaml")
    if not os.path.isfile(env_yaml_path):
        print(f"[WARN] --auto_gravity_from_ckpt: no dumped config at {env_yaml_path}; skipping.")
        return None

    # env.yaml is dumped with !!python/tuple tags, so use unsafe_load.
    with open(env_yaml_path) as f:
        env_dump = yaml.unsafe_load(f)
    try:
        gravity_stages = env_dump["events"]["curriculum_gravity"]["params"]["gravity_stages"]
    except (KeyError, TypeError):
        print(f"[WARN] --auto_gravity_from_ckpt: no curriculum_gravity in {env_yaml_path}; skipping.")
        return None

    current_step = iteration * num_steps_per_env

    # Mirror curriculum_gravity_with_randomization: take the last stage whose threshold <= current_step.
    g_min: tuple[float, float, float] = (0.0, 0.0, -9.81)
    g_max: tuple[float, float, float] = (0.0, 0.0, -9.81)
    for step_threshold, gravity_value_min, gravity_value_max in sorted(gravity_stages, key=lambda s: s[0]):
        if current_step >= step_threshold:
            g_min = tuple(gravity_value_min)
            g_max = tuple(gravity_value_max)
        else:
            break

    gravity = tuple((a + b) / 2.0 for a, b in zip(g_min, g_max))
    print(
        f"[INFO] --auto_gravity_from_ckpt: iter={iteration} x {num_steps_per_env} steps = {current_step} -> "
        f"gravity range {g_min}..{g_max} -> midpoint {gravity}"
    )
    return gravity  # type: ignore[return-value]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    model_name = resume_path.split('/')[-1].split('.')[0]

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # optionally fix gravity to the value the checkpoint was trained under (gravity curriculum)
    if args_cli.auto_gravity_from_ckpt:
        inferred_gravity = _infer_gravity_from_checkpoint(resume_path, agent_cfg.num_steps_per_env)
        if inferred_gravity is not None:
            env_cfg.sim.gravity = inferred_gravity

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": model_name,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(export_model_dir, filename=f"{model_name}_policy.pt")
    runner.export_policy_to_onnx(export_model_dir, filename=f"{model_name}_policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    completed_episodes = 0
    successful_episodes = 0
    termination_counts: Counter[str] = Counter()
    reward_sum = 0.0
    reward_samples = 0
    rollout_samples: list[dict[str, np.ndarray]] = []
    rollout_command = None
    rollout_joint_action = None
    rollout_frame_limit = 0
    if args_cli.rollout_path is not None:
        if env.num_envs != 1:
            raise ValueError("--rollout_path requires --num_envs 1")
        try:
            rollout_command = env.unwrapped.command_manager.get_term("reference")
        except (AttributeError, KeyError) as error:
            raise RuntimeError("rollout export requires a command term named 'reference'") from error
        required = (
            "current_hand_wrist_pos",
            "current_hand_wrist_quat",
            "current_hand_joint_pos",
            "current_object_pos",
            "current_object_quat",
        )
        missing = [name for name in required if not hasattr(rollout_command, name)]
        if missing:
            raise RuntimeError(f"reference command cannot export a floating rollout; missing {missing}")
        rollout_joint_action = env.unwrapped.action_manager.get_term("joint_pos")
        if not hasattr(rollout_joint_action, "last_joint_target"):
            raise RuntimeError(
                "floating rollout export requires joint_pos.last_joint_target"
            )
        rollout_frame_limit = args_cli.rollout_frames or rollout_command.reference.frames
        if rollout_frame_limit <= 0:
            raise ValueError("--rollout_frames must be positive")
        initial_action = torch.zeros((1, env.action_space.shape[-1]), device=env.unwrapped.device)
        rollout_samples.append(
            _floating_snapshot(
                rollout_command,
                initial_action,
                rollout_command.target_hand_joint_pos,
            )
        )
        print(
            f"[ROLLOUT] recording environment 0 for at most {rollout_frame_limit} frames; "
            "recording stops on the first episode termination"
        )
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            if args_cli.zero_actions:
                actions.zero_()
            obs, rewards, dones, extras = env.step(actions)

        reward_sum += float(rewards.detach().sum().cpu())
        reward_samples += int(rewards.numel())
        done_count = int(dones.detach().sum().cpu())
        if done_count:
            success_count = 0
            termination_manager = getattr(env.unwrapped, "termination_manager", None)
            if termination_manager is not None:
                for term_name in termination_manager.active_terms:
                    try:
                        term = termination_manager.get_term(term_name)
                    except (AttributeError, KeyError):
                        continue
                    termination_counts[term_name] += int((term & dones).detach().sum().cpu())
            try:
                success = termination_manager.get_term("success")
                success_count = int((success & dones).detach().sum().cpu())
            except (AttributeError, KeyError, TypeError):
                # Generic Isaac Lab tasks may not define a named success term.
                pass
            completed_episodes += done_count
            successful_episodes += success_count
            if args_cli.eval_episodes:
                print(
                    f"[EVAL] completed={completed_episodes} "
                    f"successful={successful_episodes} "
                    f"rate={successful_episodes / completed_episodes:.2%}"
                )

        timestep += 1
        if rollout_command is not None:
            if bool(dones[0].item()):
                print(
                    f"[ROLLOUT] episode terminated after {len(rollout_samples)} saved frames; "
                    "the automatic reset state was not appended"
                )
                break
            rollout_samples.append(
                _floating_snapshot(
                    rollout_command,
                    actions,
                    rollout_joint_action.last_joint_target,
                )
            )
            if len(rollout_samples) >= rollout_frame_limit:
                break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break
        if args_cli.eval_episodes > 0 and completed_episodes >= args_cli.eval_episodes:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.max_steps > 0 or args_cli.eval_episodes > 0 or args_cli.rollout_path is not None:
        success_rate = (
            successful_episodes / completed_episodes if completed_episodes else 0.0
        )
        mean_step_reward = reward_sum / max(reward_samples, 1)
        print("\n[deterministic policy evaluation]")
        print(f"  steps:              {timestep}")
        print(f"  completed episodes: {completed_episodes}")
        print(f"  successful episodes:{successful_episodes}")
        print(f"  success rate:       {success_rate:.2%}")
        print(f"  mean step reward:   {mean_step_reward:.8g}")
        if termination_counts:
            print("  termination counts:")
            for term_name, count in sorted(termination_counts.items()):
                print(f"    {term_name}: {count}")

    if args_cli.rollout_path is not None:
        _save_floating_rollout(args_cli.rollout_path, rollout_samples, rollout_command, dt)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
