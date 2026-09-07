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
parser.add_argument("--video-output-dir", default=None,
                    help="Optional fresh recording directory; existing directories are rejected to preserve videos.")
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
parser.add_argument(
    "--arm-tracking-path",
    "--arm_tracking_path",
    type=str,
    default=None,
    help=(
        "For the online RB3 task, save the first episode's commanded/measured "
        "arm and wrist response as NPZ and print a delay/tracking report."
    ),
)
parser.add_argument(
    "--arm-tracking-max-lag",
    "--arm_tracking_max_lag",
    type=int,
    default=8,
    help="Maximum causal control-step lag considered by arm tracking analysis.",
)
parser.add_argument("--arm-execution-trace", default=None,
                    help="Optional read-only physics-substep JSONL trace of the online arm path.")
parser.add_argument("--arm-execution-full-state", action="store_true",
                    help="Also record all arm/hand targets and reset states for paired contact replay.")
parser.add_argument("--arm-contact-replay", default=None,
                    help="Replay one episode's recorded arm/hand substep commands from a full-state trace.")
parser.add_argument("--arm-contact-episode", type=int, default=15)
parser.add_argument("--arm-contact-condition", choices=("present", "absent"), default="present")
parser.add_argument("--arm-contact-output", default=None)
parser.add_argument("--arm-actuator-diagnostic", action="store_true",
                    help="Read-only arm drive/effort observations during deterministic contact replay.")
parser.add_argument("--arm-velocity-path-trace", default=None,
                    help="Recorded replay only: use same-step v_path from a prior actuator trace with the same contact condition.")
parser.add_argument("--arm-velocity-target", choices=("zero", "v_path"), default="zero",
                    help="Online RB3 only: velocity from final interpolated position commands at physics dt (default: existing zero targets).")
parser.add_argument("--arm-evaluation-states", default=None,
                    help="Online paired evaluation only: restore named initial placements/states from a prior full-state execution trace.")
parser.add_argument(
    "--rb3-stiffness-scale",
    type=float,
    default=1.0,
    help="Runtime multiplier for the assembled RB3 stiffness (play-arm only).",
)
parser.add_argument(
    "--rb3-damping-scale",
    type=float,
    default=1.0,
    help="Runtime multiplier for the assembled RB3 damping (play-arm only).",
)
parser.add_argument(
    "--rb3-effort-scale",
    type=float,
    default=1.0,
    help="Runtime multiplier for the assembled RB3 effort limits (play-arm only).",
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
import importlib.util
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


def _save_arm_tracking_telemetry(
    path: str,
    samples: list[dict[str, np.ndarray]],
    dt: float,
    max_lag_steps: int,
    actuator_scales: tuple[float, float, float],
) -> None:
    """Persist the first online-arm episode and run the standalone analyzer."""

    if not samples:
        raise RuntimeError("no online RB3 samples were captured")
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError(f"--arm-tracking-path must end in .npz: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.stack([sample[key] for sample in samples])
        for key in samples[0]
    }
    arrays["dt"] = np.asarray(dt, dtype=np.float64)
    arrays["quat_convention"] = np.asarray("xyzw")
    arrays["rb3_stiffness_scale"] = np.asarray(actuator_scales[0], dtype=np.float64)
    arrays["rb3_damping_scale"] = np.asarray(actuator_scales[1], dtype=np.float64)
    arrays["rb3_effort_scale"] = np.asarray(actuator_scales[2], dtype=np.float64)
    np.savez_compressed(output, **arrays)

    # Load by repository path so this also works with Isaac launchers that do
    # not put the repository root on sys.path.
    analyzer_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "rb3_revo2_ik"
        / "analyze_arm_tracking.py"
    )
    spec = importlib.util.spec_from_file_location(
        "regrind_arm_tracking_analyzer", analyzer_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arm tracking analyzer: {analyzer_path}")
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)
    report = analyzer.analyze_arrays(arrays, max_lag_steps=max_lag_steps)
    analyzer.print_report(report, output)


def _scale_numeric_config(value, factor: float):
    if factor <= 0.0 or not np.isfinite(factor):
        raise ValueError(f"actuator scale must be positive and finite, got {factor}")
    if isinstance(value, dict):
        return {key: item * factor for key, item in value.items()}
    return value * factor


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

    actuator_scales = (
        args_cli.rb3_stiffness_scale,
        args_cli.rb3_damping_scale,
        args_cli.rb3_effort_scale,
    )
    if actuator_scales != (1.0, 1.0, 1.0):
        try:
            rb3_actuator = env_cfg.scene.robot.actuators["rb3_arm"]
        except (AttributeError, KeyError) as error:
            raise RuntimeError(
                "RB3 actuator scale options require an assembled RB3 task"
            ) from error
        rb3_actuator.stiffness = _scale_numeric_config(
            rb3_actuator.stiffness, args_cli.rb3_stiffness_scale
        )
        rb3_actuator.damping = _scale_numeric_config(
            rb3_actuator.damping, args_cli.rb3_damping_scale
        )
        rb3_actuator.effort_limit_sim = _scale_numeric_config(
            rb3_actuator.effort_limit_sim, args_cli.rb3_effort_scale
        )
        print(
            "[RB3 actuator runtime scales] "
            f"stiffness={args_cli.rb3_stiffness_scale:g}, "
            f"damping={args_cli.rb3_damping_scale:g}, "
            f"effort={args_cli.rb3_effort_scale:g}"
        )

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

    if args_cli.arm_velocity_target != "zero":
        if not hasattr(env_cfg.actions.root_pose, "velocity_target_mode") or args_cli.arm_contact_replay:
            raise ValueError("--arm-velocity-target is only for live online RB3 evaluation")
        env_cfg.actions.root_pose.velocity_target_mode = args_cli.arm_velocity_target

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        if args_cli.video_output_dir:
            video_folder = os.path.abspath(os.path.expanduser(args_cli.video_output_dir))
            if os.path.exists(video_folder):
                raise FileExistsError(video_folder)
        video_kwargs = {
            "video_folder": video_folder,
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

    if args_cli.arm_evaluation_states:
        if args_cli.arm_contact_replay or not args_cli.arm_execution_full_state or env.num_envs != 1:
            raise ValueError("Paired initial states require num_envs=1 and live full-state tracing")
        state_path = Path(__file__).resolve().parents[3] / "tools/rb3_revo2_ik/paired_arm_states.py"
        state_spec = importlib.util.spec_from_file_location("regrind_paired_arm_states", state_path)
        state_module = importlib.util.module_from_spec(state_spec)
        state_spec.loader.exec_module(state_module)
        paired_states = state_module.PairedArmStates(env.unwrapped, args_cli.arm_evaluation_states, args_cli.eval_episodes)

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
    online_command = None
    online_ik_action = None
    online_object_positions: list[np.ndarray] = []
    online_episode_positions: list[np.ndarray] = []
    online_wrist_position_errors: list[float] = []
    online_arm_tracking_errors: list[float] = []
    online_ik_failures: list[int] = []
    arm_tracking_samples: list[dict[str, np.ndarray]] = []
    capture_arm_tracking = args_cli.arm_tracking_path is not None
    try:
        candidate_command = env.unwrapped.command_manager.get_term("reference")
        candidate_action = env.unwrapped.action_manager.get_term("root_pose")
        if hasattr(candidate_action, "ik_success") and hasattr(
            candidate_command, "current_object_pos"
        ):
            online_command = candidate_command
            online_ik_action = candidate_action
            online_object_positions.append(_tensor_row(online_command.current_object_pos))
            online_episode_positions.append(_tensor_row(online_command.current_object_pos))
            print(
                "[ONLINE] closed-loop floating policy -> strict RB3 IK diagnostics enabled"
            )
    except (AttributeError, KeyError):
        pass
    if capture_arm_tracking and online_command is None:
        raise RuntimeError(
            "--arm-tracking-path requires the online assembled task; use ./scripts/rl.sh play-arm"
        )
    if args_cli.arm_velocity_path_trace and (
        not args_cli.arm_contact_replay or not args_cli.arm_actuator_diagnostic
    ):
        raise ValueError("--arm-velocity-path-trace requires contact replay and --arm-actuator-diagnostic")
    if args_cli.arm_actuator_diagnostic and not args_cli.arm_contact_replay:
        raise ValueError("--arm-actuator-diagnostic requires --arm-contact-replay")
    if args_cli.arm_contact_replay:
        if online_command is None or env.num_envs != 1 or not args_cli.arm_contact_output:
            raise ValueError("contact replay requires online num_envs=1 and --arm-contact-output")
        if args_cli.arm_execution_trace or args_cli.zero_actions:
            raise ValueError("contact replay cannot combine normal tracing or zero-actions")
        test_path = Path(__file__).resolve().parents[3] / "tools/rb3_revo2_ik/replay_arm_contact.py"
        spec = importlib.util.spec_from_file_location("regrind_contact_replay", test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run(env, obs, policy, args_cli, resume_path)
        env.close()
        return
    if args_cli.arm_execution_full_state and not args_cli.arm_execution_trace:
        raise ValueError("--arm-execution-full-state requires --arm-execution-trace")
    execution_trace = None
    if args_cli.arm_execution_trace:
        if online_command is None or env.num_envs != 1:
            raise ValueError("--arm-execution-trace requires play-arm with num_envs=1")
        trace_path = Path(__file__).resolve().parents[3] / "tools/rb3_revo2_ik/trace_arm_execution.py"
        spec = importlib.util.spec_from_file_location("regrind_execution_trace", trace_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        execution_trace = module.ArmExecutionTrace(
            env.unwrapped, args_cli.arm_execution_trace, resume_path,
            vars(args_cli), hydra_args, full_state=args_cli.arm_execution_full_state,
        )
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
            if execution_trace is not None:
                execution_trace.after_env_step(dones)

        if online_command is not None and not bool(dones[0].item()):
            online_object_positions.append(_tensor_row(online_command.current_object_pos))
            online_episode_positions.append(_tensor_row(online_command.current_object_pos))
            online_wrist_position_errors.append(
                float(
                    torch.linalg.vector_norm(
                        online_command.current_hand_wrist_pos[0]
                        - online_ik_action.target_pos[0]
                    ).item()
                )
            )
            actual_arm = online_command.robot.data.joint_pos.torch[
                0, online_ik_action._joint_ids
            ]
            online_arm_tracking_errors.append(
                float(
                    torch.linalg.vector_norm(
                        actual_arm - online_ik_action.last_joint_target[0]
                    ).item()
                )
            )
            if not bool(online_ik_action.ik_success[0].item()):
                online_ik_failures.append(timestep)
            # ManagerBasedRLEnv has already reset a done environment before it
            # returns from step(). Exclude that reset sample, then stop at the
            # first episode boundary so lag estimation never crosses a reset.
            if capture_arm_tracking:
                arm_tracking_samples.append(
                    {
                        "frame_index": np.asarray(
                            int(online_command.time_steps[0].item()), dtype=np.int64
                        ),
                        "target_rb3_joints": _tensor_row(
                            online_ik_action.last_joint_target
                        ),
                        "actual_rb3_joints": _tensor_row(actual_arm.unsqueeze(0)),
                        "actual_rb3_joint_velocity": _tensor_row(
                            online_command.robot.data.joint_vel.torch[
                                0, online_ik_action._joint_ids
                            ].unsqueeze(0)
                        ),
                        "target_wrist_pos": _tensor_row(online_ik_action.target_pos),
                        "actual_wrist_pos": _tensor_row(
                            online_command.current_hand_wrist_pos
                        ),
                        "target_wrist_quat_xyzw": _tensor_row(
                            online_ik_action.target_quat
                        ),
                        "actual_wrist_quat_xyzw": _tensor_row(
                            online_command.current_hand_wrist_quat
                        ),
                        "object_pos": _tensor_row(online_command.current_object_pos),
                    }
                )
        if online_command is not None and capture_arm_tracking and bool(dones[0].item()):
            capture_arm_tracking = False
        if online_command is not None and bool(dones[0].item()):
            # Autoreset replaces terminal state before step returns. Report
            # the last observed pre-reset sample explicitly, per episode;
            # never interpret randomized episode placement as object motion.
            positions = np.asarray(online_episode_positions)
            print(f"[arm episode env=0] start={positions[0].tolist()} "
                  f"last_pre_reset={positions[-1].tolist()} "
                  f"last_lift={positions[-1, 2] - positions[0, 2]:.6f} m "
                  f"max_lift={positions[:, 2].max() - positions[0, 2]:.6f} m")
            online_episode_positions = [_tensor_row(online_command.current_object_pos)]

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

    if online_command is not None and online_object_positions:
        object_positions = np.asarray(online_object_positions)
        object_delta = object_positions[-1] - object_positions[0]
        reference_delta = (
            online_command.reference.object_pos[-1]
            - online_command.reference.object_pos[0]
        )
        print("\n[online RB3+Revo2 physical diagnostics]")
        if completed_episodes > 1:
            print("  XYZ below spans episodes; use per-episode lift above for grasp validation.")
        print(f"  object start xyz:       {object_positions[0].tolist()}")
        print(f"  object final xyz:       {object_positions[-1].tolist()}")
        print(f"  object delta xyz:       {object_delta.tolist()}")
        print(f"  object max z:           {float(object_positions[:, 2].max()):.8g} m")
        print(f"  reference delta xyz:    {reference_delta.tolist()}")
        print(f"  online IK failed steps: {sorted(set(online_ik_failures))}")
        if online_wrist_position_errors:
            print(
                "  wrist target error mean/max: "
                f"{float(np.mean(online_wrist_position_errors)):.8g} / "
                f"{float(np.max(online_wrist_position_errors)):.8g} m"
            )
        if online_arm_tracking_errors:
            print(
                "  arm q target error mean/max: "
                f"{float(np.mean(online_arm_tracking_errors)):.8g} / "
                f"{float(np.max(online_arm_tracking_errors)):.8g} rad"
            )

    if args_cli.rollout_path is not None:
        _save_floating_rollout(args_cli.rollout_path, rollout_samples, rollout_command, dt)

    if args_cli.arm_tracking_path is not None:
        _save_arm_tracking_telemetry(
            args_cli.arm_tracking_path,
            arm_tracking_samples,
            dt,
            args_cli.arm_tracking_max_lag,
            actuator_scales,
        )

    # close the simulator
    if execution_trace is not None:
        execution_trace.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
