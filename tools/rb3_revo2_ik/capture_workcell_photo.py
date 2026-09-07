"""Capture actual Isaac single/parallel workcells at the normal reference reset.

No policy calls, training or replacement assets. Uses the existing online-play
task. The optional extended-arm pose is a one-time, photo-only initialization.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, choices=(1, 16), default=1)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--extended-arm', action='store_true', help='Photo-only: initialize six arm joints at zero, preserve hand posture')
parser.add_argument('--headless', dest='legacy_headless', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = args.legacy_headless
del args.legacy_headless
args.enable_cameras = True
if args.output.exists():
    raise FileExistsError(args.output)
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import regrind.tasks  # noqa: F401
from tools.rb3_revo2_ik.trace_arm_execution import array


def main():
    task = 'Regrind-RB3-Revo2-TunaCan-Online-Play-v0'
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs, use_fabric=True)
    cfg.seed = 42
    cfg.viewer.resolution = (1920, 1080)
    cfg.video_recorder.window_width = 1920
    cfg.video_recorder.window_height = 1080
    cfg.viewer.origin_type = 'world'
    cfg.viewer.eye = (2.5, 2.4, 1.7) if args.num_envs == 1 else (9., 8., 7.)
    cfg.viewer.lookat = (.25, 0., .18)
    if args.extended_arm:
        cfg.viewer.eye = (3., 2.8, 1.85) if args.num_envs == 1 else (10., 9., 8.)
        cfg.viewer.lookat = (.2, 0., .38)
    env = gym.make(task, cfg=cfg, render_mode='rgb_array').unwrapped
    try:
        env.reset()
        robot = env.scene['robot']
        # A normal zero-residual control step propagates articulation transforms
        # into the renderer; render-only after reset can show the stale USD pose.
        with torch.inference_mode():
            if args.extended_arm:
                from regrind.data.rb3_revo2_reference import RB3_JOINT_NAMES
                arm_ids = [robot.joint_names.index(name) for name in RB3_JOINT_NAMES]
                limits = array(robot.data.soft_joint_pos_limits)[:, arm_ids]
                if np.any(limits[..., 0] > 0) or np.any(limits[..., 1] < 0):
                    raise ValueError('Extended zero pose exceeds arm joint limits')
                zero = torch.zeros((env.num_envs, 6), device=env.device)
                robot.write_joint_state_to_sim(zero, zero, joint_ids=arm_ids)
                robot.set_joint_position_target_index(target=zero, joint_ids=arm_ids)
                robot.set_joint_velocity_target_index(target=zero, joint_ids=arm_ids)
                env.scene.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
                np.testing.assert_allclose(array(robot.data.joint_pos)[:, arm_ids], 0., atol=.005)
            else:
                env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
        for _ in range(30):
            pixels = env.render()
        pixels = np.asarray(pixels)
        if pixels.shape != (1080, 1920, 3) or np.std(pixels) < 1:
            raise ValueError(f'Invalid render: {pixels.shape}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.output, pixels)
        metadata = dict(task=task, num_envs=env.num_envs,
                        mode=('extended zero arm pose plus one physics step; no policy or training' if args.extended_arm else
                              'normal reference reset plus one zero-residual control step; no policy or training'),
                        extended_arm=args.extended_arm,
                        physics_dt=env.physics_dt, control_dt=env.step_dt,
                        reference=cfg.commands.reference.trajectory_path,
                        env_origins=array(env.scene.env_origins).tolist(),
                        joint_names=robot.joint_names,
                        initial_joint_positions=array(robot.data.joint_pos).tolist(),
                        eye=cfg.viewer.eye, lookat=cfg.viewer.lookat)
        args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('[photo complete]', args.output, 'environments', env.num_envs, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
