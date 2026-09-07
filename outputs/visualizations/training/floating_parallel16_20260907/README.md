# Parallel PPO training capture

Final: `isaac_floating_parallel16_ppo_training_10s.mp4`, 1280x720, 30fps,
300 frames / 10 seconds. Sixteen floating Revo2/tuna environments are rendered
together during actual stochastic PPO collection and updates, not policy play.
Playback is simulation time; optimization/rendering wall-clock pauses are not
represented. This is an illustrative training scene, not a grasp success test.

Source: `logs/rsl_rl/floating_revo2_video/2026-09-07_15-42-01_parallel16_capture_10s_final/videos/train/rl-video-step-0.mp4`.
The raw video has 301 frames; the edit uses the first 300 with title/footer only.
The camera includes margins for lifted hands throughout the clip. Two framing
takes preceded the final capture; all use fresh runs from the same checkpoint
and only differ in camera/framing settings.

Executed through the existing launcher (no training source/config changes):

```bash
bash scripts/rl.sh train --sequence 20200709_143747_left \
  --num_envs 16 --max_iterations 14 --headless --resume \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --logger tensorboard --run_name parallel16_capture_10s_final \
  --video --video_length 300 --video_interval 100000 --max_visible_envs 16 \
  agent.experiment_name=floating_revo2_video \
  'env.viewer.eye=[0.4,2.6,5.8]' 'env.viewer.lookat=[0.4,0.0,0.12]' \
  env.viewer.origin_type=world
```

Validation: run exited 0, iterations 4999–5012 completed (14 updates, 24 steps
per environment per iteration), all 28 reported value/surrogate losses finite.
Training log: `training_final.log`; resolved configs, TensorBoard events and new
checkpoints remain in the source run directory. Normal training RSI,
randomization and gravity curriculum remain enabled. A fresh environment starts
its curriculum independently of the resumed model; this is not an exact
continuation of the original run's simulator state.

Original checkpoint SHA-256 remains:
`10eeeff86c405b07595cdd692d3447e9385da43dc90c881dbe1807465f9c4721`.
Existing checkpoint and previous runs were not overwritten. No physical robot
commands were sent. No source changes were needed for this capture.
