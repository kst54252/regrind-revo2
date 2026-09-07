# Floating Revo2 + tuna can RL

[Repository map](architecture.md) · [Current status](current-status.md) ·
[Isaac replay](ISAAC_SIM_REPLAY.md)

## Design

PPO controls a floating Revo2 hand, not RB3 joints. Deployment keeps the
floating policy contract and solves its wrist target with bounded RB3 IK:

```text
reference -> floating wrist/Revo2 residual PPO -> rollout
                                               -> online RB3 IK deployment
                                               -> offline RB3 reference conversion
```

The older combined RB3+Revo2 residual task remains available through
`--legacy-arm-rl`, but it is not the primary training path.

`scripts/rl.sh play-arm` remains the original mounted baseline. For opt-in
minimal/response/fast-IK experiments and paired-state reproduction, use the
[diagnostic command index](../scripts/README.md#experiments-and-failure-reproduction-opt-in).
These candidates do not alter training or default play settings.

## Registered tasks

| Task | Environments | Randomization | Purpose |
|---|---:|---|---|
| `Regrind-Floating-Revo2-TunaCan-Play-v0` | 1 | Off | deterministic play/export |
| `Regrind-Floating-Revo2-TunaCan-Smoke-v0` | 16 | On | short integration test |
| `Regrind-Floating-Revo2-TunaCan-v0` | 4096 | On | full training |

Configuration lives under
`regrind/source/regrind/regrind/tasks/manager_based/dexterous/config/revo2_floating/`.
Shared MDP terms are in the sibling `mdp/` package.

## Policy and environment contract

- Action `(N,12)`: wrist position/orientation residual `(6)` followed by six
  Revo2 leader residuals. Five distal joints remain deterministic mimics.
- Actor observation `(N,67)`: object and wrist state/history, Revo2 joint
  history, previous action, phase, and reference/action-base targets.
- Privileged critic observation `(N,94)`: actor data plus object velocity, five
  physical fingertips, and Revo2 joint velocity.
- Reward: 50 tuna surface-keypoint tracking, object velocity, wrist pose,
  residual magnitude/rate/bounds, and early termination.
- RSI: selects a reference phase and initializes wrist, leaders/followers, and
  rigid tuna pose/velocity near that state.

The task reuses REGRIND observation delay/noise, mass/friction/actuator
randomization, pushes, and gravity curriculum. It does not add tactile input,
tuna rotational symmetry, a new network, or a new RL algorithm.

The PPO config is `config/revo2_floating/agents/rsl_rl_ppo_cfg.py` relative to
the task package. Baseline settings include 24 steps per environment, the
`[1024,512,256,128]` ELU actor/critic, five epochs, four mini-batches,
`learning_rate=1e-3`, `gamma=0.998`, `lambda=0.95`, clip `0.2`, and entropy
coefficient `0.002`.

## Reference and placement

The default sequence/reference is selected by `scripts/_common.sh`; when
present, its stable reference is preferred. The current primary sequence is
`20200709_143747_left`.

Training can translate the can and the complete object/wrist reference together
within the configured strict-IK XY region. Observations subtract that placement
offset, preserving the policy's canonical coordinate contract while simulation
and rollout files retain world coordinates. Can yaw and Z are not randomized by
this placement term. Deterministic play disables RSI and reset perturbations;
use `--random-placement` to test placement generalization explicitly.

## Train and evaluate

Smoke test:

```bash
./scripts/rl.sh train \
  --sequence 20200709_143747_left \
  --num_envs 16 --max_iterations 2 --headless \
  --logger tensorboard --run_name floating_smoke
```

Full training example:

```bash
./scripts/rl.sh train \
  --sequence 20200709_143747_left \
  --full --num_envs 4096 --max_iterations 1000 --headless \
  --logger tensorboard --run_name floating_full_1000
```

Reference-only validation:

```bash
./scripts/rl.sh zero --sequence 20200709_143747_left --gui --real_time
```

Evaluate and export environment 0:

```bash
./scripts/rl.sh play \
  --sequence 20200709_143747_left \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_ITERATION.pt \
  --rollout-path outputs/floating/20200709_143747_left/rollout.h5 \
  --real_time
```

Add `--headless` for export without a viewer or `--random-placement` for the
configured XY sampling. An episode failure ends the exported rollout rather
than joining states across an automatic reset. Rollout quaternions are `xyzw`
and the file includes wrist, Revo2, object, action/reference, phase, and MANO21
data.

Training logs are written below `logs/rsl_rl/floating_revo2_tuna/`. A valid
smoke run has finite observations/rewards, RSI resets, a PPO learning iteration,
loss output, and a generated checkpoint.

## Convert a rollout to RB3

Offline conversion rigidly aligns object, wrist, and MANO together, then solves
strict IK with previous-frame warm starts and joint limits:

```bash
./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/20200709_143747_left/rollout.h5 \
  --out outputs/floating/20200709_143747_left/reference_12dof.h5
```

For a settled/upright trajectory, the existing options include
`--drop-leading-frames`, `--level-object-on-table`, and `--object-start X Y Z`.
If an orientation is specified, use `--object-quat X Y Z W`; the transform is
still applied to object, wrist, and MANO together.

Accept the conversion only when all frames report finite, in-limit solutions,
no failed indices, and pose errors within the configured tolerances. View the
result through [ISAAC_SIM_REPLAY.md](ISAAC_SIM_REPLAY.md).

## Online RB3 deployment

This path observes the current simulation state, evaluates the same floating
actor, solves the wrist command with warm-started bounded IK, and applies Revo2
leader/mimic targets:

```bash
./scripts/rl.sh play-arm \
  --sequence 20200709_143747_left \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_ITERATION.pt \
  --num_envs 1 --real_time
```

Use `--headless --eval_episodes N` for repeated evaluation. To record arm target
and measured telemetry, add
`--arm-tracking-path outputs/diagnostics/rb3_arm_tracking.npz`, then run:

```bash
python3 -m tools.arm_diagnostics.analyze_arm_tracking \
  outputs/diagnostics/rb3_arm_tracking.npz
```

Runtime gain comparisons use the existing `--rb3-stiffness-scale`,
`--rb3-damping-scale`, and `--rb3-effort-scale` options. Treat them as simulator
experiments, not real-hardware settings.

Online reset order matters: Isaac Lab resets actions **before** commands.
`RB3WristIKAction.reset` therefore only invalidates action state;
`RB3Revo2ReferenceCommand` calls `reset_from_reference` after selecting and
writing the new RSI frame/object placement. That synchronizes measured arm
joints, the IK warm start, and interpolation targets before observations.
Solving IK in the earlier action reset uses the previous episode's pose.
The online task interpolates each 30 Hz IK target over four 120 Hz substeps.

Run the simulator regression for asynchronous RSI and random placement:

```bash
./scripts/rl.sh debug --sequence 20200709_143747_left \
  --task Regrind-RB3-Revo2-TunaCan-Online-Play-v0 \
  --num_envs 2 --check_online_reset --max_steps 60
```

Each `[reset check]` must pass. For policy evaluation, `[arm episode env=0]`
reports lift from that episode's start to its last observed pre-reset sample;
it is not the terminal sample because Isaac autoresets before returning.
The REGRIND `success` term means reference completion without deviation, not
a separate measured grasp/contact classifier.

Measured on 2026-09-06 with the unchanged floating checkpoint
`2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`:
fixed placement X=0.40/Y=0 passed 10/10 episodes after the reset fix (3/5
before, with the same interpolation); randomized X=[0.40,0.50],
Y=[-0.20,0.20] passed 19/20, with no reported nonterminal IK failures.
This finite evaluation does not guarantee all placements. Arm tracking lag
and contact dynamics still differ from floating-hand impedance control.

The floating checkpoint can be transfer-fine-tuned against the assembled arm
dynamics without changing its 67-D observation or 12-D wrist/finger action
contract. The transfer task keeps the public PPO architecture and loss, lowers
only the optimizer step to `1e-4`, and writes separate logs below
`logs/rsl_rl/rb3_revo2_tuna_transfer/`. Start with the deterministic
16-environment, full-gravity task:

```bash
./scripts/rl.sh train-arm \
  --sequence 20200709_143747_left \
  --resume \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_ITERATION.pt \
  --num_envs 16 --max_iterations 100 \
  --run_name rb3_transfer --headless
```

An explicit existing checkpoint path is accepted for cross-experiment resume.
Transfer training is experimental: the short 25-update trials before the reset
fix did not improve repeated evaluation. Use the original floating checkpoint
for the verified deployment above; do not select those trials merely because
their iteration number is larger. Additional training is not needed for the
reset correction.
Use `--full` only after the smoke transfer is stable; strict IK is CPU-bound, so
increase `--num_envs` from 16 deliberately instead of assuming the floating
hand's 4096-environment throughput.

`scripts/random_can_full_replay.sh` chains random placement, floating play,
offline IK, and workcell replay. Its built-in checkpoint is dated, so pass
`--checkpoint` explicitly when model identity matters.
