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

`scripts/rl.sh play-arm` now uses the user-approved video controller and compliant
distal contacts. `train-arm` shares that configuration; floating train/play/zero
remain unchanged. The previous strict-IK controller is available explicitly with
`--arm-controller baseline`. Start with the
[current commands](#approved-video-controller-and-timed-transfer); historical
experiments remain in the [diagnostic index](../scripts/README.md#experiments-and-failure-reproduction-opt-in).

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

`--max_iterations N` is the number of updates in this invocation, including on
resume; it is not an absolute checkpoint-iteration ceiling. The full floating
config defaults to 20,000 updates, so specify this flag for a bounded run.
Watch logs with `tensorboard --logdir logs/rsl_rl/floating_revo2_tuna` in an
environment with TensorBoard installed. W&B uses `--logger wandb --log_project_name NAME`.
For a separate capture experiment use the Hydra override
`agent.experiment_name=presentation_capture`; the current repository's
`cli_args.update_rsl_rl_cfg` accepts but does not apply `--experiment_name`.

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

## Current evaluation checkpoint

The primary sequence (`20200709_143747_left`) now uses the completed 10,000-update
floating policy by default:

```text
logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt
```

`scripts/_common.sh` owns the pinned selection (`DEFAULT_FLOATING_CHECKPOINT`).
It is used by `rl.sh play|play-arm`, `play_arm_candidate.sh`, both comparison
capture scripts, and `random_can_full_replay.sh`. Override via
`REGRIND_FLOATING_CHECKPOINT` or the launcher's explicit checkpoint argument.
`rl.sh` preserves explicit task/agent/run/model selection, other sequences and
legacy-arm selection; it does not inject this checkpoint into training or zero
agents. Missing default files fail rather than selecting an unrelated latest run.
Direct Python evaluators still require their own model selection.

This changes model selection only: controller presets, IK, reference, phase,
normalization, gravity and training schedules are unchanged. Historical 5,000-update
reports/media remain historical evidence, not validation of this new checkpoint.
The new model has loaded in floating and arm GUIs; a fresh matched 20-placement
comparison has not been performed. Existing rollout/reference/video files are
not regenerated automatically. Checkpoints are ignored by Git and must be copied
separately to another machine, or selected with an explicit path.

Default-selection smoke on 2026-09-08: run `bash scripts/rl.sh play` and
`bash scripts/rl.sh play-arm`, each with
`--headless --num_envs 1 --eval_episodes 1 --max_steps 150`, without a checkpoint
argument. Both loaded `model_9999.pt`, exited normally and reported one successful
episode under the existing criterion. Arm IK reported no failed steps; wrist
position error mean/max was 18.05/41.04 mm (not a precision-controller claim).
Logs: `outputs/diagnostics/checkpoint_10000_defaults_kGxp0i/{floating,arm}.log`
(local, ignored). These are smoke runs, not a paired grasp-performance study.
Shell checks and all 120 root regression tests passed, including six launcher
tests for defaults, overrides, missing checkpoints and capture forwarding.

## Online RB3 deployment

Current default commands are in the [shared video controller section](#approved-video-controller-and-timed-transfer).
The timing, telemetry flags and reset details immediately below describe the
**preserved strict-IK baseline**, now selected explicitly:

This path observes the current simulation state, evaluates the same floating
actor, solves the wrist command with warm-started bounded IK, and applies Revo2
leader/mimic targets:

```bash
./scripts/rl.sh play-arm --arm-controller baseline \
  --sequence 20200709_143747_left \
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

This timing describes the **original `--arm-controller baseline` path**. The
`play_arm_candidate.sh` transfer configuration instead shapes the wrist target
and solves IK at 120 Hz; policy/reference remain 30 Hz. Its
`rb3_smooth_bounded_ik.json` variant permits explicitly logged approximate IK
near singularities. See [candidate comparison](ARM_IK_SINGULARITY_FIX.md).
`--real_time` paces a run only when computation is faster than the simulated
period; it cannot guarantee 1× wall-clock speed or zero physical tracking error.

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

## Opt-in last-phalanx rubber approximation

`config/experiments/revo2_rubber_contact.json` is an **uncalibrated**, opt-in
contact-response proxy. It covers all five distal rigid links and their fixed
touch children (10 existing collision shapes); proximal links/palm/arm stay
unchanged. It does not simulate a visibly deforming pad, measured Shore hardness,
pad thickness, contact-area growth or tangential rubber deformation.

The first comparison deliberately retains static/dynamic friction **0.8/0.8**,
and adds a force-unit normal contact spring **10,000 N/m**, damping **10 N·s/m**
(per contact response, not a measured whole-pad stiffness).
These are trial values, not Revo2 specifications. Contact spring values are not
motor gains. Existing geometry, mass/inertia, mimic joints, actuator gains/limits,
gravity, timing, policy and success criteria are preserved.

`regrind/utils/revo2_contact_material.py` wraps the existing USD spawn operation,
binds material opinions before environment cloning, and never saves source USDs.
It verifies resolved USD bindings/parameters in every environment and reads
PhysX material triples. A negative runtime restitution field encodes compliant
spring stiffness; it is not negative bounce
([PhysX material API](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxMaterial.html)).
Runtime damping/force-mode verification is limited to authored USD settings;
there is no calibrated force–indentation test or independent damping sensor.

Use the maintained evaluator for both embodiments, including GUI. From the root:

```bash
FLOATING=logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt
CONTACT=config/experiments/revo2_rubber_contact.json
STATES=outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl

# Floating policy, compliant fingertips; choose a NEW output directory.
./scripts/evaluate_mounted_interface.sh --mode floating --episodes 20 \
  --checkpoint "$FLOATING" --states "$STATES" --visualizer kit \
  --fingertip-contact-config "$CONTACT" --output outputs/diagnostics/rubber_floating_view

# Same original policy with the preserved arm controller.
./scripts/evaluate_mounted_interface.sh --mode legacy --transfer-evaluation --episodes 20 \
  --checkpoint "$FLOATING" --states "$STATES" --visualizer kit \
  --fingertip-contact-config "$CONTACT" --output outputs/diagnostics/rubber_arm_view

# Separate arm fine-tuning, not a new floating training run or a second policy.
./scripts/rl.sh train-arm --arm-controller baseline --transfer-init "$FLOATING" --num_envs 16 \
  --max_iterations 100 --run_name rubber_transfer_100 --headless \
  --fingertip-contact-config "$CONTACT"
```

For quantitative evaluation replace `--visualizer kit` with `--headless`.
Evaluate a new rubber transfer checkpoint by replacing only `--checkpoint` in
the arm command; keep the contact JSON. Resume uses the existing
`train-arm --arm-controller baseline --resume --checkpoint TRANSFER.pt` with the same contact flag.
Contact settings/hash become part of the transfer controller contract, so an
incompatible transfer resume/evaluation is rejected. Omit the contact flag with
the **baseline** selector to retain the original hard-contact paths; the video
controller defaults to compliant contacts. Contact training is currently limited to the
deterministic arm transfer path; it intentionally rejects ordinary floating
training, where startup material randomization would need a separate audit.

The material-only comparison reuses the existing analyzer:
`scripts/analyze_transfer_recovery.sh NEW_OUTPUT --transfer-before HARD_RUN
--transfer-after SOFT_RUN --material-only`. It requires the same checkpoint,
embodiment, measured initial states and all non-material conditions. Without
`--material-only`, it compares policies under identical physics as before.

Training from scratch is not inherently more physically accurate: fine-tuning
also updates the policy under the new dynamics. Both remain dependent on the
quality of the contact approximation and require held-out physical validation.

### Executed rubber-contact comparison — 2026-09-09

Result: **do not promote the rubber transfer checkpoint**. Same original 10k
floating policy, reference and measured initial states; old20 and heldout20 are
the two banks listed in the transfer-validation section below. The hard arm
baseline reuses the previously completed `arm_transfer_20260909/before_*20`
runs after exact condition checks; floating hard baselines were rerun.
All new results live in `outputs/diagnostics/revo2_rubber_20260909/`.

| Embodiment / policy | old20 task / hold | heldout20 task / hold | Total task / hold |
|---|---:|---:|---:|
| Floating, original contacts and policy | 20 / 20 | 20 / 20 | 40 / 40 |
| Floating, compliant contacts, original policy | 20 / 20 | 20 / 20 | 40 / 40 |
| Arm, original contacts and policy | 20 / 19 | 15 / 14 | 35 / 33 |
| Arm, compliant contacts, original policy | 20 / 18 | 17 / 16 | 37 / 34 |
| Arm, compliant contacts, 100-update transfer | 16 / 14 | 13 / 11 | 29 / 25 |

Each bank contains 20 placements; these counts are not general success-rate
estimates. `task` is the unchanged termination success flag. `hold` is a separate
diagnostic: final 0.2 s minimum can rise >=0.1 m, with can–robot contact >0.01 N
for >=80% of those samples. It does not establish a longer stable hold.

Under unchanged compliant physics, fine-tuning reduced placement-averaged wrist
error **23.24→22.42 mm** and leader error **0.02666→0.02074 rad**, but increased
object-keypoint error **22.20→24.05 mm** and reduced hold **34→25/40**.
New task failures: old20 `[14,16,17,19]`, heldout20 `[2,7,8,16,17]`;
heldout20 `[6]` recovered. All task failures had `object_deviation` set.
Hold regressions: old20 `[4,13,14,16,17]`, heldout20 `[2,4,7,16,17,18]`;
hold recoveries: old20 `[15]`, heldout20 `[6]`. Indices are zero-based.
Lower tracking errors alone therefore do not demonstrate better grasping.
This short, fixed-placement transfer does not distinguish insufficient learning,
overfitting or an unsuitable uncalibrated contact approximation; none is proven.

The separate checkpoint is
`logs/rsl_rl/rb3_revo2_tuna_transfer/2026-09-09_19-33-22_rubber_transfer_100/model_99.pt`.
It was initialized from the original floating checkpoint, not from the earlier
hard-contact transfer. The run completed 100 updates, 38,400 transitions,
2,000 optimizer steps and 1,050 resets in **1,646.95 s** with 16 envs.
IK took 1,469.21 s over 39,450 calls. Maximum actor-weight change was 0.002824;
initial deterministic-action difference was zero. Initial LR was 1e-4 and the
unchanged adaptive schedule used 1e-5. Finite checks and source SHA256 preservation
passed; checkpoint reload and frozen evaluation passed on both banks.

Evidence: `comparison_summary.json/.png` aggregate the results;
`contact_{floating,arm}_{old20,heldout20}/transfer_comparison.json/.png` compare
materials with the same policy; `transfer_{old20,heldout20}/` compare policies
with the same compliant physics. Their source run directories retain per-step
physics, contact, action, initial-state and terminal records. Training evidence
is `transfer_summary.json`, `transfer_updates.jsonl` and `contact_verification.json`
inside the separate run. Compliant bindings were verified across all 16 training
envs; subsequent 1-env evaluations also read exactly 10 compliant shapes from PhysX.
Runtime material triples encode stiffness -10000; damping/force mode remain
USD-verified settings rather than independent measurements.

All 134 regression tests passed. With the contact option omitted, one floating
and one arm rollout reproduced 152 baseline physics samples **exactly** (joint,
wrist, can, action, targets and timestamps); see `default_regression.json`.
Source USDs were not saved or modified. Tiny simulated hand position-limit
overshoots remain (maximum 0.00192 rad across comparisons); no verified
drive-only torque signal is available, so saturation remains UNKNOWN.
No fresh floating training, long transfer, material sweep or hardware validation
was performed. Existing play/train defaults remain unchanged.

## Opt-in arm fine-tuning

### Approved video controller and timed transfer

On 2026-09-09 the user authorized making the matched video controller plus the
last-phalanx rubber approximation the mounted default. Original 10k policy under
these settings passed the task criterion and lift/contact proxy on both saved
20-placement banks (40/40 observed, not a general success-rate guarantee).
The older strict-IK transfer results below remain historical and are **not**
evidence for this controller's fine-tuned policy.

`regrind/utils/arm_execution_config.py` is shared by training and evaluation.
It reuses `SimpleMountedWrist`, the `c3` gains in
`config/experiments/rb3_precision_candidates.json`, the response parameters in
`config/experiments/rb3_transfer_recovery_candidate.json`, and
`config/experiments/revo2_rubber_contact.json`. Arm Kp is
`[700,20000,12000,900,2400,250]`, Kd `[35,260,140,45,70,20]`;
effort/velocity limits are unchanged. There is a causal 0.1 s wrist-target
response, warm-first IK and velocity-path targets at 120 Hz, while policy/phase
stay 30 Hz. This is **not** the later acceleration-bounded IK candidate. Runtime
controller/material/reference hashes are checked when loading transfer models.

The existing controller supports **num_envs=1 only**. Video transfer rejects RSI
and starts at frame zero, fixed placement, full gravity, randomization OFF.
It uses the existing PPO/network and 67/94 observation, 12-action contracts.
Fresh transfer loads the explicit floating model and normalizers, starts a new
optimizer at `1e-4` (existing adaptive schedule may reduce it), and updates the
policy and normalizers. Evaluation freezes both. No recorded-action replay or
per-step state overwrite is used. One-env rollout batches are small/correlated;
an hour of training does not establish convergence or better grasp reliability.

```bash
FLOATING=logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt

# Current default GUI, original 10k policy, saved 20-placement bank (not fresh XY samples).
./scripts/rl.sh play-arm --episodes 20
# Previous strict-IK controller, original contact settings/flags.
./scripts/rl.sh play-arm --arm-controller baseline --num_envs 1 --real_time

# Optional video/rubber transfer, one-hour collection/update budget.
./scripts/rl.sh train-arm --transfer-init "$FLOATING" --num_envs 1 \
  --training-seconds 3600 --max_iterations 100000 --run_name video_rubber_hour --headless

# Resume THIS controller's transfer; preserve optimizer, model, normalizer and counters.
TRANSFER=logs/rsl_rl/rb3_revo2_tuna_transfer_video/YOUR_RUN/model_N.pt
./scripts/rl.sh train-arm --resume --checkpoint "$TRANSFER" --num_envs 1 \
  --training-seconds 3600 --max_iterations 100000 --run_name video_rubber_resume --headless
./scripts/rl.sh play-arm --checkpoint "$TRANSFER" --episodes 20

# Matched bank: run separately with FLOATING and TRANSFER, new output directories.
STATES=outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl
./scripts/evaluate_mounted_interface.sh --mode simple --arm-controller video \
  --transfer-evaluation --checkpoint "$FLOATING" --states "$STATES" \
  --episodes 20 --headless --output outputs/diagnostics/MY_VIDEO_TRANSFER/before
./scripts/evaluate_mounted_interface.sh --mode simple --arm-controller video \
  --transfer-evaluation --checkpoint "$TRANSFER" --states "$STATES" \
  --episodes 20 --headless --output outputs/diagnostics/MY_VIDEO_TRANSFER/after
./scripts/analyze_transfer_recovery.sh outputs/diagnostics/MY_VIDEO_TRANSFER/comparison \
  --transfer-before outputs/diagnostics/MY_VIDEO_TRANSFER/before \
  --transfer-after outputs/diagnostics/MY_VIDEO_TRANSFER/after
```

Repeat the paired commands with the distinct bank
`outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl`, using
new output directories. The launcher defaults to this bank and validates its
reference. Use `--states` for an explicit matching bank. Baseline-only flags
(`--arm-tracking-path`, gain multipliers, Hydra overrides) require the baseline
selector; current evaluation writes `metadata.json`, `physics.jsonl`, `policy.json`.

The time budget excludes simulator/model startup and is checked **after a complete
PPO update and logging**, then saves the final checkpoint. `max_iterations` is an
additional upper bound; the runner's iteration-based ETA is not the time budget.
Logs live under `logs/rsl_rl/rb3_revo2_tuna_transfer_video/`; floating model selection
never scans this directory. Existing 16-env strict-IK transfer runs must use
`--arm-controller baseline` for continuation, not the video selector.

#### One-hour execution — 2026-09-09

Run `logs/rsl_rl/rb3_revo2_tuna_transfer_video/2026-09-09_20-43-36_video_rubber_hour/`
completed **3,601.07 s**, **2,679 updates**, **64,296 transitions**, **53,580 optimizer
steps** and **1,741 resets** in one environment. Final checkpoint: `model_2678.pt`.
IK measured 258,925 calls / 374.36 s. Maximum actor parameter change was 0.11608;
finite checks passed and the original floating checkpoint hash was unchanged.
Initial deterministic-action difference was exactly zero; configured LR `1e-4`
became `1e-5` under the unchanged adaptive schedule. The 100,000-update argument
was only a ceiling; the elapsed-time stop executed after the final complete update.

Artifacts: `outputs/diagnostics/video_transfer_hour_20260909/`. `train_hour.log`
records the complete run; `before_*20` / `after_*20` and `comparison_*20` use the
same two pre-existing state banks and shared video/rubber controller. Final-model
evaluation does not update the model/normalizers. No intermediate checkpoint was
selected using these placements.

**Result: retain the original 10k policy as the default model.** Both saved banks
passed actual-state/controller parity checks (state tolerance `2e-6`, not just a
shared seed). Existing task success and the separate lift/hold proxy were each
20/20 before and after in old20, and 20/20 before and after in heldout20. Neither
bank had success→failure or failure→success changes, and neither had an IK failure.
No termination-failure flag accompanied success. This is no observed grasp-rate
improvement; the following tracking metrics worsened.

| All 5,928 physics samples per policy, both banks | Original 10k | + one hour |
|---|---:|---:|
| Task success / lift-hold proxy | 40/40 / 40/40 | 40/40 / 40/40 |
| Object keypoint error mean / P95 / max [mm] | 3.94 / 9.79 / 29.76 | 11.16 / 24.01 / 40.84 |
| Desired→actual wrist position mean / P95 / max [mm] | 26.26 / 40.36 / 43.63 | 26.46 / 42.86 / 48.19 |
| Wrist rotation mean / P95 / max [rad] | .07119 / .12285 / .15070 | .07335 / .13938 / .25428 |
| Finger target→actual mean / P95 / max [rad] | .02235 / .08796 / .12580 | .02880 / .11181 / .12943 |
| Mean final can rise [mm] | 227.38 | 221.52 |

The wrist metric includes the approved causal response delay relative to the
decoded policy target; it is not a static-precision metric or the floating-versus-
mounted actual-pose difference. Both policies use the same timestamps/definition.
After contact, episode-averaged object error increased 5.43→15.73 mm and finger
error .02569→.03416 rad. Largest observed arm acceleration changed 927.9→880.1
rad/s², but wrist2's maximum increased 111.6→182.6 rad/s²: not a general smoothing
improvement. Wrist1/wrist3 still reached the configured 10 rad/s boundary (maximum
numerical overrun after training .00043 rad/s). No arm position-limit violation
was recorded. Small hand limit overruns remained: index proximal .00120 rad before;
index distal .000186 rad after. Implicit solver drive-effort saturation is UNKNOWN.

Paired per-placement outcomes, pre/post-contact metrics, and PNG plots are in
`comparison_old20/transfer_comparison.{json,png}` and
`comparison_heldout20/transfer_comparison.{json,png}` beneath the artifact directory.
Training was fixed-placement/full-gravity/no-randomization with one environment;
these 40 placements and one hour do not establish convergence or broad robustness.
The weaker tracking does not identify a unique cause (e.g. overfitting versus
short-run optimization/normalizer drift); no further tuning was performed.

To view the retained default: `./scripts/rl.sh play-arm --episodes 20`.
To view this experimental policy explicitly:

```bash
./scripts/rl.sh play-arm --episodes 20 --checkpoint \
  logs/rsl_rl/rb3_revo2_tuna_transfer_video/2026-09-09_20-43-36_video_rubber_hour/model_2678.pt
```

Validation: short timed training, checkpoint reload, two-update resume and the
original floating-policy one-episode smoke all completed. Default arm execution
matched the prior approved run for all 152 physics samples. Root regressions:
142 passed. An initial direct-training smoke failed to resolve `tools.*`; the
shared configuration now resolves the repository root before importing the existing
IK module. The preserved failed log is `train_smoke.log`; successful rerun is
`train_smoke2.log`. No controller/IK algorithm was replaced to fix that import.

### Preserved strict-IK transfer contract

The remainder of this section documents the original `--arm-controller baseline`
implementation and its completed experiments, not the new mounted default.

`train-arm` reuses the existing Online task and floating observation/action,
reward and PPO implementations. It updates the original policy, not a second
residual policy. The default floating train/play/zero commands are unchanged.
Transfer logs/checkpoints live only under `logs/rsl_rl/rb3_revo2_tuna_transfer/`;
they are not candidates for floating checkpoint auto-selection.

### Fixed execution contract

Training and `evaluate_mounted_interface.sh --transfer-evaluation` share
`configure_transfer_baseline()` in `config/rb3_revo2/rb3_revo2_online_env_cfg.py`
(relative to the task package). They use the original `RB3WristIKAction`: strict
IK once per 30 Hz policy action, four position-interpolation substeps at 120 Hz
physics, zero arm velocity targets. This is the preserved baseline, **not** the
experimental 120 Hz smooth-IK/precision-gain controller. Neither is promoted
based on recency. The zero-velocity baseline preserves the prior contact-policy
result (19/20 versus 7/20 with `v_path` in the
[historical paired test](ARM_VELOCITY_CONTACT_POLICY_VALIDATION.md)); minimizing
wrist error alone was not a sufficient selection criterion.
`controller_contract` records runtime gains/limits, joint
names, timestep, scales, IK settings and reference hash; a transfer checkpoint
with a different contract is rejected by the paired evaluator.

Arm joint order is base/shoulder/elbow/wrist1/wrist2/wrist3. Unchanged baseline
Kp=`[300,500,500,300,200,50]`, Kd=`[20,20,20,20,20,10]`, effort limits
`[10,100,100,100,100,10]`, velocity limits=`[10,10,10,10,10,10]`.
These are simulation settings, not manufacturer specifications. Implicit drive
torque saturation remains UNKNOWN; joint speed/position proximity is observable.
Mounted-hand inertia, mimic drives, gravity and dynamic can contacts remain.

Actor order (67): object position3, object rot6d6, actual wrist position history6,
actual wrist rot6d history12, default-relative six-leader history12, previous
action12, phase1, reference wrist position/rot6d9, reference leaders6. Critic94
adds object linear/angular velocity6, five physical fingertip positions15 and
leader velocity6. Positions remove env origin and the shared placement offset;
runtime quaternions are xyzw. Actual mounted body/joint states feed the existing
observation functions; RB3 angles never substitute for wrist pose. As in floating,
object linear velocity observations are COM
world velocities, whereas reset writes link-origin twist and is checked against
the matching link-origin API. This is not a reward/observation-frame change.
Policy normalization
and action distribution are loaded with actor/critic state. Wrist residuals are
clipped to [-1,1], scaled by 1/30 m and 3.2/30 rad, then combined with the
reference once; hand residuals use 3.2/30 rad and the existing mimic mapping.

Default transfer: fixed placement, full gravity, dynamics/observation/placement
randomization OFF, RSI OFF. Optional RSI uses
`env.commands.reference.rsi_enabled=true`; after reference selection/reset, the
arm hook synchronizes IK warm start, interpolation targets, raw/processed action
buffers and leader target cache. Selected reference arm velocities are restored
for RSI; frame-zero evaluation preserves the saved bank's zero arm velocity.
Indexed resets leave other environments' buffers unchanged. Existing observation
history reset is retained; latency is disabled for this initial experiment.

### Commands

Run from the repository root. Use an explicit successful floating checkpoint:

```bash
FLOATING=logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt

# Fresh transfer: model/normalizer/distribution loaded, optimizer NEW, iteration 0.
./scripts/rl.sh train-arm --arm-controller baseline \
  --transfer-init "$FLOATING" --num_envs 16 --max_iterations 100 \
  --seed 42 --run_name transfer_initial_100 --headless

# Resume a TRANSFER checkpoint: restore optimizer/LR/next iteration/training counters.
TRANSFER=logs/rsl_rl/rb3_revo2_tuna_transfer/YOUR_RUN/model_99.pt
./scripts/rl.sh train-arm --arm-controller baseline --resume --checkpoint "$TRANSFER" \
  --num_envs 16 --max_iterations 100 --run_name transfer_resume --headless

# Longer experiment, not executed automatically: 1,000 ADDITIONAL updates.
./scripts/rl.sh train-arm --arm-controller baseline --resume --checkpoint "$TRANSFER" \
  --num_envs 16 --max_iterations 1000 --run_name transfer_long --headless

# Actual RSI/reset contract checks without training.
./scripts/rl.sh debug --task Regrind-RB3-Revo2-TunaCan-Online-Smoke-v0 \
  --num_envs 2 --check_online_reset --max_steps 60
```

Do not use `--resume` for the first transfer from a floating checkpoint. Fresh
initialization uses LR `1e-4`; the unchanged adaptive PPO schedule can reduce it
(observed `1e-5`). The runtime LR is printed and logged. Training updates the
loaded normalization statistics; evaluation freezes both weights and statistics.
Initialization checks exact deterministic actions and all loaded actor/critic
buffers, plus source observation order/scales when `params/env.yaml` exists.
Resume starts a new physical episode: optimizer/RNG/counters are restored, but
hidden PhysX contacts and an in-progress rollout are not serialized. It is not
claimed bitwise equivalent to uninterrupted training. The resumed segment has
a new timestamped log directory linked to the source checkpoint; parent files
are not overwritten. Keep the deterministic
16-env task; do not use `--full` or scale to 4096 without measuring CPU IK cost.

### Paired arm evaluation

Use the same saved actual state bank, controller and explicit reference metadata
for both policies; each policy acts on its own measured rollout. The evaluator
checks actual reset states, freezes policy/normalizers, and records terminal
states **before** autoreset. It never substitutes a floating action recording.

```bash
BANK=outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl
./scripts/evaluate_mounted_interface.sh --mode legacy --transfer-evaluation \
  --checkpoint "$FLOATING" --states "$BANK" --episodes 20 --headless \
  --output outputs/diagnostics/MY_TRANSFER/before
./scripts/evaluate_mounted_interface.sh --mode legacy --transfer-evaluation \
  --checkpoint "$TRANSFER" --states "$BANK" --episodes 20 --headless \
  --output outputs/diagnostics/MY_TRANSFER/after
./scripts/analyze_transfer_recovery.sh outputs/diagnostics/MY_TRANSFER/comparison \
  --transfer-before outputs/diagnostics/MY_TRANSFER/before \
  --transfer-after outputs/diagnostics/MY_TRANSFER/after
```

Output directories must be new. Repeat with the pre-existing held-out bank
`outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl` and
different output directories. For the viewer, remove `--headless` and add
`--visualizer kit` (required by the installed Isaac Lab 3.0 launcher). Task success
remains the existing reference-end criterion; report measured lift/contact/hold
proxies and drop behavior separately, not as a replacement success definition.
Twenty placements or 100 updates do not establish convergence/general success.

`transfer_initialization.json`, per-update `transfer_updates.jsonl` and
`transfer_summary.json` beside checkpoints record load parity, finite checks,
gradient/weight changes, resets, transitions and measured IK/wall time. Standard
PPO losses/rewards remain in TensorBoard (`tensorboard --logdir
logs/rsl_rl/rb3_revo2_tuna_transfer`). Evaluation saves `metadata.json`,
`physics.jsonl`, policy observations/actions and contact-split comparison plots.
Historical pre-reset 25-update transfer trials are not this experiment and are
not recommended checkpoints.

### Executed transfer validation — 2026-09-09

This is an opt-in candidate; neither floating nor arm play defaults were replaced.
Source: the explicit 10,000-update checkpoint in the commands above, SHA-256
`a8e0b36efef8884953a8e05694925dbda2ae035ae88e32d98dca6848226bd906` (unchanged).
Reference: `outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5`,
38 frames, SHA-256 `8b8de4be08db8b466ebb5b16e817dea78081c3c458e29b0f020c888105510796`.

Final checkpoint, selected by the predeclared 100-update budget, not evaluation:
`logs/rsl_rl/rb3_revo2_tuna_transfer/2026-09-09_17-38-37_transfer_initial_100/model_99.pt`.
Training seed42, 16 environments, 38,400 transitions, 1,591.04 s collection/update
wall time (~24.1 transitions/s), 39,438 IK calls including resets, 1,426.73 s in
IK. Other diagnostic processes briefly shared the machine, so this is measured
run throughput, not an isolated hardware benchmark. No 4096-env test was run.
All 100 updates had finite observations/rewards/losses/gradients, with 2,000
optimizer steps and 1,038 environment resets. Maximum actor parameter change
from initialization was 0.00282649. Actor/critic normalizer counts both advanced
from 983,040,000 to 983,078,400. Initial LR1e-4 fell to1e-5 under the existing
adaptive schedule; no extra tuning was performed.

Same-state, same-controller evaluation (episode-equal means; task flags unchanged):

| Metric | Existing20 before → after | Held-out20 before → after |
|---|---|---|
| Existing task success | 20 → 20 | 15 → 20 |
| Lift/contact hold proxy | 19 → 20 | 14 → 19 |
| Mean wrist position error [mm] | 24.077 → 24.028 | 22.817 → 22.837 |
| Mean wrist rotation error [rad] | 0.1417 → 0.1332 | 0.1695 → 0.1773 |
| Mean leader target/actual error [rad] | 0.02866 → 0.03706 | 0.02621 → 0.03417 |
| Mean object keypoint error [mm] | 20.572 → 18.644 | 23.731 → 19.963 |

Initial arm/hand/follower/object states, velocities, phase and controller contracts
were numerically checked, not inferred from seed. Each bank contains 20 unique XY
placements; nearest cross-bank separation is 8.13 mm. All four evaluations verified
frozen policy/normalizers and observation/action parity on every policy step.
Both post-training evaluations had zero recorded IK-failed physics samples and
zero arm position-limit violation. Small finger limit overshoot remained (up to
0.00202 rad, index proximal). Wrist1 speed was within 5% of its limit for an
episode-averaged 2.50% / 1.89% of samples after training (existing/held-out banks).
This is limit proximity, **not** verified drive torque saturation.

Held-out task failures4,6,8,14,15 became successes; there were no task-flag
success→failure transitions. However, held-out16 became a **physical drop**:
can rise peaked at 0.193 m, ended at 0.041 m, and final can–robot contact was0 N.
Its unchanged termination logic emitted both `success` and `object_deviation`.
The hold proxy therefore changed success→failure for16; it changed failure→success
for4,6,8,14,15,17. Existing-bank18 changed drop/proxy failure→hold, and its prior
simultaneous success/failure flags disappeared. Thus 40/40 task flags must not be
reported as 40/40 reliable grasps. The physical proxy improved33/40→39/40, while
finger tracking error increased and wrist position tracking was nearly unchanged.
No long post-reference hold, convergence, or general placement reliability is proven.

Artifacts under `outputs/diagnostics/arm_transfer_20260909/`:

- `floating_before.log`, `floating_after.log`: original floating smoke, success1/1
  each; `contract_1env.log`, `smoke_16env.log`: 1-env and 16-env PPO checks.
- `resume_smoke.log`: model1 resumed at iteration2 with optimizer LR1e-5,
  saved model2; exact deterministic load-action difference0. The final100-update
  model was also reloaded successfully by both actual evaluators.
- `rsi_history_final.log`: 2-env/60-step RSI, leader/follower/arm/object pose and
  velocity, action/interpolation buffers and reset history checks. Regression
  tests also verify reset isolation from untouched environments.
- `train_100.log`; detailed per-update and initialization records beside model99.
- `before_old20/`, `after_old20/`, `before_heldout20/`, `after_heldout20/`:
  raw physics/observation/action/terminal evidence. Contact-split wrist/hand/object
  metrics, per-joint limit proximity and residual magnitudes are in
  `comparison_old20/transfer_comparison.json` and
  `comparison_heldout20/transfer_comparison.json`, each with a PNG plot.
- `regression_handoff.log`: 127 simulator-independent tests passed. Shell syntax,
  launcher help, focused diff and `git diff --check` also passed.

Preserved failed validation attempts: `rsi_reset_2env.log` rejected unsupported
`--headless` (debug is already headless unless its GUI option is used);
`rsi_reset_extended.log` compared COM velocity to link-origin reset velocity.
The latter was a diagnostic API error, corrected without changing physics or
observation/reward semantics; subsequent extended/history runs passed. An initial
test fixture lacked the new optional config field; it was updated with the
baseline `False` value without weakening its existing assertions.

`scripts/random_can_full_replay.sh` chains random placement, floating play,
offline IK, and workcell replay. It uses the shared current checkpoint and writes
to `outputs/floating/random_can_replay_10000` by default. Its existing frame trimming
and object-leveling options are unchanged; this offline path is not the online
mounted policy evaluation.
