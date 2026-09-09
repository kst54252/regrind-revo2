# Arm live execution performance (2026-09-07)

Status: actual policy execution accelerated, **1x wall-clock target NOT met**.
See [controller recovery](ARM_TRANSFER_RECOVERY.md) for unchanged gains, policy,
state bank, contact settings and the distinction between equilibrium and actual pose.

## Changes, all opt-in

- `scripts/play_arm_candidate.sh` selects the already verified transfer config and
  adds lightweight actual-policy viewing plus `--fast-ik`. No recorded actions,
  joint-state replay or prerecorded observations are used.
- `tools/rb3_revo2_ik/warm_start_ik.py` first solves from previous accepted q
  with the same SciPy bounded least-squares solver, tolerances and model. Accept
  only a successful, finite, bounded solution with max raw joint step <=.15 rad.
  Otherwise use the original full multi-seed IK. This shortcut does not prove
  global branch optimality; the recorded-input and live tests below validate the
  present trajectory, not arbitrary workspace motions.
- Its analytic residual Jacobian avoids repeated finite-difference FK. Model
  offsets, axes and mounted transform come from the existing verified model.
  A per-solve cache shares residual/Jacobian calculations. The original FK is
  still used for acceptance errors. Existing `inverse()` only gains an optional
  Jacobian argument; its default remains the original two-point derivative.
- `simple_mounted_interface.py` adds `fast_ik=False` by default. No changes to
  response tau, Kp/Kd, position/velocity/effort limits, hand coupling, phase,
  dt, gravity, contact, object dynamics or normal training/play defaults.
- Evaluator `--realtime-view` skips expensive diagnostic snapshots, repeated
  observation/action parity checks and JSON writes. It still runs frozen policy
  inference from actual observations, native environment updates and all physics
  steps. It sleeps only if ahead of the original control period; no steps are
  skipped and no trajectory timestamps are changed. Lightweight runs are excluded
  from the full diagnostic analyzer because `physics.jsonl` is intentionally empty.
- `realtime.json` records actual compute/paced wall time, simulated time, IK and
  component durations, plus fallback count. Component timings are inclusive
  (root_apply includes IK); do not sum nested values twice. Initialization is
  outside the loop timing, but episode resets remain included. Physics time may
  include asynchronous work completed at synchronization, so it is an observed
  wall-time bottleneck, not a GPU-kernel attribution.

## Executed comparisons

All rates below are measured headless except the explicitly labelled old GUI
capture. Ratios are simulated seconds / wall seconds; 1x means real-time speed.

| Run under outputs/diagnostics/arm_transfer_recovery/ | Mode | Rate | Mean IK | Success |
|---|---|---:|---:|---:|
| realtime_view_20260907_01 (prior turn) | Lightweight GUI, original IK | .155x | 36.57 ms | 3/3 |
| fast_ik_speed01 | Warm-first, numerical derivative | .471x | 5.55 ms | 3/3 |
| fast_jac_profile01 | Warm-first + analytic derivative | .606x | 1.77 ms | 3/3 |
| restored_baseline3 | Original IK, verified default workers | .172x | 35.91 ms | 3/3 |
| fast_ik_live20_v2 | Final fast IK, same workers | .632x | 1.91 ms | 20/20 |

`fast_ik_gui20_v1` was also launched with the Kit window verified visible. Its
observed rolling30-control-step rate is .455–.485x (IK about1.6–2.0ms), slower
than headless because rendering is included. At the original measurement handoff
the run was still active; this is not a current process-status assertion or a
completed20-episode GUI success claim.

Full20 final IK falls back31 times (including reset calls); fallback remains
expensive and does not guarantee an8.33ms deadline. PhysX step wall mean8.659ms
versus baseline8.785ms, already above the8.333ms physics time budget before the
other calculations. Thus this is about3.66x faster than the same-worker baseline,
but **not a real-time controller**. No claim of hardware schedulability.

`fast_ik_same_input01.json` tests all2,964 original live20 IK inputs independently
using their recorded previous target and measured q, not modified policy states:
0 failures, max joint difference2.384e-7 rad, mean solver.977ms,10 fallbacks.
The smaller standalone timing excludes tensor transfers and live reset work.
All20 initial and final joint q/dq, wrist and object states are bitwise equal to
`live_new20_v2_candidate`. This verifies endpoints and outcomes, not bitwise
equality of every unlogged intermediate state. The restored original IK3 run
also matches its original three endpoints bitwise. Frozen model/normalizer
verification passes. Success/termination definitions are unchanged.

## Failed worker experiment and recovery

Calling installed PhysX `set_thread_count(1)` unexpectedly persisted
`/persistent/physics/numThreads=1` in the Kit user settings. It was not actually
process-local. `fast_jac_thread01` and the subsequent `fast_ik_live20_v1` stalled
at physics-scene initialization; neither is valid performance evidence. They
were explicitly terminated after ordinary termination did not complete.

The setting was restored to **8**, read from installed
`SETTING_NUM_THREADS_DEFAULT`, using `/tmp/recovery_physx_settings_restore.py`.
The exact prior user override was not saved to disk before the stall, so this
is a verified restoration of the installed default, not a claim of byte-for-byte
restoration of prior preferences. The worker-changing CLI option was removed.
No more worker experiments were performed. Recovery verified user config value8,
normal startup, original3 endpoint equivalence, and successful full20 fast run.
Restoration log: `/tmp/recovery_physx_settings_restore.log`. This limitation is
retained instead of silently describing the API as process-local.

Runtime metadata now records native settings: threads8, updateToUsd=false,
updateVelocitiesToUsd=false, PhysX CPU dispatcher=false. Redundant USD writes were
already disabled, so disabling them is not a further performance fix.

## Reproduce

**Current selection (2026-09-09, after the comparisons below):** the user approved
the video controller plus compliant distal contacts as `rl.sh play-arm` default.
`train-arm` shares this one-environment configuration. Use
`--arm-controller baseline` for the former strict-IK path. Historical statements
below about unchanged defaults describe the experiments at their execution time;
`--match-recording` still restores the saved video's exact inputs. Current commands
and timed transfer are in [RL task](RL_TASK.md#approved-video-controller-and-timed-transfer).

### Video versus current-path diagnosis — 2026-09-09

The preserved presentation comparison under
`outputs/visualizations/presentation/robot_media_20260908/arm_comparison/`
is **not** the recent `--mode legacy --transfer-evaluation` experiment:

| Setting | Preserved right-panel recording | Recent transfer/rubber evaluation |
|---|---|---|
| Policy | original 5k `model_4999.pt` | original 10k, or separate transfer `model_99.pt` |
| Arm action | `SimpleMountedWrist`, existing recovery candidate | `RB3WristIKAction`, original baseline |
| Kp | `[700,20000,12000,900,2400,250]` | `[300,500,500,300,200,50]` |
| Kd | `[35,260,140,45,70,20]` | `[20,20,20,20,20,10]` |
| Wrist response / IK | tau=.1 s at 120 Hz; warm-first existing IK | raw equilibrium target, 30 Hz IK |
| Arm velocity target | final delivered joint-path difference / dt | zero |
| State bank | heldout20, first placement | old20 or heldout20 |
| Contact | original rigid | original rigid or opt-in compliant |

Both have 30 Hz policy and 120 Hz physics. The presentation .5× playback and
terminal still holds are editing choices, not changed simulation dt. Source
reference and wrist/mount definitions match; no mount/FK changes were needed.
Floating `SE3ImpedanceActionTerm` computes a Cartesian force/torque from the
decoded equilibrium pose (`mdp/actions.py`); it does not teleport to that pose.
The original arm baseline sends that equilibrium to joint IK, whereas the
existing video candidate first shapes its response at physics rate
(`mdp/simple_mounted_interface.py`). Different physical observations then cause
the same frozen policy to emit different subsequent actions. A lower error to
the raw equilibrium is not necessarily closer to floating physical motion.

Executed evidence in `outputs/diagnostics/arm_run_parity_20260909/`:

- `video5000_reproduction` and `matched_profile_smoke2` each reproduced all
  152 recorded physics samples exactly: arm/hand q/dq, wrist, can, action,
  delivered targets and timestamps. This is live inference, not action replay.
- Current 10k policy with the **unchanged video controller preset**: task and
  final lift/contact proxy both **40/40** for original contacts, and **40/40**
  with compliant contacts. No new tuning or policy training was performed.
  Same-checkpoint recent baselines were task35/40, hold33/40 (rigid) and
  task37/40, hold34/40 (compliant). Presets remain opt-in, not training defaults.
- Against actual floating states at identical physics timestamps, rigid wrist
  mean difference old20 **12.48→5.51 mm**, heldout20 **13.35→5.63 mm**;
  compliant **12.27→5.91 mm** and **13.33→5.82 mm**. Rigid wrist angle
  differences **.127→.048 rad** / **.153→.047 rad**. Rigid leader differences
  **.01996→.00881 rad** / **.01977→.00975 rad**. No time shifting; comparisons
  stop at the common recorded horizon if baseline terminates early.
- Common initial wrist/hand/object state and phase verified within 2e-6;
  initial policy observation differences <=1.20e-7, actions <=5.97e-8.
  Arm limits, hand gains, reference, material, checkpoint and initial states
  match within each controller comparison. Only the pre-existing controller
  preset differs; these are not isolated per-gain causal experiments.
- `floating_motion_comparison.json`, `actual_motion_comparison.png`,
  `summary.json` and `matched_profile_verified.json` retain the comparison.
  138 regressions passed. The failed first smoke command used unsupported
  `--visualizer null`; the successful rerun uses `--visualizer none`.

Remaining limitations: actual motion is not identical (about 5–6 mm mean
difference); max actual arm acceleration reached 927.88 rad/s² in these short
simulations. The old video controller's max delivered joint step is .083334 rad;
tiny hand limit overshoots remain <=.001896 rad. IK failure count was zero in
the 80 candidate episodes; drive-only torque saturation remains UNKNOWN.
These are fixed finite simulation placements, not hardware safety or general
grasp reliability guarantees. The exact GUI command the user saw was not
provided; the diagnosis above concerns the preserved recording and identified
recent evaluation paths, not an inferred running window.

Commands (fresh output directories required):

```bash
# Exact original recorded controller, 5k policy and first held-out placement.
# Exits after the recorded one episode. No video slow-motion/holds are applied.
bash scripts/play_arm_candidate.sh outputs/diagnostics/matched_video_view \
  --match-recording outputs/visualizations/presentation/robot_media_20260908/arm_comparison/residual_rl

# Current 10k original policy, same video controller, 20 saved random placements.
bash scripts/play_arm_candidate.sh outputs/diagnostics/current_video_controller_view \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt
# Optionally add the separately validated compliant material to this command:
# --fingertip-contact-config config/experiments/revo2_rubber_contact.json
```

`--match-recording` restores recorded resolved values instead of mutable preset
defaults, checks saved input hashes and runtime gains/limits/dt/native settings,
and verifies actual reset states. Contact/transfer overrides are rejected for
this hard-contact recording mode. GUI/capture choice remains explicit. The
evaluator now prints `[resolved execution]` (checkpoint, controller, gains,
contact, timing and state bank) before rollout. Do not run a transfer checkpoint
under this different controller and call it a transfer-training comparison;
the existing transfer controller contract and defaults remain unchanged.

Optional scheduling experiment: append `--ik-policy-rate` to `play_arm_candidate.sh`.
This samples the existing 120 Hz shaped pose on the first physics substep of
each 30 Hz policy action, then holds the accepted IK goal for the remaining
substeps. Existing 120 Hz joint slew limits and velocity-target calculation
remain unchanged; no new interpolation or prediction is added. Reset IK is
separate and clears the pending flag. The GUI prints solve/apply counts, and
metadata records the option. Full traces distinguish the last sampled IK input
from the continuously shaped target. This is not the validated 120 Hz IK
candidate and requires separate grasp evaluation; the default remains unchanged.

GUI smoke executed with `bash scripts/play_arm_candidate.sh
outputs/diagnostics/arm_transfer_recovery/live_ik30_20260907_01 --ik-policy-rate`.
Runtime reported 180 IK solves / 720 physics applies (reset IK excluded),
confirming 30/120 Hz scheduling. Early rolling throughput was about .55–.58x;
this is not a completed paired success comparison. All 102 lightweight tests
passed, including new scheduling, unchanged-default, failure-hold and reset tests.

```bash
# Actual live policy + dynamic can + Kit viewer, fresh directory required.
bash scripts/play_arm_candidate.sh outputs/diagnostics/arm_transfer_recovery/my_fast_view \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt

# Recorded-input IK comparison, no Isaac GUI needed.
bash scripts/benchmark_warm_start_ik.sh \
  --source outputs/diagnostics/arm_transfer_recovery/live_new20_v2_candidate \
  --output outputs/diagnostics/arm_transfer_recovery/my_ik_comparison.json --stride 1

./scripts/run_tests.sh
```

To reproduce the original controller, use `scripts/evaluate_mounted_interface.sh`
with the same config/inputs but without `--fast-ik`; normal `scripts/rl.sh` defaults
are unchanged. Omit `--realtime-view` to recover full diagnostic recording (slower).

Validation:97 tests passed, including five new IK tests (Jacobian vs central
difference, model residual equivalence, valid warm solve, discontinuity/failure
fallback, unchanged default seeds), compile/shell checks and focused diff review.
The preceding1x video remains a recording, NOT evidence of live real-time speed.
Remaining task: detailed PhysX CPU/GPU profiling without lowering contact/solver
accuracy; no hardware or broad physics-engine changes undertaken here.

## Zero-agent versus learned-policy video

`outputs/visualizations/comparisons/rb3_revo2_fast_zero_vs_rl_20260907/`
contains separately executed native Isaac captures and
`isaac_rb3_retargeting_vs_residual_rl.mp4` (2560x720,30fps,4.267s).
Left:12 zero residuals, no policy inference. Right:frozen checkpoint from above,
actual observations every control step. Both use fast IK, identical transfer
controller, initial state and dynamic contact. The recorded single placement
failed via object_deviation for zero action and succeeded for the learned policy;
this video is an illustration, not an additional success-rate estimate.

Playback is1x simulation time with1s initial and2s final presentation holds.
Gym's final auto-reset image is removed on each side; the earlier-ending side
freezes its last genuine pre-reset frame with an explicit termination label.
Physics commands and termination criteria are never extended or disabled.
`comparison.json` records pairing checks and outcomes. Raw videos, metadata,
policy/physics traces and console logs remain alongside it.

```bash
bash scripts/record_arm_comparison.sh outputs/visualizations/comparisons/my_arm_pair \
  logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt
bash scripts/compose_arm_comparison.sh outputs/visualizations/comparisons/my_arm_pair
```

New evaluator option `--zero-actions` defaults off and is restricted to live
grasp evaluation.98 regression tests passed after adding it; paired initial
states were bitwise equal and every recorded zero-agent action was verified zero.

### Match the floating-hand comparison edit

`compose_arm_comparison.sh RECORD_DIRECTORY --floating-style` uses the original
captures (excluding terminal reset images) at the same 0.25x simulation-time
playback as `isaac_floating_20260907_take2`: 1920x540, 30fps, 10 seconds, a 1-second
initial hold and final-frame padding. Titles, divider and footer match the
floating edit; the earlier-ended panel is explicitly labelled as held. This
does not stretch the failed episode to the successful episode's motion duration.
It is incompatible with `--motion-duration`; normal composition is unchanged.

```bash
bash scripts/compose_arm_comparison.sh \
  outputs/visualizations/comparisons/rb3_revo2_fast_zero_vs_rl_20260907 --floating-style
```

Executed output: `isaac_rb3_retargeting_vs_residual_rl_floating_style.mp4` and
`comparison_floating_style.json` in that directory. Existing outputs are refused;
raw footage and previous edits remain intact. This is re-editing, not a new
simulation or grasp evaluation.
