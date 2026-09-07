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
than headless because rendering is included. At handoff it is still running;
these are rolling measurements, not a completed20-episode GUI success claim.

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
bash scripts/play_arm_candidate.sh outputs/diagnostics/arm_transfer_recovery/my_fast_view

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
bash scripts/record_arm_comparison.sh outputs/visualizations/comparisons/my_arm_pair
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
