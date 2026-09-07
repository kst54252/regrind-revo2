# Successful measured floating motion replayed through the arm

2026-09-07. Executed diagnostic, **not a closed-loop policy evaluation**.
[Interface](MINIMAL_MOUNTED_INTERFACE.md) · [Previous response comparison](MOUNTED_RESPONSE_COMPARISON.md)

## Question and result

Does replacing policy equilibrium targets with the successful floating hand's
**actual** wrist/hand motion make the mounted arm reproduce the grasp?

**Not with the current drive response.** Same 20 initial placements:

| Execution | Outcome |
|---|---:|
| Recorded floating closed-loop policy | 20/20 |
| Mounted actuator replay of measured floating motion | 0/20 |

The 20 arm runs first hit the unchanged object-deviation criterion at 0.900 s
(placement 0: 0.933333 s). All recorded samples were still replayed for diagnosis,
without extending the motion or resetting early. No final lift/contact proxy
passed either. Final can lift is approximately zero vs source 0.225 m.
Do not interpret these results as a general RL success probability or compare
them as another live-policy variant: this replay loads/calls no policy.

## Exact replay semantics

- Source: `outputs/diagnostics/minimal_interface_20260907/floating20/` contains
  successful runtime states every **1/120 s**, not just the 30 Hz policy targets.
- Wrist: measured `state.wrist_pos`, `state.wrist_quat`, same env-local
  `right_hand_base_link` point and world-aligned XYZW rotation as verified IK.
  The mounted link6 offset remains inside the existing FK/IK. No new mount math.
- Six hand leaders: measured `state.hand_q`, matched by names. Existing hand
  action implementation clips position limits and generates its five existing
  deterministic mimic targets. Recorded follower states are **measurements for
  comparison**, not five extra independent commands. Thus this is six-leader
  measured-motion replay, not an imposed 11-joint finger-state animation.
- Right-endpoint zero-order hold: recorded state at `t_k` is the target sent
  before physics interval `(t_k-dt,t_k]`; actual state is measured after that
  interval at `t_k`. This is offline one-timestep endpoint preview, not a live
  causal policy. No fitted time shift, added settling, retiming or interpolation.
- IK is solved each physics step using the existing previous-goal/actual-neutral
  seeds. Existing arm velocity-bound position slew and zero velocity target are
  preserved. Hand and arm targets share a timestamp. No extra feedforward.
- Joint drives remain physical; no joint/root state writes except original reset.
  Can remains dynamic, never driven by its recorded pose. Table, gravity,
  collisions, gains, effort/velocity limits, reference and assets are unchanged.
- Source command frames determine reference/termination phase, including original
  post-reset phase behavior. Existing termination functions are checked at the
  original 30 Hz boundaries. First termination determines diagnostic outcome;
  later reference-end flags cannot turn an earlier failure into success.

## Measured results

2,964 matched physics samples over 20 complete recordings. All common initial
wrist/hand/follower/object states match the floating source within 2e-6; mounted
joint/root initial states, gains, limits, gravity and spawn config match prior
simple-arm evaluation. Six hand gains/limits also match floating by joint names:
Kp=3, Kd=0.1, effort limit=0.5, velocity limit=100 (runtime configuration values,
not evidence of drive effort saturation).

| Error versus measured floating motion | Mean | P95 | Maximum |
|---|---:|---:|---:|
| Wrist position, full recording | 17.67 mm | 36.10 mm | 45.70 mm |
| Wrist orientation, full recording | 7.58 deg | 27.19 deg | 52.64 deg |
| Wrist orientation, through first failure only | 4.92 deg | 9.18 deg | 17.53 deg |
| Six leader joint absolute error, full recording | 0.57 deg | 1.78 deg | 16.11 deg |

Statistics pool actual samples/joints; no episode reset discontinuities are
included. The 52.64-degree maximum occurs **after failure**, placement 13 at
1.183333 s. Position maximum is placement 4 at 0.683333 s, before failure.

- IK failures **0**, maximum IK position error **6.42e-8 m**.
- Arm position-limit and actual velocity-limit violations **0**. Existing
  command slew limiter active on **99** steps; targets were not silently
  reported as perfectly delivered.
- Tiny source measured leader excursions required clipping on **606** steps;
  largest target correction **0.00052654 rad**. This is explicitly logged.
- Maximum actual joint-limit excursion **0.00122536 rad**, in the hand; do not
  claim all robot joints stayed inside limits.
- Mean contact onset: source **0.393333 s**, arm **0.230417 s**.
- Placement 4: before contact, at 0.083333 s, wrist is **11.99 mm lower** than
  source; contact starts 0.175 s vs source 0.391667 s. Pre-contact wrist error
  mean/max 12.12/19.45 mm. Leader absolute error mean/max 0.0110/0.0273 rad.
- All analyzed runtime positions, orientations, velocities and joint states are
  finite. Effort saturation remains **UNKNOWN**.

Verified conclusion: policy feedback differences are not required for this
motion-reproduction failure; actual arm and finger trajectories still differ
from the recorded successful motion. This is not proof that matching full actual
motion would fail. The test did **not achieve** such a match.

Important limitation: a finite-PD position drive needs position error to exert
load-dependent force. Reusing a measured angle as a new target does not recreate
the original drive effort or grasp preload. The hand's six-leader/mimic control,
mounted gravity/inertia and changed contact response remain relevant. This test
does not isolate each contribution or prove a particular gain/torque fix. No
retraining, controller tuning, new successful-grasp claim, or default promotion
was performed.

## Code and reproduction

- `scripts/replay_floating_actual_motion.sh`: maintained diagnostic launcher.
- `tools/rb3_revo2_ik/actual_motion_replay.py`: validated source loading and replay.
- `tools/rb3_revo2_ik/evaluate_mounted_interface.py`: opt-in `--stage actual`,
  existing environment/reset/PhysX logging reused; no runner/policy constructed.
- `scripts/analyze_actual_motion.sh` and `tools/arm_diagnostics/analyze_actual_motion.py`:
  same-timestamp comparisons and plots.
- `tests/test_actual_motion_replay.py`: measured-vs-target distinction, timestamp,
  success/quaternion validation, no-policy/no-teleport boundary checks.

```bash
bash scripts/replay_floating_actual_motion.sh --episodes 20 --headless \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --actual-source outputs/diagnostics/minimal_interface_20260907/floating20 \
  --states outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl \
  --output outputs/diagnostics/floating_actual_replay_repeat
bash scripts/analyze_actual_motion.sh outputs/diagnostics/floating_actual_replay_repeat
```

Use a new output path. For GUI, replace `--headless` with `--visualizer kit`;
`--episodes 1` shows the first
motion once and closes. The checkpoint argument validates the recording's SHA;
weights are not loaded. Reference is the source's existing stable HDF5.

Executed output: `outputs/diagnostics/floating_actual_replay_20260907/replay20/`:

- `metadata.json`: configuration, source, states and outcome; policy_loaded=false.
- `physics.jsonl`: desired measured-source poses, actual runtime states, IK and
  delivered q targets, leader/follower errors, contact, limits and timestamps.
- `actual_motion_targets.npz`: concatenated `episode`, `time_s`, `wrist_pos`,
  `wrist_quat_xyzw`, `revo2_joints`, dt/quaternion metadata. **Split by episode**;
  timestamps restart at reset. Initial t=0 states are in metadata/source logs.
- `termination_checks.json`, `analysis.json`, `comparison.png`.
- `policy.json`: originally an empty array (zero policy calls in this replay).
  The empty placeholder was removed in the later JSON retention cleanup;
  the evaluator still emits it on new runs. Metadata and physics traces remain.

Console: `/tmp/floating_actual_replay20_20260907.log`. First smoke encountered
`TypeError: 'bool' object is not callable` from installed `has_gui` property;
fixed to property access, then smoke2 and full20 completed. The failed smoke
log/output remain intact rather than being overwritten.

Validation: **83 tests passed**, Python compilation, launcher shell syntax and
diff whitespace checks. Source sample counts/timestamps, measured targets and
no-policy flags were verified against all 2,964 source samples. These tests do
not establish real-hardware safety or successful closed-loop policy transfer.
