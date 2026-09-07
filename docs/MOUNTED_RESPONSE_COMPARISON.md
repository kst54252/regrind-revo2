# Mounted grasp transfer: target error is not floating-motion error

2026-09-07. Diagnosis plus one executed command-response experiment.
**Not solved or promoted:** first-order wrist shaping reduced grasp outcomes
from 17/20 to 5/20. Default remains disabled (`response_tau=0`). No retraining,
gains/limits, physics, hand commands, references, assets, or termination edits.
[Interface/contract](MINIMAL_MOUNTED_INTERFACE.md) · [Architecture](architecture.md)

## Verified cause separation

`mdp/actions.py::SE3ImpedanceActionTerm.apply_actions` interprets the decoded
wrist target as a **force/torque PD equilibrium**. Floating Revo2 uses
Kp/Kd = 300/30 translationally and 3/0.3 rotationally; its robot gravity is
disabled. `mdp/simple_mounted_interface.py::process_actions` instead converts
that equilibrium directly to an IK position target. Attached-arm gravity and
joint-drive dynamics remain active. These are different physical response maps,
even with exactly the same initial observation and decoded action.

Existing logs: floating achieved 20/20 with a mean desired–actual wrist error
of 29.85 mm; simple arm achieved 17/20 at 18.90 mm. This is **not** evidence that
18.90 mm is closer to floating's actual trajectory. At placement 4, 0.300 s,
floating's actual wrist is 17.25 mm from its own equilibrium target. Perfectly
tracking that target would therefore not reproduce its actual position.

Matched placement 4 before the original arm's first can contact:

- 0.083333 s: decoded target difference only 0.247 mm, but actual mounted wrist
  is 18.48 mm below floating (actual orientation differs 2.376 degrees).
- First can contact: floating 0.391667 s; simple arm 0.100000 s.
- At 0.200 s the mounted can has already moved approximately +7.3 mm in X;
  floating can remains at its reset XY. At 0.400 s hand joint-error norms are
  0.093 vs 0.029 rad (arm vs floating).
- New per-body contact measurements identify index distal/touch and middle
  touch links at the first arm contact. Placement 4 peak per-body normal contact
  is 43.23 N at onset. It is **contact**, not drive torque. Placements 6 and 13
  also first touch with finger links. Native RB3 link contact with can is zero
  throughout both newly executed conditions. Fixed parts merged into an arm
  link are not independently sensed; table/self-contact was not newly measured.

Thus an early physical-state difference precedes changed object observations
and subsequent policy actions. The mount/FK path was already verified and was
not changed. This evidence does not apportion the early vertical transient
between gravity support, inertial response, and damping. Nor does it establish
that the old 19/20 legacy behavior is a robust transfer rather than favorable
contact timing. Effort saturation remains **UNKNOWN**.

## One attempted adaptation (rejected for use)

Added optional `--response-tau 0.1` to the isolated evaluator's simple mode.
This is a causal first-order command filter, not a floating simulator or an
observation substitution. At each 1/30 s policy boundary:

```
alpha = 1 - exp(-control_dt / tau)
p_ik = p_ik_previous + alpha * (p_policy - p_ik_previous)
q_ik = Exp(alpha * Log(q_policy * inverse(q_ik_previous))) * q_ik_previous
```

Quaternions are XYZW, shortest rotation, world-aligned axes; positions env-local.
The single selected tau = 0.1 s follows both floating Kd/Kp ratios. It is only
an overdamped approximation: it omits inertia, gravity support and contact.
History resets at each reference initialization. Real observations, frozen
normalization, policy/phase rate, original six finger targets, IK, position
limits and physics-step joint slew limits remain unchanged. No velocity
feedforward or extra wrist wrench is applied. `desired_wrist_*` retains the
original policy equilibrium; `ik_input_*` explicitly records the shaped target.
IK errors are measured against the latter, wrist tracking against the former.

## Executed paired results

Identical saved 20 placements, checkpoint SHA, initial arm/hand/object states,
phase, gains, limits and robot spawn verified. Each run uses its own actual
observations/actions. Floating and legacy numbers reuse the named prior runs;
baseline simple and shaped simple were freshly simulated.

| Condition | Existing task success | Mean wrist position error | Mean rotation error | Mean object keypoint error |
|---|---:|---:|---:|---:|
| Floating (prior run) | 20/20 | 29.85 mm | 4.58 deg | 3.10 mm |
| Simple arm, default (rerun) | 17/20 | 18.90 mm | 7.58 deg | 25.98 mm |
| Simple arm, tau=0.1 | 5/20 | 38.62 mm | 8.27 deg | 41.04 mm |

These are means of per-episode means over actual episode durations; failures
terminate earlier. They are not equal-duration motion scores. End-of-episode
lift/contact diagnostic counts also equal 20,17,5; task success itself is the
unchanged reference-end criterion, not a newly defined grasp test.

- Default failure indices: 4,6,13. Shaped failures:
  0,1,3,4,5,7,9,11,12,13,15,16,17,18,19 (object deviation).
- Default failure → shaped success: **6**.
- Default success → shaped failure: **0,1,3,5,7,9,11,12,15,16,17,18,19**.
- Mean first contact 0.125 → 0.205 s, still earlier than floating 0.393 s.
  Placement 4 moves 0.100 → 0.141667 s, not to floating's 0.391667 s.
- Arm absolute speed P95/max: 1.871/9.443 → 1.404/6.139 rad/s.
  Acceleration P95/max: 41.83/331.76 → 35.51/229.14 rad/s², calculated from
  successive actual velocities at dt=1/120 without crossing reset boundaries.
  Lower peaks do not establish absence of all oscillation, nor better grasping.
- Both runs: zero IK failures, arm position-limit and actual speed-limit
  violations. Tiny hand position-limit excursions remain: max 0.001273 →
  0.001136 rad. Do not report the entire robot as violation-free.

The attempt delays the wrist without delaying hand closing, and does not remove
the initial downward discrepancy. Worsened coordination is a supported hypothesis,
not an isolated causal proof. Simply adding delay is not the required correction.
Further transfer work should first address actual pre-contact wrist response
under mounted dynamics, retaining the common hand clock; do not select gains by
target-error score alone. No second tuning experiment was run here.

## Reproduction and artifacts

Run from repository root with a **new output directory**:

```bash
bash scripts/evaluate_mounted_interface.sh --mode simple --episodes 20 \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --states outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl \
  --response-tau 0.1 --headless \
  --output outputs/diagnostics/mounted_response_repeat/response20
```

Use `--response-tau 0` and a separate output for baseline. Reference remains
`outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5`.

New artifacts: `outputs/diagnostics/mounted_response_20260907/`:

- `baseline20/`, `response20/`: `metadata.json`, per-step `physics.jsonl`,
  actual-observation/action `policy.json`.
- `baseline_analysis/`, `analysis/`: `comparison.json`, `comparison.png`,
  `placement4.png` (wrist height, can height, finger contact/error in time).
- Console logs: `/tmp/mounted_response_baseline_20260907.log`,
  `/tmp/mounted_response_20260907.log`.

```bash
bash scripts/analyze_mounted_interface.sh outputs/diagnostics/minimal_interface_20260907 \
  --simple-directory outputs/diagnostics/mounted_response_20260907/response20 \
  --output-directory outputs/diagnostics/mounted_response_20260907/analysis
./scripts/run_tests.sh
```

Validation: **79 tests passed**, shell syntax, Python compilation and diff
whitespace checks. Default rerun's 2,848 samples exactly match previous simple20
for joint q/dq, wrist position/quaternion, can root state, policy actions and
final arm commands (maximum absolute difference 0). New observation/action parity
checks: 712 baseline / 618 shaped; frozen policy/normalizer verified in both.
These 20 placements do not establish a general success probability.
