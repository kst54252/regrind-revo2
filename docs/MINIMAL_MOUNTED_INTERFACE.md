# Frozen floating policy → minimal mounted interface

2026-09-07. Implemented and executed A/B validation, one-episode smoke, and
20 paired placements for three live-policy modes. **Not promoted to default:**
floating 20/20, legacy arm 19/20, new simple arm 17/20 using unchanged task success.
No training, reward/termination edits, gains/limits changes, asset regeneration,
or checkpoint/reference overwrites. [Architecture](architecture.md) · [RL task](RL_TASK.md).

Follow-up: [target-response diagnosis and rejected filter experiment](MOUNTED_RESPONSE_COMPARISON.md).
An opt-in 0.1 s wrist response filter achieved only 5/20; default stays unfiltered.
The subsequent [measured floating-motion replay](FLOATING_ACTUAL_MOTION_REPLAY.md)
achieved 0/20 without policy calls; it is an actuator reproduction diagnostic,
not a live-policy comparison.

## New path and reused components

- `scripts/evaluate_mounted_interface.sh` → `tools/rb3_revo2_ik/evaluate_mounted_interface.py`.
  Independent of accumulated `play.py` diagnostic switches. Uses the maintained
  deterministic PLAY environment configs and checkpoint runner; does not export/write policy files.
- `mdp/simple_mounted_interface.py::SimpleMountedWrist`: new action class,
  **not** the legacy `RB3WristIKAction`. Inherits the floating
  `SE3ImpedanceActionTerm.process_actions` decoder unchanged, overrides physical
  application with IK/position commands; never calls floating force application.
- `tools/rb3_revo2_ik/frozen_policy_adapter.py`: receives the **existing
  ObservationManager result**, checks shape/finite values, forwards to frozen
  policy. No independent observation normalization or desired-state substitution.
- Existing `RB3Revo2ReferenceCommand` reads actual PhysX articulation body/joint
  and object state; `FloatingObservationsCfg` owns observation order/history.
- Existing `RB3Revo2ResidualJointPositionAction` decodes six leader actions and
  deterministic five mimic followers. Followers receive existing derived drive
  targets, not additional policy actions.
- Existing `RB3730Kinematics` owns fixed transforms, FK and multi-seed bounded IK.
  New path seeds with previous accepted solution and measured joints as neutral.
  Failed IK holds the previous accepted target and logs failure; never reports
  a held target as a successful new solve.

## Policy contract verified in code

Actor shape `(1,67)`; order and half-open offsets:

| offsets | term | representation |
|---|---|---|
| 0:3 | object_pos | actual position minus env origin and placement offset |
| 3:9 | object_ori | first two rotation-matrix columns, row-major flatten |
| 9:15 | hand_wrist_pos | actual mounted wrist, 2-frame history |
| 15:27 | hand_wrist_rot6d | actual wrist rotation, 2-frame history |
| 27:39 | hand_joint_pos | 6 actual leaders minus default, 2-frame history |
| 39:51 | actions | previous 12-dimensional manager action |
| 51:52 | command | existing reference phase |
| 52:61 | action_base_wrist_pos_and_rot6d | reference action base, not an actual-state replacement |
| 61:67 | action_base_hand_joint_pos | reference leader action base |

Terms are the original `mdp/observations.py` functions. History is oldest→newest,
managed/reset by installed ObservationManager. PLAY disables observation noise
and delay keys, as before. Existing actor `EmpiricalNormalization` is loaded
with the checkpoint and remains in eval mode. `torch.inference_mode()`, disabled
parameter gradients, and exact state-dict comparisons verify weights and
normalizer buffers are unchanged after each evaluation. Critic stays 94-D and
is not used for actions.

Action `(1,12)` = translation residual 3, rotation-vector residual 3, six Revo2
leader residuals. Original decoder clips raw actions to [-1,1], then scales:
position `1.0*control_dt = 1/30 m`, rotation `3.2*control_dt rad`, hand joints
`3.2*control_dt rad`. Wrist position adds residual **once** to reference;
orientation is `quat(delta_rotvec) * quat(reference)` (left multiplication).
Hand adds scaled residual once to reference joints, clips positions, generates
followers with the existing ratios and clips follower targets.

Physics dt = 1/120 s, policy/reference = 1/30 s, four physics substeps.
Existing env.step applies action→physics→termination/reward→autoreset if needed
→reference update→observation/history. Adapter does not update phase/history.
Reset clears previous action/history using the original managers. In the existing
autoreset sequence the command can advance to frame 1 before the returned next
observation; this behavior is preserved, not silently replaced with frame 0.

Runtime parity is checked on every policy step against
`ObservationManager.compute_group('policy', update_history=False)` and direct
frozen-policy invocation. Counts: floating 741, legacy 731, simple 712.
Unit tests exercise maintained physical-position observation functions,
canonical translation/default joint subtraction, same-input action parity,
normalizer freezing, invalid inputs, shared decoder/no teleport, and target slew.
The test helper does not claim to independently implement all history semantics;
full 67-D/history parity is additionally checked in actual Isaac runs.

## Frames and timing

All quaternions here are **XYZW**. Position units m, joint/rotation units rad.
Policy positional terms are env-local minus placement offset. Actual and desired
wrist log positions are env-local, rotation axes align with world. `object_state`
and robot root state logs are world-frame PhysX state. Validated `num_envs=1`
has origin [0,0,0]; no physical-arm state is substituted for floating wrist DOFs.

`right_hand_base_link` is the mounted wrist used by both policy observation and
IK. `rb3_model.json` defines:

```
T_link6_wrist = translation(0, 0, 0.141304972), rotation = identity
T_world_wrist = T_world_env @ T_env_link6(q) @ T_link6_wrist
T_env_link6_target = inv(T_world_env) @ T_world_wrist_target @ inv(T_link6_wrist)
```

The reused IK API **already targets the mounted wrist**, so the new decoder
passes its env-local wrist target directly. It must not subtract the mount
offset again. RB3 base in that env is `[0,0,-.02]`, identity rotation.

New command mode: pose target held for each policy interval, arm q goal held
between IK calls, with per-physics raw joint-coordinate slew bounded by native
velocity limit × dt. Velocity target explicitly zero; no feedforward, filters,
retiming, differential IK, compensation or gains changes. Slew-limited steps
are logged (186 in simple20); this can add delay, not hidden by timestamp shifting.
Legacy still uses its existing four-substep linear position interpolation.
Finger target and arm IK target are generated from the same policy action at
the same control boundary. Only reset may write root/joint states directly;
the dynamic can is never moved by reference during grasp rollout.

## Executed validation and comparison

Checkpoint:
`logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`.
Reference: `outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5`.
State bank: `outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl`.
It contains explicit reset states, not just a seed. New metadata saves the actual
initial states again and verifies common states against the bank.

### A/B: no can contact

Can is moved +10 m X **at reset only**. No collision/gravity/controller changes.
38 original reference frames/152 physics ticks, zero residual, no policy inference
used for this diagnostic. Table/self-collision settings remain unchanged.

- IK input vs FK(accepted IK): max 5.13e-8 m / 2.07e-7 rad, zero IK failures.
- Actual-joint FK vs runtime wrist: max 4.79e-7 m / 1.17e-6 rad.
- Delivered-target FK vs actual: mean/P95/max **17.50 / 37.48 / 47.22 mm**,
  rotation **.1216 / .2009 / .2052 rad**. Tracking is not precision-grade.
- Filtered can–robot contact force = 0 N. This is object-noncontact validation,
  not a claim that all robot/table contacts were absent.

### C: live policy, actual observations, common 20 initial placements

| mode | original task success | supplementary lift/contact proxy |
|---|---:|---:|
| floating | 20/20 | 20/20 |
| legacy arm | 19/20 | 19/20 |
| simple arm | 17/20 | 17/20 |

Task success is the original reference-end criterion; failures use the original
object deviation and wrist-object distance terms. Proxy is only supplemental:
can rise >=.1 m throughout last .2 s with can–robot contact on >=80% of those
samples. It is not a new reward/termination or proof of robust force closure.

Legacy success→simple failure: **4,6,13**. Legacy failure→simple success: **15**.
Simple failures and legacy failure all terminate by `object_deviation`.
Floating final can rise .22492–.22512 m; failed arm trials return to table level.

Mean of per-episode mean desired–actual wrist errors [mm]:

| mode | all | before first can contact | after first can contact |
|---|---:|---:|---:|
| floating | 29.85 | 20.66 | 34.01 |
| legacy | 22.50 | 14.38 | 23.42 |
| simple | 18.90 | 12.28 | 19.61 |

Lower wrist error did **not** imply better grasping. No rollout actions were
forced equal. Initial action max difference from floating is only 8.20e-8 for
both arm modes; later each policy responds to its actual, diverging observations.
No stochastic policy sampling/learning is used.

Actual arm initial q/dq/root and common hand/follower/object/wrist state+phase
match with atol 2e-6 (arm q/dq to bank 1e-6). Same wrist twist was set in floating
**during reset** to match the arm's zero initial wrist velocity. Dynamic behavior
still differs: floating hand gravity is disabled and SE3 force/torque impedance
is used; mounted hand/arm gravity is enabled and arm joint drives/attached
mass/inertia are present. Tables differ in shape, but both tops are Z=0.
No claim is made that these embodiments are physically identical.

Legacy evaluation reproduces all **2924** saved baseline physics samples:
all joint q/dq, actual wrist pose and object root state have max difference **0**.
Thus the new recorder/contact instrumentation did not alter this baseline.
Private PhysX contact caches are not serialized as part of reset-state claims.

Arm position-limit violations = 0; actual velocity-limit violations = 0; IK
failures = 0 in all arm trials. Small **hand** physical limit excursions exist:
max .000527 rad floating, .001021 legacy, .001273 simple (mainly index proximal).
These are reported, not hidden by target clipping. No NaN/Inf was observed.
No verified drive-only effort signal: effort saturation **UNKNOWN**, not inferred
from implicit-actuator approximate torque or total joint reaction.

Contact measurement uses installed PhysX `create_rigid_contact_view` with the
can as source and exact runtime articulation link paths as filter partners;
`get_contact_force_matrix(dt=physics_dt)` reports normal contact force [N].
Onset threshold .01 N is diagnostic, not grasp proof. It does not include tangential
friction as a separate signal or prove hand–table clearance. Failed broad-wildcard
sensor attempts were stopped; their console logs are retained.

## Commands, logs, tests

```bash
# A/B, use a fresh output directory each run
bash scripts/evaluate_mounted_interface.sh --mode simple --stage ab --headless \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --output outputs/diagnostics/minimal_repeat/simple_ab

# Smoke: --episodes 1. Full three-way comparison:
for mode in floating legacy simple; do
  bash scripts/evaluate_mounted_interface.sh --mode "$mode" --headless --episodes 20 \
    --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
    --states outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl \
    --output "outputs/diagnostics/minimal_repeat/${mode}20"
done
bash scripts/analyze_mounted_interface.sh outputs/diagnostics/minimal_repeat
```

Actual output root: `outputs/diagnostics/minimal_interface_20260907/`.
Each mode has `metadata.json` (states/config/termination/frozen checks),
`physics.jsonl` (every timestep: command/measurement time, wrist/hand/object
errors, force, actual q/dq, IK/limit flags). New captures also save raw per-policy
observations/actions in `policy.json`; earlier floating20/legacy20 captures
predate that optional addition but retain actions in every physics sample and
runtime observation/action parity counts. Their logs were not rewritten.

[Comparison plot](../outputs/diagnostics/minimal_interface_20260907/comparison.png) ·
[Per-placement/contact-split results](../outputs/diagnostics/minimal_interface_20260907/comparison.json).
Console logs: `/tmp/minimal_interface_{floating20,legacy20,simple20,simple_smoke}.log`,
`/tmp/minimal_interface_simple_ab_take3.log`. No default path was replaced.

Tests: `tests/test_frozen_policy_adapter.py` plus full `./scripts/run_tests.sh`:
**75 tests passed**, Python compilation, shell syntax and `git diff --check` passed.
Remaining: robustness beyond these 20 known XY placements, arm-aligned training
or a different physical wrist response (not attempted), drive-only torque,
true force-closure assessment, and hardware deployment. No automatic tuning or
retraining is included; this task establishes a clean, comparable interface.
