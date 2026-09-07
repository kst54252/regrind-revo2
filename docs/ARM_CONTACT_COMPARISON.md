# Recorded-command can-contact comparison

Measured 2026-09-07. Follow-up to [command delivery diagnosis](ARM_EXECUTION_DIAGNOSIS.md).
Question: does the large mounted-wrist tracking error require contact with the can?
No retraining or controller tuning was performed.

## Finding

**Large tracking errors remain without the can in reach.** This rules out can
contact/load as a necessary cause of the observed error, not arm drive torque
limits, arm inertia/gravity, table contact, or self-contact. It does not establish
which gains should change or whether a motor is torque-saturated.

Failure episode 15, same 108 physics-step commands over 0.9 s:

| C: command FK versus actual wrist | Can present | Can absent |
|---|---:|---:|
| Mean position error | 16.056 mm | 16.554 mm |
| P95 position error | 40.516 mm | 45.147 mm |
| Maximum position error | 44.602 mm at .575 s | 51.234 mm at .56667 s |
| Mean angular error | .118241 rad | .134093 rad |
| P95 angular error | .226505 rad | .217411 rad |
| Maximum angular error | .340548 rad (19.51 deg) at .9 s | .328776 rad (18.84 deg) at .9 s |

These episode statistics exclude initialization, but include the first control
interval. Startup/work statistics are separately saved. Work-only mean position
error: 16.443 mm present / 16.960 mm absent.

Can contact does affect the motion: paired actual wrist positions differ by up
to 22.040 mm at .33333 s; orientations differ by up to .085175 rad at .50833 s.
Removing the can therefore is not dynamically identical, but does not eliminate
the large late tracking error. Do not subtract the two error norms and interpret
that scalar as a contact-force contribution.

The can-present replay raised the can only 23.386 mm before it returned to the
table, reproducing the original failure. No grasp/contact-force measurement was
added. The absent can stayed at least 10.071 m from the wrist. Table collision
and robot self-collision settings were preserved. Correction from the actuator
audit: the assembled robot configuration sets self-collisions **false**, not
enabled; no collision setting was changed by either experiment.

Arm joint order: base, shoulder, elbow, wrist1, wrist2, wrist3. Largest absolute
joint errors (rad), by joint:

- Present: `[.050419, .128000, .057594, .411727, .148354, .734730]`.
- Absent: `[.050408, .134554, .051236, .411731, .161714, .722778]`.

Measured peak speeds were below the configured 10 rad/s limits in both runs.
Differences of successive position commands divided by physics dt give maximum
command rates `[.9431, 3.3961, 1.2611, 8.6267, 2.6152, 8.8261]` rad/s (excluding
the first sample). These are **position-command slopes**, not delivered velocity
targets; the actual velocity-target buffers were zero. Neither observation
establishes available motor torque or rules out transient constraint effects.

## Controlled experiment and reproduction checks

1. Re-ran the original seed-42/random-placement failure condition with the same
   `model_4999.pt`, reference, physics and actuator configuration. Captured full
   state and arm/hand drive targets using an optional extension of the existing
   trace. The 108 samples of episode 15 matched the earlier diagnostic exactly
   in q_cmd, q_actual, runtime base pose, and object position.
2. Started a separate Isaac process per condition. Ran the first 15 episodes
   normally to preserve reset history and PhysX state, rather than attempting to
   reconstruct hidden contact caches from a position-only snapshot.
3. Verified initial full joint positions/velocities, robot root state and object
   root state against the source before any intervention: **all maximum deltas
   were exactly zero in both runs**.
4. In the selected episode, invoked **no policy and no IK**. Applied the recorded
   full joint position/velocity/feedforward-effort targets at each physics step.
   This includes the simulator's 17 joint coordinates, covering six arm joints
   and the hand's leaders/followers; it does not create additional independent
   policy actions. The policy still has the original 12-D action contract.
5. Present: no intervention. Absent: moved only the can +10 m in world X once,
   keeping its orientation, velocity, dynamics and collision properties. No
   persistent asset/config files were changed. Target/reference trajectory was
   **not** translated with the can.
6. Replayed exactly the source duration without termination/autoreset or
   reference/policy updates during the test. Table, robot geometry, gravity,
   gains, effort/velocity limits and physics dt remained unchanged.
7. Compared targets including **actual native PhysX position-setter arguments**
   after name-based mapping: all identical between conditions. Can-present full
   measured joints, velocities, wrist pose and object root state were also
   **bit-for-bit equal to the source execution**. The quaternion angle routine
   reports up to 5.2e-8 rad even for identical raw quaternion arrays due to
   normalization/acos roundoff; this is not a reproduction mismatch.

No measured motor torque is saved. `all_effort_targets` is the commanded
feedforward effort buffer, not measured implicit-drive effort. The existing
implicit actuator's estimated PD effort is likewise not proof of physical motor
torque saturation.

## Implementation and commands

- `regrind/scripts/rsl_rl/play.py`: opt-in full tracing and contact-replay flags;
  normal play/training/controller behavior is unchanged when these are absent.
- `tools/rb3_revo2_ik/trace_arm_execution.py`: optional full-state/target capture.
- `tools/rb3_revo2_ik/replay_arm_contact.py`: prefix reproduction, initial-state
  checks, isolated recorded-command replay and actual setter observation.
- `tools/arm_diagnostics/analyze_arm_contact.py`: existing mounted FK/error helpers,
  paired-command/config validation, phase statistics, CSV and comparison plot.
- `scripts/analyze_arm_contact.sh`: maintained standalone analysis launcher.
- `tests/test_arm_contact_analysis.py`: rejects changed hand commands, timing
  and actuator configuration when comparing a pair.

Equivalent commands to the completed experiment, from the repository root:

```bash
checkpoint=logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt
common=(./scripts/rl.sh play-arm --sequence 20200709_143747_left
        --checkpoint "$checkpoint" --num_envs 1 --headless --random-placement)

"${common[@]}" --eval_episodes 16 \
  --arm-execution-trace outputs/diagnostics/arm_contact_source_20260907.jsonl \
  --arm-execution-full-state

for condition in present absent; do
  "${common[@]}" \
    --arm-contact-replay outputs/diagnostics/arm_contact_source_20260907.jsonl \
    --arm-contact-episode 15 --arm-contact-condition "$condition" \
    --arm-contact-output "outputs/diagnostics/arm_contact_${condition}_20260907.jsonl"
done

bash scripts/analyze_arm_contact.sh \
  outputs/diagnostics/arm_contact_present_20260907.jsonl \
  outputs/diagnostics/arm_contact_absent_20260907.jsonl \
  --output-dir outputs/diagnostics/arm_contact_analysis_20260907
```

Use fresh output names to repeat: existing trace/analysis paths are deliberately
not overwritten. This diagnostic replay is headless; use normal `play-arm` for
interactive policy visualization, not these manually stepped test episodes.

Artifacts:

- `outputs/diagnostics/arm_contact_source_20260907.jsonl` (~20 MB).
- `outputs/diagnostics/arm_contact_present_20260907.jsonl`.
- `outputs/diagnostics/arm_contact_absent_20260907.jsonl`.
- `outputs/diagnostics/arm_contact_analysis_20260907/`: `summary.json`,
  `errors.csv`, `contact_comparison.png`.
- Console logs in `/tmp/arm_contact_{record,present,absent,analysis}_20260907.log`.

All three Isaac runs and analysis exited successfully. Root regression suite:
**43/43 passed**. Python compilation, shell syntax and diff whitespace checks
passed. The runtime experiment, not the unit tests alone, validates the PhysX
behavior. Existing unrelated working-tree changes were preserved.

## Remaining uncertainty / next measurement

The evidence does not support tuning gains specifically to compensate for can
weight. Before choosing stiffness/damping or effort-limit changes, establish a
verified source of actual implicit-drive torque and measure it alongside q_cmd,
q_actual and velocity during this same fixed-command test. Arm inertia/gravity,
position/zero-velocity drive behavior, and remaining table/self-contact are not
separated by the current comparison. No further intervention was performed.

Follow-up: [arm actuator response and measurement provenance](ARM_ACTUATOR_DIAGNOSIS.md)
records the subsequently authorized read-only actuator measurements.
