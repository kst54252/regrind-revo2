# Online arm command and tracking diagnosis

Measured 2026-09-07, using the existing `model_4999.pt` floating policy and
`20200709_143747_left/rb3_revo2_reference_stable.h5` (38 reference frames).
This follows [the verified wrist frame contract](WRIST_FRAME_DIAGNOSIS.md).
It diagnoses the **live residual-policy rollout**, not separate reference IK.
No policy, controller, gains, limits, physics timestep, collision settings,
mount transforms, or IK algorithm were changed for these measurements.

## Result

- Fixed X=0.50/Y=0: 10/10 task successes, but substantial tracking error.
  Its previous description as a failure refers to tracking, not task termination.
- Existing random-placement condition, seed 42: **19/20 task successes**.
  Episode 15 (zero-based, the 16th trial) failed with `object_deviation` at
  simulation time 0.900 s. Initial can position was
  `[0.45724237, 0.15630923, 0.01263600]` m.
- Both runs: no rejected IK solution, optimizer failure, or fallback. All
  observed target buffers and actual PhysX position-setter arguments matched
  `q_cmd` exactly after matching joints by name. No arm-target overwrite was
  observed in this delivery path.
- The largest sustained error is **C: delivered joint target FK versus actual
  runtime wrist**, not IK geometry or target delivery. B contains the existing
  four-substep interpolation transient and approaches zero at substep 4.
- This localizes tracking error; it does **not** prove that arm tracking alone
  caused the failed grasp. Successful trials also have substantial C error.

## Actual execution path

Line numbers below describe the measured checkout, not a stable API. Repository
paths use `mdp/` and `config/` relative to
`regrind/source/regrind/regrind/tasks/manager_based/dexterous/`.
External paths use `/home/wanjunkim/IsaacLab/source/` as their prefix.
The trace metadata also records source files and function start lines at run time.

| Stage | File, function, line | Observed contract |
|---|---|---|
| Launcher/task | `scripts/rl.sh:143`, play-arm branch at 153 | `Regrind-RB3-Revo2-TunaCan-Online-Play-v0` |
| Action configuration | `config/rb3_revo2/rb3_revo2_online_env_cfg.py:30`, `OnlineActionsCfg`; scales at 71 | `base_action_source="motion_target"`, four interpolation substeps |
| Policy → action | `regrind/scripts/rsl_rl/play.py:563`, `main` | `policy(obs)` then `env.step(actions)` at 566 |
| Target generation | `mdp/rb3_revo2_actions.py:288`, `RB3WristIKAction.get_base_pose`; `process_actions:307` | Reference wrist plus policy residual, not observed floating-hand pose |
| Actual IK call | `mdp/rb3_revo2_actions.py:344` → `tools/rb3_revo2_ik/rb3_kinematics.py:249`, `RB3730Kinematics.inverse` | Exact residual-adjusted target; previous accepted IK as warm start |
| Accept/fallback | `mdp/rb3_revo2_actions.py:357`, `process_actions` | Accept finite, in-limit success; otherwise hold previous valid target or measured q |
| Final joint target | `mdp/rb3_revo2_actions.py:385`, `apply_actions` | `torch.lerp(previous_sent, accepted_q, substep/4)`; setter at 400 |
| Per-physics-step loop | external `isaaclab/isaaclab/envs/manager_based_rl_env.py:183`, `step` | Process at 215; apply 239; write 241; physics 243; update 251 |
| Actuator processing | external `isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:247`, `write_data_to_sim`; `_apply_actuator_model:4477` | User-order buffer → actuator staging buffer → backend name/order conversion |
| Implicit actuator | external `isaaclab/isaaclab/actuators/actuator_pd.py:117`, `ImplicitActuator.compute` | Returns input target unchanged; approximate effort fields are not measured motor torque |
| Native target delivery | external `isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:346` | `root_view.set_dof_position_targets`; velocity setter at 347; exact arguments intercepted |
| Physics advance | external `isaaclab/isaaclab/sim/simulation_context.py:660`, `step` | Existing PhysX step; no inserted simulation steps |
| Actual state | external `isaaclab/isaaclab/scene/interactive_scene.py:527`, `update`; `mdp/rb3_revo2_commands.py:327/331`, `current_hand_wrist_pos/quat` | Fresh `robot.data.body_pos_w/body_quat_w` for `right_hand_base_link`, not USD XformCache |
| Reference clock | external manager `step:289` → `mdp/rb3_revo2_commands.py:532`, `_update_command` | Reference advances after physics and reset handling |

The native setter implementation path (including installed extension version) is
stored as `metadata.source_locations.backend_position_setter` in the JSONL.

### Which pose does IK receive?

For the wrist action after clipping each component to [-1, 1]:

```python
p_ik = p_reference + (1.0 * command_dt) * action_xyz
q_ik_target = quat_mul(
    rotvec_to_quat((3.2 * command_dt) * action_rotation),
    q_reference,
)
result = existing_kinematics.inverse(p_ik, q_ik_target, ...)
```

Quaternion multiplication applies the residual rotation on the left. The trace
captures the arguments at `inverse` itself, so it includes Euler/quaternion
conversion and clipping actually performed by the action term. `q_ik` in the
logs means **six joint angles returned by IK**, not `q_ik_target` above.
There is no separate floating robot being simulated to supply an observed wrist
target. The policy observes the mounted environment, then produces residuals.

## Instrumentation and timing

`--arm-execution-trace` in `play.py` loads
`tools/rb3_revo2_ik/trace_arm_execution.py` only when requested, for one online
PhysX environment. Instance-local observers call each original method exactly
once; they do not solve extra IK, change targets, draw random numbers, or step
physics. GPU reads can slow wall-clock execution, but do not change simulated dt.

JSONL records:

- `metadata`: model/checkpoint/reference paths, CLI/Hydra overrides, seed,
  source locations, joint names and user/backend mappings, actual gains/limits.
- `initialization`: state after reset, separate from dynamic samples.
- `command`: reference index, exact IK input, raw solver q/status/message,
  previous accepted q, acceptance/fallback, policy wrist action.
- `physics_sample`: command context; interpolated `q_cmd`; buffers before send,
  after actuator processing and around physics; actual native position/velocity
  setter arguments; intermediate buffer writes; pre/post q, velocity, object and
  PhysX base poses. Carries global physics step, episode-local step/time,
  command generation step, reference index and interpolation substep.
- `episode_end`: terminal state/termination flags **before automatic reset**.
- `reference_update`, `env_step_return`, `trace_end`: reference and reset timing.

Arrays use `[base, shoulder, elbow, wrist1, wrist2, wrist3]`. Both simulator joint
orders are mapped by name. Pose logs explicitly use **XYZW**, in world coordinates
at the Revo2 base; env origin is added back to environment-relative positions.
The standalone analyzer reuses `RB3730Kinematics.forward_batch` and the existing
shortest-angle quaternion error routine. A/B/C norms are separate quantities,
not additive scalars.

First-control-interval samples are labelled startup (0 < t <= 1/30 s); remaining
samples are work. Reset snapshots are separate. There is **no explicit warmup or
settling hold** in this execution; these labels do not invent one. Global
statistics have separate startup/work/all groups. Per-episode statistics cover
that episode including startup; CSV contains the phase label for finer filtering.
Do not pair a post-`env.step` reference index with the preceding command without
checking `reference_frame_at_command`: the reference can already have advanced.

Actual runtime arm parameters (same joint order):

| Parameter | Value |
|---|---|
| Physics / command dt | 1/120 s / 1/30 s |
| Actuator | `isaaclab.actuators.actuator_pd.ImplicitActuator` |
| Stiffness | `[300, 500, 500, 300, 200, 50]` |
| Damping | `[20, 20, 20, 20, 20, 10]` |
| Effort limits | `[10, 100, 100, 100, 100, 10]` |
| Velocity limits (rad/s) | `[10, 10, 10, 10, 10, 10]` |
| Observed native velocity targets | All zero rad/s |

Each physics sample had one position setter call and one velocity setter call.
Zero velocity targets are a measured implementation fact, **not proof of a
controller bug or insufficient torque**. Actual motor torque/contact force was
not measured; no torque-saturation conclusion is justified.

## Measured error separation

Failed episode 15, all 108 physics samples (initial reset excluded):

| Error | Mean | 95th percentile | Maximum | Maximum time (s), global step, reference frame |
|---|---:|---:|---:|---|
| A position (m) | 3.76e-16 | 3.65e-16 | 5.79e-15 | .175, 2245, 6 |
| A angle (rad) | 1.10e-9 | 0 | 2.98e-8 | .40833, 2273, 13 |
| B position (m) | .004069 | .010262 | .019671 | .54167, 2289, 17 |
| B angle (rad) | .012981 | .033687 | .044968 | .475, 2281, 15 |
| C position (m) | .016056 | .040516 | **.044602** | **.575, 2293, 18** |
| C angle (rad) | .118241 | .226505 | **.340548 (19.51 deg)** | **.900, 2332, 27** |

Work-only C mean/p95/max: .016443/.040524/.044602 m and
.122070/.230175/.340548 rad. At interpolation substep 4 only, B maximum falls
to 5.61e-8 m / 1.66e-7 rad, but C still reaches .043966 m / .340548 rad.
Thus the tracking discrepancy remains after interpolation completes.
Tiny A angle values are at numerical roundoff scale, not meaningful IK error.

The actual can first rose >2 cm at .30833 s, peaked only 2.3386 cm above its
initial position, and ended back at table level. The plot marks this **rise
proxy**, not verified contact or grasp acquisition. Contact/grasp timing remains
unconfirmed. Peak C error occurs later than the initial rise; direction of
causality between loss of the can, changed observations/residuals and tracking
error is not established.

For the fixed X=.50 run, work-only C mean/p95/max was
.019637/.045585/.051495 m and .136066/.362347/.405076 rad despite 10/10 successes.
The old control-rate first-episode NPZ and new control-rate NPZ had **zero maximum
difference** in target joints, measured joints, measured wrist position and
object position. Their old 37-sample target error (.050430 m / .401068 rad) is
unchanged. New statistics include every substep, all episodes and terminal
samples, so their maxima need not equal the old telemetry summary.

## Reproduce and inspect

Both commands below were actually run headless with the existing policy, physics
settings and seed 42, exiting successfully. They do not train. Choose a fresh
trace path on rerun: trace and analysis refuse to overwrite existing outputs.

```bash
./scripts/rl.sh play-arm \
  --sequence 20200709_143747_left \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --num_envs 1 --headless --eval_episodes 20 --random-placement \
  --arm-execution-trace outputs/diagnostics/arm_execution_random20_20260907.jsonl

./scripts/rl.sh play-arm \
  --sequence 20200709_143747_left \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --num_envs 1 --headless --eval_episodes 10 \
  --arm-execution-trace outputs/diagnostics/arm_execution_x050_20260907.jsonl \
  --arm-tracking-path outputs/diagnostics/arm_execution_x050_control_20260907.npz \
  env.commands.reference.randomize_object_xy=true \
  'env.commands.reference.object_start_x_range=[0.50,0.50]' \
  'env.commands.reference.object_start_y_range=[0.0,0.0]'

bash scripts/analyze_arm_execution.sh \
  outputs/diagnostics/arm_execution_random20_20260907.jsonl \
  --output-dir outputs/diagnostics/arm_execution_random20_analysis_20260907
```

Fixed-run analysis was also run with the same Python analyzer. Artifacts under
`outputs/diagnostics/`:

- `arm_execution_random20_20260907.jsonl`: 2,924 physics samples, 731 commands.
- `arm_execution_x050_20260907.jsonl`: 1,484 physics samples, 371 commands.
- `arm_execution_{random20,x050}_analysis_20260907/summary.json`: full metadata,
  phase statistics, solver/delivery checks, per-episode outcomes and extrema.
- Each analysis directory: `errors.csv`, `object_rise_all_episodes.png` and
  `stage_errors_episode_015.png` (random) / `stage_errors_episode_000.png` (fixed).
- Console logs: `/tmp/arm_execution_random20_20260907.log` and
  `/tmp/arm_execution_x050_20260907.log` (temporary; structured JSONL is durable).

Validation: both Isaac runs completed; `./scripts/run_tests.sh` passed **39/39**
tests (the existing 37 plus two new analysis regressions). Targeted tests cover
pose units/quaternion sign and masked maximum timestamp attribution. Logged
joint targets, actual joints/velocities and runtime base poses had no NaN/Inf.
Python compilation, shell syntax and `git diff --check` also passed. No mount/FK
implementation was changed; unrelated pre-existing working-tree edits remain.

## Follow-up experiment

The paired recorded-command experiment was subsequently authorized and
completed. See [can-contact comparison](ARM_CONTACT_COMPARISON.md) for full
arm/hand capture, identical-command replay with/without the can, and results.
The original traces above remain unchanged.
