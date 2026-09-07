# Arm transfer recovery — execution journal

2026-09-07. Completed bounded diagnosis, six-candidate search and actual policy validation.
Baselines/checkpoints/reference/assets and dirty working tree are preserved.

**Final result:** live mounted baseline/new = 19/20 → 20/20 on existing placements,
17/20 → 20/20 on twenty distinct held-out placements. These are finite simulation
tests, not a general success-rate guarantee or a hardware-safe controller.
The first attempted held-out bank accidentally repeated one placement: its logs
are preserved but EXCLUDED. Only `live_new20_v2_*` counts as held-out evidence.
See [final results](#final-executed-results) and [reproduction](#reproduction).
For subsequent actual live speed optimization and its limitations, see
[arm real-time execution](ARM_REALTIME_EXECUTION.md).

## Frozen scope and experiment order

1. Capture original floating controller inputs at physics boundaries (F0), then
   policy-free original-command replay (F1) and measured-state target replay (F2).
2. R0 retains measured wrist + measured leader targets. R1 changes only the
   fingers to original submitted commands, with measured wrist as arm reference.
3. Can-only removal diagnostics: five static holds, original-speed motion,
   four-times-slower motion. Keep attached hand, gravity and other collisions.
4. At most six major candidates; use first five existing placements for screening.
   Previously measured finite-gain profiles c1/c2/c3 may be tested individually;
   no effort/velocity limit increases, retraining, or new controller architecture.
5. R2 and R1 use identical recorded inputs. Promising candidate gets full20;
   live policy gets baseline/new old20 and independently saved new20 placements.

Existing reference: `outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5`.
Checkpoint: `logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`.
Initial bank: `outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl`.
All new outputs: `outputs/diagnostics/arm_transfer_recovery/`.

## Known facts before this task

- Actual-motion arm replay 0/20 is NOT the old closed-loop policy result (19/20).
- IK/frame mapping already verified; reuse it unchanged.
- Measured finger angles are not the original drive equilibrium commands.
- Implicit force drive; baseline arm gains [300,500,500,300,200,50] /
  [20,20,20,20,20,10], effort [10,100,100,100,100,10], velocity 10 rad/s.
- Drive-only torque unverified; saturation remains UNKNOWN, not inferred from
  `applied_effort` or total reaction. See ARM_ACTUATOR_DIAGNOSIS.md.

## Progress / resume point

F0 completed: 20/20, 2,964 states bitwise identical to existing floating20.
F1 original-command replay: 5/5. All position/velocity/effort and wrist equilibrium
commands match exactly. Episodes 1–4 physical states bitwise identical. Episode 0
first q divergence 2.33e-10 rad, growing around contact: max wrist position component
29.6 µm, wrist angle .0114 deg, object position norm .154 mm, object angle .957 deg,
object angular velocity difference .720 rad/s. Thus not a bitwise reproduction of
the first episode; numerical/controller warm-up sensitivity is a hypothesis,
not a verified missing command. Explicit initial states and submitted commands
match. This limitation is retained; F1 still reproduces the grasp, unlike F2.
F2 actual-state-target replay in FLOATING: 0/5, all object deviation. This establishes
that converting measured state to a new target changes behavior even without RB3.
The original hand target differs from its measured joint state by up to .08315 rad.

R1 original fingers + baseline arm: 0/5. Can-only-removed diagnostics completed:
5 static windows max 10.66–12.93 mm; original-speed mean 16.24 mm; four-times-longer
ZOH playback mean 11.60 mm. Slow ZOH repeats samples and therefore does NOT lower
the single-step command jumps; peak sampled command speed is not divided by four.

c3 reduces the same five static windows to .459–.511 mm, while original-speed
no-can mean remains 6.16 mm. Gains c1/c2/c3 alone all 0/5 contact replay.
Candidate 4 adds ONLY bounded native velocity targets from final delivered q
differences at physics dt to c3: fixed replay **5/5**. This is NOT yet live-policy
success. No effort limits, source motion, hand commands or dt were changed.

Candidate accounting: 1=c1, 2=c2, 3=c3, 4=c3+velocity-path,
5=c3+path+physics-rate response tau=.1, 6=same with tau=.075. Budget exhausted;
no more tuning this round.

Candidate 4 fixed replay 5/5, mean wrist tracking .73 mm; live policy 0/5. This
isolates the remaining input-contract issue: policy equilibrium is not floating
actual motion. Candidates 5/6 shape equilibrium at physics dt (not additionally
at policy dt), then use the same IK/gains/velocity path. Both live **5/5**.
**Freeze candidate 5 (.1 s)** before full/held-out results: acceleration P95
39.74 vs 51.86 rad/s²; lift .223–.227 m vs .232 m, closer to floating .225 m.
Candidate 6 has a lower maximum acceleration (238 vs 277 rad/s²), a recorded
trade-off, but not selected. No hand control changes in live execution.

Resume: fixed full replay and old20/new20 policy evaluations queued by
`scripts/finish_arm_transfer_recovery.sh 0.1`. Candidate selection is frozen.

Full replay completed: **R0 0/20; R1 0/20; R2 20/20**, each 2,964 samples.
R0 joint q/dq, wrist and object states reproduce the original actual-motion
replay exactly (max component difference zero). R2 uses c3+velocity-path only;
the response filter is NOT used for recorded actual-wrist replay.
Live baseline old20 is now running, followed by frozen candidate and held-out runs.

R2 caveat: the unchanged task-success flag is 20/20, but the supplementary final
.2 s lift/contact proxy is **19/20**. Placement 13 loses contact near 1.2 s and
ends with lift .168 m rather than .225 m; it remains within the old termination
tolerance. Do not call all 20 sustained grasps. No success rule was changed.
R1→R2 target poses/timestamps and all original hand q/v/feedforward commands
verified identical; initial states and non-gain physical configuration match.
Mean wrist error 17.65→.91 mm, but max R2 position error is 15.42 mm: not a
uniform 1-mm controller guarantee.
Legacy old20 rerun completed 19/20 and exactly reproduces old joint/wrist/object
state traces. New live policy old20 is running.

First full live runs: old20 baseline 19/20 → candidate **20/20** (placement 15
recovered, no regression). The original `live_new20_*` 20/20 runs were subsequently
INVALIDATED as held-out evidence: PLAY disabled placement randomization, and all
twenty resets were X=.4, Y=0. Those runs and `heldout_initial_states.jsonl` remain
preserved, excluded from final summary and plots. The sampler-only flag was fixed,
duplicate-position rejection was added, and both conditions were rerun as v2.
Controller selection stayed frozen throughout; no held-out tuning occurred.

F1 strict reproduction completed with `F1_pipeline`: replay captured RAW actions
through original action-manager decoding/reset/history exactly once, policy not
loaded/called. All decoded/native commands AND all joint/wrist/object states match
F0 bitwise for all 744 samples/5 episodes, including episode 0. This fixes the
replay execution-path discrepancy without a physics/controller change. The first
manual F1's tiny initial divergence cannot be uniquely attributed to one low-level
numeric/JIT detail and remains documented; use F1_pipeline for strict reproduction.

Resume: task complete; no further tuning. All final plots and paired statistics
use the corrected v2 held-out runs. See limitations below before deployment.

## Verified input and execution contracts

- Shared decoder remains `SE3ImpedanceActionTerm.process_actions`: policy outputs a wrist
  equilibrium pose residual, NOT an already achieved wrist trajectory. The
  original floating controller applies a force/torque PD wrench toward that
  equilibrium. Its actual motion contains load error and response lag.
- F0 records pre-step native position/velocity/explicit-effort targets and
  post-step PhysX state. One command applies on `(t-dt,t]`, measured state at `t`.
  No error trajectory is time-shifted. Quaternion order is **xyzw**, rotations
  world-aligned; pose commands are env-local, env origin is explicitly handled.
  Wrist point is `right_hand_base_link`; validated mounted FK already includes
  link6-to-hand translation `[0,0,.141304972]` m and identity rotation. No second
  mount correction was added.
- Replay never loads/calls a policy. Strict F1 uses recorded raw actions through
  the ORIGINAL manager exactly once; raw action replay is a diagnostic, not live
  policy evaluation. F0, R0 and legacy baseline also reproduce their prior state
  traces bitwise. Initial controller/history differences in manual F1 remain
  documented above; the original-pipeline F1 eliminates the discrepancy.
- Live inference uses measured mounted wrist, fingers and object through the
  existing observation/normalization/action path. No floating world or desired
  state is substituted. Each candidate full20 run passes 741 observation/action
  parity checks and verifies frozen model/normalizer state. Previous action,
  history and phase remain managed by the original environment.
- Policy dt=1/30 s, physics dt=1/120 s. The selected opt-in command filter updates
  only at physics rate: alpha=1-exp(-dt/.1); position and shortest-angle rotation
  approach the decoded equilibrium. This deliberately shapes the input; it is
  NOT an exact model of floating dynamics. Existing IK then computes bounded
  joint goals. Final qcmd is limited by existing position and velocity bounds;
  velocity target is `(qcmd[k]-qcmd[k-1])/dt`, not the IK raw output difference.
  Reset clears this history; no root/joint overwrites occur during tracking.
- Native setter interception confirms position/velocity/effort delivery once
  per physics step and detects overwrites. Hand commands and mimic coupling
  retain their original path; six independent leaders, five followers.
- Runtime backend is PhysX, implicit force drives (`get_drive_types()==1`).
  Installed distributions: Isaac Sim 6.0.1.0, isaaclab 13.3.0 (editable checkout
  reports 3.0.0), isaaclab_physx 3.1.1. Arm stiffness/damping below are runtime
  verified, not inferred from hand settings. Native explicit effort targets are
  logged in joint order, N·m, at submission time; these are NOT the implicit
  solver drive torque. **Drive effort saturation UNKNOWN**. No added gravity
  compensation. Gravity, contact, dt, mass/inertia, limits and assets preserved.

## Selected experimental configuration

`config/experiments/rb3_transfer_recovery_candidate.json` opts into c3 gains,
bounded velocity-path targets and physics-rate response tau=.1 s. Existing gain
profile definitions are reused from `config/experiments/rb3_precision_candidates.json`.
No normal play/train defaults changed.

| Arm joint | Baseline Kp → selected | Baseline Kd → selected | Effort limit N·m |
|---|---:|---:|---:|
| base | 300 → 700 | 20 → 35 | 10 |
| shoulder | 500 → 20000 | 20 → 260 | 100 |
| elbow | 500 → 12000 | 20 → 140 | 100 |
| wrist1 | 300 → 900 | 20 → 45 | 100 |
| wrist2 | 200 → 2400 | 20 → 70 | 100 |
| wrist3 | 50 → 250 | 10 → 20 | 10 |

Force-drive units: Kp N·m/rad, Kd N·m·s/rad. All arm speed limits remain 10 rad/s.
Hand Kp=3, Kd=.1, effort=.5 N·m remain unchanged. Inherited floating-wrist gain
fields in mounted metadata are not used to apply a wrench to the mounted robot.

## Final executed results

Counts below use the **unchanged task success flag**. Supplementary sustained
lift/contact proxy means final .2 s minimum lift >=.1 m and contact >.01 N for
at least 80% of that window. This is a separate diagnostic, NOT a changed reward
or success condition. Indices are zero-based.

| Fixed input replay | Task success | Sustained proxy | Wrist mean position | Mean rotation |
|---|---:|---:|---:|---:|
| R0 actual wrist + actual finger targets | 0/20 | 0/20 | 17.675 mm | 7.579° |
| R1 actual wrist + original finger commands | 0/20 | 0/20 | 17.646 mm | 7.561° |
| R2 R1 inputs + c3/path | 20/20 | 19/20 | .911 mm | .230° |

R1/R2 exact target timestamps, wrist poses and all original hand q/v/effort
commands are verified identical. R2 has no response filter because its input is
already measured floating motion. IK failures=0. R2 raw wrist max error15.422 mm;
FK(delivered qcmd)-runtime max2.021 mm. Placement13 slips near1.2 s and ends lift
.168 m; do NOT interpret its unchanged task-success flag as a sustained grasp.

| Live policy, actual observations | Baseline success → new | Sustained proxy | Failure → success | Success → failure |
|---|---:|---:|---|---|
| Existing20 | 19 → 20 /20 | 19 → 20 | 15 | none |
| Distinct held-out20, v2 | 17 → 20 /20 | 16 → 20 | 6,15,16 | none |

Baseline failed episodes terminate with `object_deviation`. Held-out baseline14
passes the old criterion but fails supplementary lift proxy (.0913 m). All live
candidate episodes satisfy both checks. All compared arm initial q/dq, root,
hand, object and phase match within2e-6; invariants are checked by the analyzer.
Floating/arm initial wrist/hand/object match, but embodiment mass, actuation and
arm collisions differ. Existing20 XY span X[.416882,.492048], Y[-.186788,.156309].
Held-out20 XY span X[.410102,.489489], Y[-.197026,.190156]. The new collection uses
seed20260908 AND saves actual reset states. All20 are distinct and have no overlap
with the old bank within10µm. No yaw, mass or friction generalization is claimed.

| Live metric (episode means averaged) | Existing baseline → new | Held-out baseline → new |
|---|---:|---:|
| Raw policy equilibrium → actual wrist position | 22.495 → 25.591 mm | 21.472 → 25.604 mm |
| Raw equilibrium → actual wrist rotation | 7.653 → 3.833° | 9.948 → 3.873° |
| FK(final joint command) → actual wrist position, max | 56.141 → 5.852 mm | 56.410 → 5.998 mm |
| Commanded 5-tip FK → actual tips, mean | 20.302 → 2.375 mm | 18.970 → 2.353 mm |
| Hand leader target/actual absolute error, mean | .02545 → .02081 rad | .02414 → .02103 rad |
| Object keypoint error, mean | 22.007 → 3.922 mm | 23.987 → 4.303 mm |
| Actual arm acceleration, maximum | 365.605 → 636.681 rad/s² | 273.377 → 729.266 rad/s² |

The raw policy equilibrium error deliberately does not hide the filter delay:
it increases even while final-command tracking and grasp improve. Better
equilibrium tracking alone (candidate4) failed0/5 live grasps. This supports an
input/response mismatch, not a claim that a smaller single wrist metric implies
better grasp. Original hand state is not interchangeable with original drive
command; F2 failed0/5 even without an arm. Finger-command restoration alone
(R1) was insufficient. Static gain response and moving-target response both matter.

### Stability and limits

- Five static final1s windows: baseline max10.66–12.93 mm versus c3 .459–.511 mm;
  selected max rotation .101°. This verifies static improvement, not uniform
  original-speed precision. Four-times-longer ZOH result retains instantaneous
  position jumps; no smooth-trajectory speed claim is made.
- Full R2 wrist3 max acceleration2400.001 rad/s², episode11 at .983333 s; near
  opposite speed limits. Old20 live new peak636.681, wrist3 episode11 at1.008333 s;
  held-out new peak729.266, wrist1 episode12 at1.183333 s. These are finite
  differences of runtime velocity at dt, no angle wrapping or post-hoc filtering.
  This is a material stability/portability limitation, not a hardware command.
- Native measured speeds reach10.000034 rad/s (about34µrad/s numerical excess).
  Submitted arm targets remain bounded; no arm position-limit excursions found.
  Maximum hand position-limit excursions: R2 .000526 rad; old20 live new .001782;
  held-out live new .001849. Thus do not report all actual joint limits as perfect.
- High Kp is finite and runtime verified, but effort saturation remains UNKNOWN.
  No observed gain result proves a torque deficit or physical RB3 limitation.
- Contact is can-to-robot filtered sensor force (onset threshold .01 N), not a
  verified per-finger grasp wrench. Lift/contact proxy is explicitly separate.
- No NaN/Inf or IK failures in completed full runs. The tests cover this checkpoint,
  reference, short episode and XY bank only. No longer hold, new task or hardware
  safety validation. Success does not guarantee low acceleration or robustness.

## Files and recorded artifacts

Implementation is isolated to `mdp/simple_mounted_interface.py` opt-in fields,
`tools/rb3_revo2_ik/evaluate_mounted_interface.py` diagnostic options, new
`recovery_probe.py`, `recovery_replay.py`, `analyze_transfer_recovery.py`,
`tests/test_transfer_recovery.py`, selected config and root recovery launchers.
The mdp path prefix is `regrind/source/regrind/regrind/tasks/manager_based/dexterous/`.
No underlying FK/IK algorithm, normalizer, policy, reward, termination, USD or
reference was rewritten. Existing unrelated dirty-tree edits are retained.

Under `outputs/diagnostics/arm_transfer_recovery/`:

- `F0`, `F1`, `F1_pipeline`, `F2`: original and floating replay diagnostics.
- `holds_*`, `motion_*`, `slow_base`, `*_screen`: bounded screening evidence.
- `R0_full`, `R1_full`, `R2_full`: separate fixed-input replay comparisons.
- `live_old20_*`, `live_new20_v2_*`: real closed-loop policy evaluations.
- `heldout_initial_states_v2.jsonl`: explicit actual reset bank; not a full
  recorded command trace and not compatible with a full-trace-only loader.
- Every run: `metadata.json` (inputs/SHA256, initial states, native configuration,
  frozen-policy checks, termination), `physics.jsonl` (precommand/poststate timing,
  original and delivered targets, actual joint/wrist/tip/object state, contact,
  FK/IK errors); policy records and termination checks when applicable.
- `summary.json`: per-placement errors before/after contact, delay estimates,
  per-joint step/speed/acceleration and limit proximity. Delay correlation is
  exploratory only; no metric uses time alignment to improve an error.
- `paired_comparison.json`: invariant checks and all outcome transitions.
- `comparison.png`: fixed-input wrist/lift, live success and actual placement XY.

Checkpoint SHA256 `10eeeff86c405b07595cdd692d3447e9385da43dc90c881dbe1807465f9c4721`;
reference `8b8de4be08db8b466ebb5b16e817dea78081c3c458e29b0f020c888105510796`;
held-out v2 bank `06448c87203191e31bdcb12129288ebb5d69a5e51338bf624bfd55c64c0399aa`.
Runtime metadata is authoritative for gains used in each run. Updating only the
selected config's final status label after evaluation does not change its four
controller fields.

## Reproduction

Run from repository root. Use fresh output names; evaluator refuses overwrite.
For a local GUI, omit `--headless` AND pass `--visualizer kit` (num_envs=1).
The installed Isaac Lab otherwise defaults to no local window.

```bash
# Selected actual mounted-policy evaluation, held-out bank
bash scripts/arm_transfer_recovery.sh --mode simple --episodes 20 --visualizer kit \
  --transfer-config config/experiments/rb3_transfer_recovery_candidate.json \
  --states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --output outputs/diagnostics/arm_transfer_recovery/view_selected

# Baseline: same checkpoint/state bank, original controller, fresh output
bash scripts/arm_transfer_recovery.sh --mode legacy --episodes 20 --headless \
  --states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --output outputs/diagnostics/arm_transfer_recovery/recheck_legacy

# R2 policy-free replay (R1: omit the last two controller flags)
bash scripts/arm_transfer_recovery.sh --mode simple --stage recovery --recovery-kind R1 \
  --actual-source outputs/diagnostics/arm_transfer_recovery/F0 --episodes 20 --headless \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --output outputs/diagnostics/arm_transfer_recovery/recheck_R2 \
  --arm-gains-key c3 --arm-velocity-path

bash scripts/analyze_transfer_recovery.sh outputs/diagnostics/arm_transfer_recovery
./scripts/run_tests.sh
```

The executed bounded suites are recorded in `scripts/screen_arm_transfer_recovery.sh`,
`scripts/screen_arm_response_recovery.sh`, `scripts/finish_arm_transfer_recovery.sh`.
They intentionally refuse existing output directories; do not delete original
evidence to rerun. Normal `scripts/rl.sh play-arm` remains the original route.
Rollback means omitting the selected experimental config/flags, not reverting
unrelated repository files.

Validation: 91 tests passed (original83 +8 recovery regressions), shell syntax,
focused change review and `git diff --check`. Isaac runs are the evidence for
contact performance; pure-Python tests are not a substitute. Existing tests emit
non-failing ResourceWarnings for old trace readers; no source cleanup attempted.
