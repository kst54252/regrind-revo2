# Arm actuator response: deterministic ON/OFF replay

Measured 2026-09-07. Uses the exact source and baseline files in
[command diagnosis](ARM_EXECUTION_DIAGNOSIS.md) and
[contact comparison](ARM_CONTACT_COMPARISON.md), episode 15, 108 physics steps.
ON = can present; OFF = the established +10 m X relocation of the can. This is
not a newly disabled collision setting. Table collision and self-collision
settings were preserved. No controller tuning or physical/model change.

## Findings and limits of inference

- Both instrumented runs **exactly reproduce their respective existing ON/OFF
  baselines** in full joint positions/velocities, wrist position/quaternion,
  object state and command arrays. Maximum raw-array differences: zero.
- Actual drive-only torque is not verifiable through the inspected installed
  APIs. **Effort saturation = UNKNOWN for every joint**, at every timestep.
- At the labelled 95% diagnostic threshold, actual speed, command-path speed and
  submitted velocity target spend **0% / 0 s** near the 10 rad/s limits in both
  runs. This supplies no sampled evidence of velocity clamping; it is not proof
  that no internal solver constraint acts between observations.
- The implicit actuator's **approximate** effort is clipped briefly for wrist3,
  but not at .575 s or .900 s. That proves clipping of an auxiliary estimate,
  not clipping of the actual PhysX drive torque.
- At the late error peak, wrist1/wrist3 position targets move rapidly while
  submitted velocity targets stay zero. Pre-step P/D approximations substantially
  oppose each other. This supports testing the velocity-target contract, but
  does not establish that damping is wrong, that torque is sufficient, or that
  gains alone explain the response. Inertia/gravity and multi-joint coupling
  remain relevant; contact changes the mid-episode trajectory too.

## Installed implementation and arm configuration

Installed distribution metadata: Isaac Sim **6.0.1.0** (`isaacsim-kernel` same),
`isaaclab` **13.3.0**, `isaaclab_physx` **3.1.1**. The editable IsaacLab checkout
has `VERSION=3.0.0`, git description
`perf-2026-07-06-248-gcb129107e-dirty`. These version identifiers differ; do not
silently substitute one for another. Runtime articulation implementation:
`isaaclab_physx.assets.articulation.articulation`, device `cuda:0`.
Loaded tensor API source is in extension
`omni.physics.tensors-110.1.13+110.1.2.lx64.r.cp312.u7f4`.
The exact internal PhysX SDK revision was not inferred from an online manual.

Arm class: `isaaclab.actuators.actuator_pd.ImplicitActuator`, implicit mode true.
Native `get_drive_types()` returns **1 (force)** on all six joints;
`get_dof_types()` returns **0 (rotation)**. These are runtime reads, not an
inference from hand settings. Stage meters/unit = 1; angular tensor state and
targets use radians. Force-drive gains have units N m/rad and N m s/rad, rather
than acceleration-drive gain units. See the [PhysX drive unit contract](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/struct_px_articulation_drive.html).

| Arm joint | Runtime Kp | Runtime Kd | Runtime maxForce [N m] | Runtime max velocity [rad/s] | Max absolute joint error ON/OFF [rad] |
|---|---:|---:|---:|---:|---:|
| base | 300 | 20 | 10 | 10 | .05042 / .05041 |
| shoulder | 500 | 20 | 100 | 10 | .12800 / .13455 |
| elbow | 500 | 20 | 100 | 10 | .05759 / .05124 |
| wrist1 | 300 | 20 | 100 | 10 | .41173 / .41173 |
| wrist2 | 200 | 20 | 100 | 10 | .14835 / .16171 |
| wrist3 | 50 | 10 | 10 | 10 | .73473 / .72278 |

Values were obtained from native `get_dof_stiffnesses`, `get_dof_dampings`,
`get_dof_max_forces`, `get_dof_max_velocities` **before every physics step** and
were constant. They match the materialized arm actuator config in
`regrind/source/regrind/regrind/robots/rb3_revo2.py:29`, `:66`.
Spawn requests `drive_type="force"`; its initial 1000 rad/s drive velocity
setting is superseded by the arm's `velocity_limit_sim=10`, confirmed natively.
Do not use the hand's gains/limits (or the spawn default) as arm settings.

`play.py`'s stiffness/damping/effort multiplier options were all **1**;
`--auto_gravity_from_ckpt` was false. The online PLAY config supplies gravity
`[0,0,-9.81]`; native `get_disable_gravities()` returned zero for every robot
link. Native submitted arm feedforward effort was zero throughout, so no
nonzero command-side gravity compensation was delivered via that setter.
This does not remove physical gravity or prove it is dynamically compensated.

The arm config has no asymmetric/speed-dependent motor model. The installed
`ActuatorBase._clip_effort` is symmetric `torch.clip(-effort_limit,+effort_limit)`.
Native drive-model triplets (speed-effort gradient, max actuator velocity,
velocity-dependent resistance) are all `[0,0,0]`: no nonzero envelope parameters
were detected. These triplets are **not** the joint's 10 rad/s hard velocity
limit. An active performance envelope would require separate interpretation;
see [Omni Physics envelope semantics](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-joint-drive-and-performance-envelope).

## Measurement provenance

Installed files inspected (prefix `/home/wanjunkim/IsaacLab/`):

- `source/isaaclab/isaaclab/actuators/actuator_pd.py:37`, `:117`: implicit model
  documentation and `compute`. It returns the input action unchanged while
  computing approximate P+D+feedforward and a clipped approximation.
- `source/isaaclab/isaaclab/actuators/actuator_base.py:376`: symmetric approximate clip.
- `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:247`,
  `:4477`: action → actuator computation → target staging/reordering; native
  effort/position/velocity setters at 344–347. The clipped approximation is not
  the implicit actuator's submitted feedforward effort.
- Installed tensor `api.py:1759`, `:1823`: actuation-force and velocity setters;
  `:1963`: actuation-force getter; `:1991`: projected incoming-joint force getter;
  `:2224`: total 6D incoming-joint wrench getter. These wrappers call compiled
  backend methods; the inspected Python source does not expose solver internals.

| Logged signal | Classification, units, sign and physical meaning |
|---|---|
| `pd_p_pre_Nm`, `pd_d_pre_Nm` | **Approximate** force-PD components: Kp(q_cmd-q_pre), Kd(v_target-v_pre). Positive is along positive joint rotation. Uses pre-step state; not independently measured torques. |
| `computed_effort_approx` | **Approximate effort** copied from actual `ImplicitActuator.compute` during the original actuator call. Includes P+D+feedforward, not solved implicit integration or a force decomposition of gravity/contact/inertia. |
| `applied_effort_approx` | **Clipped approximate effort**, N m for these verified force drives. Name does not mean measured or explicitly submitted drive effort. |
| `submitted_effort_Nm` | **Explicitly submitted actuation/feedforward effort**, captured from native `set_dof_actuation_forces` before physics. Positive along the DOF coordinate; excludes the subsequently computed implicit drive torque. All zero. |
| `projected_joint_effort_Nm` | **Projected joint effort/reaction**, read after physics from `get_dof_projected_joint_forces`. API describes projection of the incoming link-joint wrench along the DOF motion direction. Not verified drive-only. |
| `drive_effort_Nm` | **Unavailable** (`null`). No verified solver drive-effort measurement used. |

For projected effort, the installed API offers no separate breakdown of drive,
constraint, contact, gravity, inertial and friction contributions. Inclusion of
each contribution and physical motor-sign calibration are not independently
verified. Its sign follows the API's projection convention, not a calibrated
motor torque sensor. **It is never compared to a drive limit** in this analysis.
The available total-wrench API uses child-joint-frame force/torque and link
indices, not DOF indices; it was inspected but not logged. No artificial sensor
or new backend was introduced.

`get_dof_actuation_forces` was also inspected. The setter documentation describes
using it to retrieve/modify the current submitted effort; it does not establish
drive-only solver torque. It is not used as a torque sensor. There is no separate
verified implicit-controller requested-torque signal beyond the approximations
and explicit feedforward stream above. Native projected effort cannot fill that gap.

## Indexing, timing and differentiation

Named order: `[base, shoulder, elbow, wrist1, wrist2, wrist3]`. User joint,
native/backend DOF, and actuator-group indices were independently matched by
name; all happen to be `[0,1,2,3,4,5]` in this installation.

Physics dt remains 1/120 s. Each row records command time `(k-1)*dt`, end-of-step
state time `k*dt`, global physics step, original source step/reference frame,
q_pre/v_pre, q_actual/v_actual, native velocity/effort setter arguments, runtime
limits, effort provenance, and stage-C wrist errors from the existing FK.
P/D and the actual actuator approximations are sampled **before** physics;
joint/wrist states and projected reaction are sampled **after** physics.
Do not evaluate the old approximate effort against post-step state.

`v_path[k]=(q_cmd[k]-q_cmd[k-1])/dt` uses raw joint coordinates with **no wrapping**.
For k=0 the preceding command is the recorded `previous_accepted_q` reset target.
This seed is saved. It is not measured velocity and is never submitted as a new
target in this experiment. Raw winding is retained across ±pi.

A labelled 0.1 rad per-joint step threshold flags suspicious command jumps;
none were found. Largest step was .073551 rad (wrist3). This diagnostic threshold
is not a safety or physical limit. Measured joint-position margins stayed at
least .21880 rad ON / .22057 rad OFF; commanded margins at least .22894 rad.

## Per-joint and temporal results

| Joint | Peak path speed | Peak actual speed ON/OFF | Peak PD approximation magnitude ON/OFF |
|---|---:|---:|---:|
| base | .9431 | .7873 / .7871 | 3.77 / 3.98 |
| shoulder | 3.3961 | 2.6806 / 2.8095 | 35.75 / 38.07 |
| elbow | 1.2611 | 1.6696 / 1.5466 | 15.82 / 14.99 |
| wrist1 | 8.6267 | 6.1841 / 6.1842 | 21.67 / 21.67 |
| wrist2 | 2.6152 | 1.9954 / 1.9937 | 9.70 / 14.45 |
| wrist3 | 8.8261 | 3.6936 / 3.6335 | 15.45 / 23.30 |

Speeds are rad/s; approximations are N m, **not measured drive maxima**.
All joints: actual/path/target speed near-limit fraction 0%, longest interval
0 s; submitted feedforward near effort limit likewise 0% / 0 s. Near means
>=95% of a meaningful finite bound, not confirmed saturation. Full mean/P95/max
absolute tracking errors, extremum times, fractions and interval endpoints are
saved per joint in `summary.json` and compact results in `per_joint.csv`.

Approximate effort near-limit/software-clipping observations:

- Base, shoulder, elbow, wrist1, wrist2: 0% / 0 s in both conditions.
- Wrist3 ON: **1/108 = .926%**, one 8.333 ms interval ending .133333 s;
  approximate -15.454 → clipped approximate -10 N m.
- Wrist3 OFF: **3/108 = 2.778%**, longest 8.333 ms, ending at .341667, .508333,
  .525 s respectively; approximate values +17.241, -11.009, -23.299 clipped to ±10.
- These are auxiliary software-estimate clipping events. **Confirmed physical
  drive clipping, its fraction and duration remain UNKNOWN**, not zero.

### At .575 s (reference frame 18)

ON stage C: 44.602 mm / .131831 rad. Shoulder has the largest angular joint
error at this instant, -.12800 rad; OFF shoulder error -.13455 rad, C position
50.781 mm. This suggests examining shoulder-related coupled kinematics, not a
proof that shoulder alone accounts for Cartesian error.

ON shoulder: path speed -2.4264 rad/s, actual speed -2.3372 rad/s, submitted
velocity 0. Pre-step P=-73.451, D=+42.450, approximate sum=-31.001 N m.
OFF P=-77.387, D=+45.428, sum=-31.959. Neither approximation is clipped here;
no velocity approaches the threshold. Physical drive saturation remains unknown.

### At .900 s (reference frame 27)

| Quantity | wrist1 ON | wrist3 ON | wrist1 OFF | wrist3 OFF |
|---|---:|---:|---:|---:|
| q_cmd - q_actual [rad] | +.411727 | -.734730 | +.411731 | -.722778 |
| Path speed [rad/s] | +8.62670 | -8.82606 | +8.62670 | -8.82606 |
| Actual speed [rad/s] | +6.18411 | -3.69359 | +6.18421 | -3.63350 |
| Submitted velocity [rad/s] | 0 | 0 | 0 | 0 |
| Approximate P [N m] | +139.785 | -38.308 | +139.786 | -37.685 |
| Approximate D [N m] | -118.305 | +34.837 | -118.307 | +34.211 |
| Approximate P+D+FF [N m] | +21.480 | -3.470 | +21.479 | -3.474 |

FF is zero. P exceeding a drive limit **on its own** does not imply clipping of
the combined drive. The approximated total is not clipped at this instant.
ON wrist1/wrist3 error grows from +.133/-.174 rad at .75 s to +.412/-.735 rad
at .9 s; OFF shows essentially the same late growth. Stage-C rotation reaches
.340548 rad ON / .328776 rad OFF. This is time-local evidence of a response
lag under moving positions and zero velocity targets, not an episode-average
argument that unsaturated means bad gains.

## Reproduce, artifacts, validation

No source/reference/baseline log was overwritten. Both new Isaac runs exited 0.
The first 15 episodes reproduce PhysX history as before; policy/IK calls are
zero within the measured episode. New optional flag: `--arm-actuator-diagnostic`.
Use a fresh output suffix for repetitions (existing outputs are protected).

```bash
checkpoint=logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt
for condition in present absent; do
  ./scripts/rl.sh play-arm --sequence 20200709_143747_left \
    --checkpoint "$checkpoint" --num_envs 1 --headless --random-placement \
    --arm-contact-replay outputs/diagnostics/arm_contact_source_20260907.jsonl \
    --arm-contact-episode 15 --arm-contact-condition "$condition" \
    --arm-contact-output "outputs/diagnostics/arm_actuator_${condition}_20260907.jsonl" \
    --arm-actuator-diagnostic
done
bash scripts/analyze_arm_actuator.sh \
  outputs/diagnostics/arm_actuator_present_20260907.jsonl \
  outputs/diagnostics/arm_actuator_absent_20260907.jsonl \
  --output-dir outputs/diagnostics/arm_actuator_analysis_20260907
```

Analysis reads the exact existing `arm_contact_present_20260907.jsonl` and
`arm_contact_absent_20260907.jsonl` baselines from `--baseline-dir` (default
`outputs/diagnostics`) and refuses to certify changed trajectories.
Artifacts in `outputs/diagnostics/arm_actuator_analysis_20260907/`:
`summary.json`, `per_joint.csv`, `response_present.png`, `response_absent.png`,
`pd_terms_present.png`, `pd_terms_absent.png`. New JSONL files above contain all
per-timestep signals/limits. Console logs:
`/tmp/arm_actuator_{present,absent,analysis}_20260907.log`.

Implementation: optional `arm_actuator_probe.py` integrated into the existing
`replay_arm_contact.py` path; new `analyze_arm_actuator.py`, root analysis wrapper,
and `tests/test_arm_actuator_analysis.py`. No normal training/controller path
was replaced. The probe calls original native setters exactly once and reads
parameters/state without extra simulation steps or actuator computations.

Regression suite **47/47 passed** (four new topology/interval/limit tests).
Both ON/OFF states and commands match baselines bit-for-bit. Approximate
P+D+FF agrees with the actuator field within 1.91e-6 N m floating-point error.

## Proposed single-variable experiment — subsequently executed separately

The subsequently authorized comparison is recorded in
[velocity-target comparison](ARM_VELOCITY_TARGET_COMPARISON.md). The original
ON/OFF diagnosis and logs above remain unchanged; the paragraph below records
the original proposal, not the status of that follow-up.

In the **OFF** recorded-command replay, change only the six submitted arm
velocity targets from zero to the already measured `v_path`. Keep position/hand
commands, gains, effort/velocity limits, dt, gravity and all physical settings
identical. Compare late wrist1/wrist3 tracking error and stage C with the saved
OFF baseline. This specifically tests the effect of velocity-target consistency
suggested by the P/D opposition; improvement would not prove adequate torque
capacity, nor guarantee contact stability. No such change was applied here.
