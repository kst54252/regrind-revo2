# Current project status

Documentation/code-path review: 2026-09-08 (no new physics evaluation).
This records implemented paths and known limits. The latest
[cleanup validation record](cleanup-plan.md#validation-of-this-change)
distinguishes fresh baseline regression runs from historical experiments.

## Active pipeline

The supported default path is:

1. Read the second DexYCB camera and convert left-hand captures to the right-hand
   convention.
2. Retarget MANO21/tuna motion to a floating wrist and six Revo2 leaders.
3. Apply one camera-to-world transform and ground the tuna mesh on table `Z=0`.
4. Solve bounded RB3 IK frame by frame and create the combined 12-DoF reference.
5. Replay kinematically, train/evaluate the floating policy, or deploy that
   policy through online RB3 IK.

`scripts/run_pipeline.sh` orchestrates the offline path. Its sequence filter is
forwarded to retargeting/reference preparation, but preprocessing and the first
HTML gallery still enumerate all dataset directories. Leading-frame trimming
is controlled separately by `REGRIND_TRIM_SEQUENCE` and
`REGRIND_TRIM_LEADING_FRAMES`.

## Implemented

### Data, retargeting, and kinematics

- Second-camera loading, valid-frame extraction, right-hand conversion, and
  MANO21 ordering metadata.
- Revo2 integration with the upstream interaction-mesh/Laplacian retargeter.
- Six Revo2 leaders producing 21 semantic FK keypoints plus five mimic joints.
- Shared rigid world alignment for object, wrist, and MANO.
- Mounted-wrist RB3 FK/IK with bounds, warm starts, multi-seed solving, failure
  recording, and continuity metrics.
- Sequence-specific alignment is centralized in
  `prepare_isaac_references.py::ISAAC_ALIGNMENT`.

### Replay

- 12-DoF Isaac replay with MANO skeleton, tuna mesh, wrist/path markers, GUI
  controls, kinematic mode, and dynamic-object comparison modes.
- Shared pedestal/table geometry under `config/workcell/`.
- Offline strict-IK diagnostics and arm target-versus-measured analysis.

### RL and deployment

- Floating Revo2+tuna task with 12-D residual action: wrist SE(3) plus six hand
  leaders.
- Actor/critic observations, five physical fingertips, object-keypoint rewards,
  RSI, resets, latency/noise, dynamics randomization, pushes, and gravity
  curriculum.
- RSL-RL train/play integration and rollout export.
- Rigid XY reference placement with canonicalized policy observations.
- Offline rollout-to-RB3 conversion and online bounded-IK arm deployment.

See [RL_TASK.md](RL_TASK.md) and [ISAAC_SIM_REPLAY.md](ISAAC_SIM_REPLAY.md) for
current commands and validation boundaries.

The isolated [arm transfer recovery experiment](ARM_TRANSFER_RECOVERY.md)
(2026-09-07) executed fixed-command replay and frozen-policy comparisons. Its
opt-in configuration improved existing20 from19/20 to20/20 and distinct held-out20
from17/20 to20/20, with increased peak acceleration. Defaults were unchanged at
that experiment; see below for the subsequent user-approved promotion.

The subsequent [fast execution experiment](ARM_REALTIME_EXECUTION.md) introduced
the candidate and warm-first IK now used by the approved video profile.
Measured headless/GUI throughput was about .63x/.48x, not guaranteed real time.
Use [the command index](../scripts/README.md) to distinguish current evaluators
from completed diagnostic recipes. Retired launchers and rejected candidate
configs are recorded in the cleanup history; reusable experiment implementations
and contract tests remain. Offline analysis now lives in `tools/arm_diagnostics/`.

Current user-facing evaluation/capture defaults use the completed **10,000-update**
floating model; see [selection and override rules](RL_TASK.md#current-evaluation-checkpoint).
Previous benchmark counts below belong to the 5,000-update checkpoint, not the
new model. Historical artifacts remain unchanged. A later matched 40-placement
comparison with the 10k policy is recorded below; it is not a population success rate.

On 2026-09-09 the user approved the video controller plus compliant distal
contacts as the `play-arm` default after 40/40 task and lift/contact-proxy outcomes
with the original 10k policy. The previous strict-IK controller is retained by
`--arm-controller baseline`; floating defaults and original assets are unchanged.
`train-arm` and evaluation share `regrind/utils/arm_execution_config.py` and use
one environment. See [current commands and limits](RL_TASK.md#approved-video-controller-and-timed-transfer).
The separate [one-hour transfer](RL_TASK.md#one-hour-execution--2026-09-09) completed
2,679 updates / 64,296 transitions in 3,601 s. Paired old20 + heldout20 maintained
40/40 task and lift/hold outcomes, but mean object error increased 3.94→11.16 mm
and finger error .02235→.02880 rad. The final transfer model is therefore opt-in;
the default policy remains the original 10k model. Root regressions: 142 passed.

The [120 Hz smooth bounded-IK candidate](ARM_IK_SINGULARITY_FIX.md) subsequently
preserved 40/40 observed task/contact-proxy outcomes and reduced the worst sampled
wrist position error to about 6 mm. It allows approximate IK near singularities
and remains opt-in, not a general grasp-success guarantee or a new default.

## Incomplete or out of scope

- Real RB3/Revo2 communication, safety control, calibration, object tracking,
  latency measurement, and sim-to-real validation.
- Guaranteed dynamic grasp reliability in the assembled-arm scene.
- Tactile sensing, tuna-symmetry rewards, new policy architectures, or new RL
  algorithms.
- Revo2 scissors retargeting; Revo2 currently declares the tuna-can object path.
- A trained full-arm policy; online deployment uses the floating policy plus IK.
- Transfer of simulator gain experiments to real hardware.
- Opt-in arm fine-tuning distinguishes floating initialization from transfer
  resume and shares the selected mounted controller with paired evaluation.
  The [2026-09-09 100-update trial](RL_TASK.md#executed-transfer-validation--2026-09-09)
  improved the measured lift/contact proxy33/40→39/40; one new held-out drop and
  larger finger tracking error remain. This is not a convergence claim or a
  replacement for floating/arm defaults. Earlier 25-update trials are historical.
- An [opt-in last-phalanx compliant-contact trial](RL_TASK.md#executed-rubber-contact-comparison--2026-09-09)
  preserves friction and rigid assets. Floating retained hold40/40; mounted
  original-policy hold changed33→34/40, but a separate 100-update compliant
  transfer regressed34→25/40. That checkpoint is **not** a replacement baseline;
  material parameters remain uncalibrated. These results concern the previous
  strict-IK controller, not the subsequently approved video/rubber default.

## Preserved or uncertain paths

- LeapHand/WujiHand and the `--legacy-arm-rl` combined task remain registered,
  but their current training quality has not been revalidated.
- Duplicate RL aliases were removed; use `scripts/rl.sh` subcommands.
- Several upstream/manual diagnostic scripts have no internal caller but have
  not been proven unused.
- `regrind/source/regrind/test/` is outside `scripts/run_tests.sh` discovery.
- Two Revo2 keypoint JSON copies currently require manual synchronization.
- Output retention is mixed: some compact artifacts are tracked while HTML,
  checkpoints, and logs are generally ignored.
- `scripts/random_can_full_replay.sh` uses the shared current checkpoint but
  retains the older offline trimming/leveling workflow, separate from online play.
- Runtime Hydra overrides of materialized actuator dictionaries may not update
  all derived values; arm evaluation has explicit `play.py --rb3-*-scale`
  options pending config-lifecycle cleanup.
- Training accepts `--experiment_name` but the repository argument updater does
  not apply it; use `agent.experiment_name=NAME` until that separate source issue
  is fixed. Capture runs must not pollute normal latest-checkpoint selection.
- The offline pipeline does not regenerate the preferred stable RL reference;
  ordinary and stable references must not be assumed to contain the same frames.
- Some standalone CLI help retains old local-Y model correction / 180° wrist
  correction suggestions. Those are not required by the current verified
  camera/mount contract; follow the data/frame docs, not those legacy hints.
- Bare evaluator `--help` hits the installed AppLauncher required-argument
  preparse; supplying mode/checkpoint/output with help works. Actual evaluation
  is unaffected; see the cleanup record for the failed command and validation.

Cleanup-specific evidence and decisions live only in
[cleanup-plan.md](cleanup-plan.md).
