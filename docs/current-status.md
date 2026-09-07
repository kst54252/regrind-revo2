# Current project status

Last documentation audit: 2026-09-06. This records implemented paths and known
limits; it is not evidence of a fresh simulator or training run.

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
from17/20 to20/20, with increased peak acceleration. Normal deployment defaults
remain unchanged; see that report for limitations and exact reproduction inputs.

## Incomplete or out of scope

- Real RB3/Revo2 communication, safety control, calibration, object tracking,
  latency measurement, and sim-to-real validation.
- Guaranteed dynamic grasp reliability in the assembled-arm scene.
- Tactile sensing, tuna-symmetry rewards, new policy architectures, or new RL
  algorithms.
- Revo2 scissors retargeting; Revo2 currently declares the tuna-can object path.
- A trained full-arm policy; online deployment uses the floating policy plus IK.
- Transfer of simulator gain experiments to real hardware.

## Preserved or uncertain paths

- LeapHand/WujiHand and the `--legacy-arm-rl` combined task remain registered,
  but their current training quality has not been revalidated.
- Compatibility shell wrappers intentionally delegate to `scripts/rl.sh`.
- Several upstream/manual diagnostic scripts have no internal caller but have
  not been proven unused.
- `regrind/source/regrind/test/` is outside `scripts/run_tests.sh` discovery.
- Two Revo2 keypoint JSON copies currently require manual synchronization.
- Output retention is mixed: some compact artifacts are tracked while HTML,
  checkpoints, and logs are generally ignored.
- `scripts/random_can_full_replay.sh` has a dated default checkpoint; pass an
  explicit `--checkpoint` when model selection matters.
- Runtime Hydra overrides of materialized actuator dictionaries may not update
  all derived values; arm evaluation has explicit `play.py --rb3-*-scale`
  options pending config-lifecycle cleanup.

Cleanup-specific evidence and decisions live only in
[cleanup-plan.md](cleanup-plan.md).
