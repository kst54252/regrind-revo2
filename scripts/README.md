# User commands

Run from the repository root. Start with the [supported command table](../README.md#어떤-경로를-실행할-것인가)
and [architecture](../docs/architecture.md). For regressions use
`./scripts/run_tests.sh`; use Isaac runs for physics claims.

Use `./scripts/rl.sh --help` for the shared sequence/reference options. The
training, evaluation, zero-agent and debug entry points are consolidated in
`rl.sh train|play|zero|debug`. Old duplicate aliases were removed; see the
[migration table](../docs/cleanup-plan.md#readable-layout-cleanup).

Primary-sequence evaluation and comparison captures default to the shared
**10,000-update** floating checkpoint; see [model selection](../docs/RL_TASK.md#current-evaluation-checkpoint).
Use explicit historical checkpoints when reproducing dated reports. Direct
evaluators remain explicit; existing videos are not replaced by this default change.

`rl.sh play-arm` now selects the approved video controller plus compliant distal
contacts; `train-arm --transfer-init FLOATING.pt` shares that configuration.
Both use one environment. Floating train/play defaults remain unchanged.
Use `--arm-controller baseline` for the preserved strict-IK path and old transfer
checkpoints; use `--resume --checkpoint TRANSFER.pt` only for transfer continuation.
See [default execution, timed training and paired evaluation](../docs/RL_TASK.md#approved-video-controller-and-timed-transfer).

`play_arm_candidate.sh` is also retained for explicit historical/experimental
profiles (not a real-time guarantee). Without an override it uses the approved
video/rubber settings; add
`--transfer-config config/experiments/rb3_smooth_bounded_ik.json` for the tested
120 Hz smooth bounded-IK candidate. That bounded-IK variant is not the default.

To reproduce a preserved comparison video's actual policy run, use
`play_arm_candidate.sh NEW_OUTPUT --match-recording VIDEO_RUN_DIRECTORY`
(the directory containing the right panel's `metadata.json`). This explicitly
restores its checkpoint, controller and state bank instead of selecting today's
model; input hashes/runtime settings are checked. It runs live policy inference,
not recorded actions, and restores the recorded episode count. Edited .5× speed
and still-frame holds are not simulator timing. See the
[executed video/current-path comparison](../docs/ARM_REALTIME_EXECUTION.md#video-versus-current-path-diagnosis--2026-09-09).

Offline `analyze_*` / `compare_*` launchers delegate to
[`tools/arm_diagnostics/`](../tools/arm_diagnostics/README.md). They require saved
experiment traces, not a running simulator.

## Presentation media

### Retargeting presentation video (no RL or physics)

`bash scripts/record_retargeting_presentation.sh NEW_OUTPUT_DIRECTORY
--sequence 20200709_143747_left` renders MANO/Revo2 semantic skeletons beside
the real Revo2 USD meshes in Isaac RTX, with the same camera and source frames.
Requires the sequence's `world_trajectory.h5` and `revo2_retargeted.h5`, Isaac
GPU access and ffmpeg. Reuses existing FK/mimic and validates saved keypoints
and composed USD link poses. No checkpoint, arm IK or physical grasp is involved.
Add `--hide-model-keypoints` to hide the right-hand model's 21 marker meshes
without hiding the left comparison skeletons. The setting is session-only.
[Clean-model presentation](../outputs/visualizations/presentation/retargeting_clean_20260908/README.md).
Output: `retargeting_skeleton_vs_isaac.mp4` (1920×1080, 30 fps), raw panel videos,
sample PNGs and `metadata.json`. Source frames are displayed at 10 fps, followed
by a one-second final hold; the current 40-frame, 30 Hz demo produces 5 seconds.
The displayed 0.33x label assumes a 30 Hz source. These are kinematic renders,
not evidence of policy inference or successful dynamic grasping.

Existing presentation-style re-edit:
`outputs/visualizations/presentation/retargeting_dual_20260908_final/retargeting_skeleton_vs_isaac_presentation.mp4`.
It matches the comparison videos at 1280×720 with a `#3A3A3A` header/footer,
DejaVu Sans white titles and cyan/orange MANO/FK labels. The original 5 s,
150 frames at 30 fps, .33× playback and final hold are unchanged. Only outer
scene margins (18 px above/below the original 900 px content) were cropped before
uniform scaling and caption replacement; the original video remains intact.
Dimensions/frame count and a visual preview were checked. The capture command
above still produces the original white-caption 1920×1080 version.

### Workcell photos and training captures

Presentation captures of the existing arm workcell:

```bash
bash scripts/capture_workcell_photo.sh --headless --num_envs 1 --output NEW_SINGLE.png
bash scripts/capture_workcell_photo.sh --headless --num_envs 1 --extended-arm --output NEW_EXTENDED.png
bash scripts/capture_workcell_photo.sh --headless --num_envs 16 --output NEW_PARALLEL.png
bash scripts/record_arm_comparison.sh NEW_PAIR_DIRECTORY
bash scripts/compose_arm_comparison.sh NEW_PAIR_DIRECTORY --floating-style
bash scripts/record_floating_comparison.sh NEW_FLOATING_PAIR_DIRECTORY
```

Capture overrides: `record_arm_comparison.sh NEW_DIR CHECKPOINT`, or
`record_floating_comparison.sh CHECKPOINT NEW_DIR`. New output directories are
required so that previous recordings remain intact.

Clean/high-resolution floating footage: prepend `--presentation-hq` to both
`record_floating_comparison.sh` and `compose_floating_comparison.sh`. The capture
hides only Revo2 keypoint meshes in the session layer and records at 3840×2160
using `env.video_recorder.window_width/window_height` (viewer resolution alone
does not control this installed recorder). The composer emits 1920×1080 at .5×,
3.433333 s, without its verified black first frame. Normal play/training defaults
are unchanged; `play --hide-revo2-keypoints` is also an explicit visual-only option.
[Executed comparison and exact inputs](../outputs/visualizations/comparisons/isaac_floating_hq_20260908_take2/README.md).

Photos use the normal online-play reset plus one zero-residual control step to
flush actual articulation transforms to the renderer (not the stale USD rest
pose). Existing geometry and task settings are reused; only camera/resolution
are changed. Parallel arm photos are not evidence of parallel arm PPO training.
`--extended-arm` instead initializes only the six RB3 joints to zero once and
renders after one physical step, keeping the reference hand posture. Its wider
camera includes the raised hand. This photo-only pose is not used for evaluation.

Actual floating-hand PPO capture uses `rl.sh train --num_envs 16 --max_iterations 13
--headless --video --video_length 300 --video_interval 100000 --resume
--checkpoint CHECKPOINT --run_name presentation_capture
agent.experiment_name=presentation_capture`, with camera overrides as needed.
This performs real updates in a separate run, not frozen-policy evaluation.
Use the Hydra `agent.experiment_name` override: the repository CLI currently
accepts `--experiment_name` but does not apply it in `cli_args.update_rsl_rl_cfg`.
Keep capture checkpoints out of the normal experiment's latest-run selection.

Recent local deliverables and their exact edit/capture conditions:
[workcell/parallel training/mounted comparison](../outputs/visualizations/presentation/robot_media_20260908/README.md)
and [floating comparison](../outputs/visualizations/comparisons/isaac_floating_20260907_take2/README.md).
The latest requested mounted edit is 0.5×/5 s; `--floating-style` itself still
produces the preserved 0.25×/10 s version. Media/checkpoints need separate copies
when absent in a fresh clone. Video playback rate is not live simulator throughput.

## Experiments and failure reproduction (opt-in)

Read only the report for the question at hand. It defines exact inputs,
conditions, commands and existing evidence; use new output paths when rerunning.
`simple` mode alone is not the selected controller configuration.
Reports are dated experiment records: their test counts, line numbers and
measurements are not assertions about the latest run. Diagnostic JSON/JSONL
and temporary `/tmp` logs may need separate backups; follow each report's
reproduction requirements. Retired presets require the documented Git revision.

| Question / entry point | Source of truth |
|---|---|
| Actual wrist/mount frame; `diagnose_wrist_frames.sh` | [Frame diagnosis](../docs/WRIST_FRAME_DIAGNOSIS.md) |
| Policy → IK → command → PhysX; `rl.sh play-arm --arm-controller baseline --arm-execution-trace`, `analyze_arm_execution.sh` | [Execution trace](../docs/ARM_EXECUTION_DIAGNOSIS.md) |
| Identical commands, can contact ON/OFF; `rl.sh play-arm --arm-controller baseline --arm-contact-replay`, `analyze_arm_contact.sh` | [Contact comparison](../docs/ARM_CONTACT_COMPARISON.md) |
| Effort provenance and velocity targets; `analyze_arm_actuator.sh` / `analyze_arm_policy_velocity.sh` | [Actuators](../docs/ARM_ACTUATOR_DIAGNOSIS.md), [paired policy evaluation](../docs/ARM_VELOCITY_CONTACT_POLICY_VALIDATION.md) |
| Frozen policy: floating/legacy/simple; `evaluate_mounted_interface.sh`, `analyze_mounted_interface.sh` | [Minimal interface](../docs/MINIMAL_MOUNTED_INTERFACE.md) |
| Last-phalanx compliant-contact approximation (opt-in); evaluator / `rl.sh train-arm` with `--fingertip-contact-config` | [Scope, caveats and commands](../docs/RL_TASK.md#opt-in-last-phalanx-rubber-approximation) |
| Original commands versus measured-motion targets; `arm_transfer_recovery.sh`, `analyze_transfer_recovery.sh` | [Completed recovery experiment](../docs/ARM_TRANSFER_RECOVERY.md) |
| Static/slow precision; `benchmark_arm_precision.sh`, `compare_arm_precision.sh` | [Precision benchmark](../docs/ARM_PRECISION_BENCHMARK.md) |
| Selected candidate + fast IK GUI; `play_arm_candidate.sh` | [Opt-in fast execution and comparison video](../docs/ARM_REALTIME_EXECUTION.md) |
| 120 Hz wrist3-only gain comparison; existing evaluator/precision launchers (candidate rejected as replacement) | [Executed 40-placement comparison](../docs/ARM_IK120_IMPROVEMENT.md) |
| 120 Hz singularity-aware velocity/acceleration-bounded IK; `play_arm_candidate.sh ... --transfer-config config/experiments/rb3_smooth_bounded_ik.json` (opt-in, approximate pose) | [Verified 40-placement IK fix](../docs/ARM_IK_SINGULARITY_FIX.md) |

Completed sweep-only launchers were retired; individual evaluators, analyzers,
tests and evidence remain. See the [cleanup record](../docs/cleanup-plan.md#supported-path-cleanup-2026-09-07)
for exact Git recovery and deferred items. Training, assets and controllers are
not cleaned up by these command wrappers.
