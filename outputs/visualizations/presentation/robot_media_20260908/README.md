# Presentation media — 2026-09-08

These are actual Isaac Sim captures, not generated illustrations. Commands and
checkpoint paths below are relative to the repository root; media filenames in
the table are relative to this folder.

| Deliverable | File | Verified content |
|---|---|---|
| Single RB3 + Revo2 + workcell | `rb3_revo2_single_workcell.png` | 1920×1080; normal reference reset plus one zero-residual control step |
| Fully extended RB3 + Revo2 | `rb3_revo2_fully_extended.png` | 1920×1080; six arm joints initialized to zero once, one physics step; wider camera includes the raised hand |
| 16 RB3 + Revo2 workcells | `rb3_revo2_parallel_workcells.png` | 1920×1080; 16 distinct environment origins; same existing scene/config |
| Actual parallel PPO training | `parallel_ppo_training_10s.mp4` | 1280×720, 30 fps, 300 frames, 10 s; 16 floating hands; normal 1× simulation-time playback |
| Parallel PPO, presentation captions | `parallel_ppo_training_10s_presentation.mp4` | 1280×720, 30 fps, 299 frames, 9.966667 s; first black frame removed; speed unchanged |
| Mounted zero-residual vs RL, wide edit | `arm_comparison/isaac_rb3_retargeting_vs_residual_rl_0p5x.mp4` | 1920×540, 30 fps, 150 frames, 5 s; same .5× playback on both panels; final-frame holds |
| Mounted comparison, 16:9 presentation | `arm_comparison/isaac_rb3_retargeting_vs_residual_rl_16x9_4s.mp4` | 1280×720, 30 fps, 103 frames, 3.433333 s; .5× motion unchanged; black intro removed after final-hold edit |

Latest correction: the presentation training clip contained one black scene frame;
the mounted and floating 16:9 comparisons each contained 17 black scene frames
(the initial hold repeated an unrendered first frame). Removed exactly 1/17/17
leading frames with `trim=start_frame=N,setpts=PTS-STARTPTS`; filenames retained.
Title bands were excluded from black detection. All three now start with a visible
scene at timestamp zero, with no scene-black intervals detected. No motion or final
hold was removed in this correction. The retargeting-only presentation starts
normally and was not trimmed. Each presentation video now has a `_poster.png`
first-frame image for players/slides that require an explicit poster. MP4 alone
cannot force every application's pre-play display. Pre-trim copies are recoverable
in `outputs/visualizations/.black_intro_backup_0vX5As/` (local).
The durations in the edit history below describe the pre-trim versions.

The earlier `arm_comparison/isaac_rb3_retargeting_vs_residual_rl_floating_style.mp4`
is preserved at .25×, 300 frames / 10 s. The latest edit doubles that video's
playback rate, including holds, and replaces the speed caption with 0.5×.
It is a re-edit of the same simulation, not a new grasp evaluation.

The subsequent 16:9 edit preserves frames 0–119 of the 5-second .5× version:
only the final 30 held frames are removed, not moving frames. Both panels use
the same spatial crop: 480×432 at local (240,64), uniformly scaled to 640×576.
Panels remain side-by-side with a 104-pixel title area and 40-pixel footer;
no stretching or independent retiming. The crop removes outer scene margins,
so the complete table is not shown. Preview framing and output duration/frame
count were checked. Source media remain intact. The matching floating edit is
`../../comparisons/isaac_floating_20260907_take2/isaac_floating_retargeting_vs_residual_rl_16x9_4s.mp4`.

The training architecture is floating-hand PPO, then online arm IK deployment.
The parallel arm photograph is NOT a claim of PPO training all those arms.
The training video is NOT a frozen-policy rollout: 13 real PPO updates were
executed (iterations 4999–5011), resuming the existing `model_4999.pt` into a
separate capture run. Original checkpoint/reference/controller files are unchanged.
The raw 301-frame training capture was trimmed to exactly 300 frames, without
changing its playback rate. Added titles identify the actual embodiment.

Presentation-caption edit: top `Floating Revo2 + Tuna Can | Parallel Simulation Training`;
bottom `Reference State Initialization | Dynamics & Observation Randomization | Gravity Curriculum`.
The capture's saved `params/env.yaml` enables RSI, dynamics/observation randomization
and the gravity curriculum event. These captions describe the training setup;
the 10-second clip does not demonstrate progression through all gravity stages.
The original video is preserved. The edit replaces only the existing 62-pixel
top and 40-pixel bottom caption bands, using DejaVu Sans at 28/21 px, respectively.
Verified by ffprobe: 1280×720, 30 fps, 300 frames, 10.000 s; preview checked for
legibility and clipping. This is not a new capture with the 10,000-update model.

Source checkpoint for both capture and paired evaluation:
`logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`.
The paired arm capture uses the existing `scripts/record_arm_comparison.sh`
transfer-recovery candidate/defaults, not an unannounced controller change.
The comparison verifier confirmed identical physical initial states and matching
controller/physics, all-zero residual actions on the left and frozen-policy
inference on the right. Left terminated with `object_deviation`; right reached
the existing success criterion. This is one illustrated placement, not a new
success-rate estimate. Terminal auto-reset images were removed and holds labelled.

Training logs/checkpoints were moved out of the normal floating experiment's
latest-run selection into:
`logs/rsl_rl/presentation_capture/2026-09-08_00-25-29_floating_parallel_20260908/`.
The original `training_capture.log` records the pre-move output path. The CLI's
`--experiment_name` flag was accepted but ignored by the repository argument
adapter; future captures should use `agent.experiment_name=presentation_capture`.
No RL implementation was changed to work around that logging-path issue.

Photo metadata is beside each PNG; raw comparison clips, policy/physics traces
and `comparison_floating_style.json` remain in `arm_comparison/`. Earlier photo
attempts remain for debugging but are not the deliverables listed above.
Tests: 114 passed; shell syntax, video dimensions/duration and visual previews checked.

Reproduce photos with `bash scripts/capture_workcell_photo.sh --headless
--num_envs 1 --output NEW_SINGLE.png` (or `--num_envs 16`). Add `--extended-arm`
for the fully extended photo; it is a capture pose, not a changed evaluation reset.
Reproduce the original .25× comparison
with `bash scripts/record_arm_comparison.sh NEW_DIR
logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`, then
`bash scripts/compose_arm_comparison.sh NEW_DIR --floating-style`.
Omitting the checkpoint now selects the 10,000-update model, not the model used
in these preserved recordings.
