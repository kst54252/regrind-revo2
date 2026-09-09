# Floating-hand Isaac comparison

For the newer keypoint-free Full HD render of the same physical trajectories,
see [the 4K recapture](../isaac_floating_hq_20260908_take2/README.md).

Latest correction to `isaac_floating_retargeting_vs_residual_rl_16x9_4s.mp4`:
removed its 17 leading black scene frames (0.566667 s), including the initial
hold of an unrendered frame. It is now 103 frames / 3.433333 s at 30 fps;
the filename is retained. Motion speed and final hold are unchanged. First-frame
preview and scene-only black detection passed. A matching `_poster.png` is saved
for explicit slide/video posters. Pre-trim copy:
`outputs/visualizations/.black_intro_backup_0vX5As/` relative to repository root.
The 4-second duration below records the earlier layout edit, before this trim.

Original edit: `isaac_floating_retargeting_vs_residual_rl.mp4`, 1920x540, 30 fps,
300 frames / 10.000 s. Actual Isaac-rendered floating Revo2 and dynamic tuna,
no arm, no kinematic object trajectory, no controller or policy changes.

The later `isaac_floating_retargeting_vs_residual_rl_0p5x.mp4` doubles playback
to 0.5× (1920x540, 30 fps, 150 frames / 5 s), including the presentation holds.
Its speed label is updated; simulation data and the original edit are preserved.
The reproduction composer below still produces the original 0.25× edit.

Presentation edit: `isaac_floating_retargeting_vs_residual_rl_16x9_4s.mp4`,
1280×720 (16:9), 30 fps, 120 frames / 4 s. It retains the first 120 frames of
the .5× edit and removes only its final one second of frozen footage; initial
hold and motion speed are unchanged. Each original panel is cropped to 480×432
at local (240,64), uniformly scaled to 640×576, and placed beneath new titles
on a 1280×720 canvas (104 px header, 40 px footer). Side-by-side order and .5×
label are preserved. ffprobe and visual previews checked; original clips remain.

Left: reference targets with zero residual action. Right: live model_4999 policy.
Checkpoint: `logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`.
Reference: stable 38-frame `20200709_143747_left`. Seed 42, fixed can placement.
Both use the same camera and original floating-task physics/controller settings.

HDF5 state logs contain initialization + 37 physical control steps at 30 Hz.
Raw videos each contain 37 rendered frames, 1.233333 s (1280x720).
Final edit uses identical 0.25x playback, 1-second initial hold and final-frame
padding to 10 s. Holds are labelled; padded frames are not continued simulation.
Raw videos and HDF5 data are retained alongside the final edit.

Exact initial equality verified for wrist position/quaternion, leader joints,
object position/quaternion. Initial object XYZ: [.400000006, ~0, .0126360003] m.
Zero residual maximum can rise: .00005383 m; final XYZ
[.39999983, .0000000207, .012635652] m.
RL maximum/final can rise: .22021426 m; final XYZ
[.37834734, .00885903, .23285027] m.
This is one illustrative trial, not a general grasp success-rate estimate.

For equal-duration footage, failure/timeout resets are disabled via explicit
recording-only overrides in BOTH conditions. The reference-end term remains
available to the original reward and capture stops after 38 states. Neither
condition resets during this recording. Normal training/play defaults are unchanged.

Reproduce using a fresh directory:

```bash
bash scripts/record_floating_comparison.sh \
  logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  outputs/visualizations/comparisons/isaac_floating_repeat
bash scripts/compose_floating_comparison.sh \
  outputs/visualizations/comparisons/isaac_floating_repeat
```

The previous `isaac_floating_20260907` attempt stopped with a reward lookup
error before a usable clip because a required termination term was removed.
Its log is preserved; the corrected take2 recordings completed normally.
