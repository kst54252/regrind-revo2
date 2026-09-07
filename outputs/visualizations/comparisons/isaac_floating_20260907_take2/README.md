# Floating-hand Isaac comparison

Original edit: `isaac_floating_retargeting_vs_residual_rl.mp4`, 1920x540, 30 fps,
300 frames / 10.000 s. Actual Isaac-rendered floating Revo2 and dynamic tuna,
no arm, no kinematic object trajectory, no controller or policy changes.

The later `isaac_floating_retargeting_vs_residual_rl_0p5x.mp4` doubles playback
to 0.5× (1920x540, 30 fps, 150 frames / 5 s), including the presentation holds.
Its speed label is updated; simulation data and the original edit are preserved.
The reproduction composer below still produces the original 0.25× edit.

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
