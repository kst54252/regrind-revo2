# Floating comparison — clean 4K capture / Full HD presentation

Deliverable: `isaac_floating_retargeting_vs_residual_rl_presentation_hq.mp4`.
1920×1080, 16:9, 30 fps, 103 frames / 3.433333 s, same .5× playback on both
panels and final-frame hold. Poster: same stem plus `_poster.png`.

Re-recorded actual Isaac physics and frozen-policy inference, not an AI edit or
kinematic imitation. Uses the **same historical 5,000-update checkpoint** as the
existing floating comparison, not the new 10,000-update default:
`logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`.
Seed 42, fixed initial placement, stable 38-state primary reference, existing
recording-only termination overrides unchanged. No gains, mass, collisions,
policy, physics dt or reference changes.

Changes are visual-only: 21 `Robot/.../kp_*` marker roots hidden through USD
session-layer visibility, without removing prims or editing source assets.
Raw clips are 3840×2160, 37 frames at 30 fps. Each panel crops 1920×1728 at
(960,256), downsamples to 960×864 with Lanczos and gets the matching dark-gray
title/footer. Final encode: H.264 CRF 14, slow preset, yuv420p, faststart.
One unrendered first frame removed on both sides before .5× retiming; no black
intro remains. Existing videos are preserved.

Validation:

- All 38 samples of wrist position/quaternion, six hand joints, object
  position/quaternion and floating actions are bitwise equal to the prior
  `isaac_floating_20260907_take2` HDF5 records for each condition (max difference 0).
- Left/right physical initial states match exactly.
- Maximum can rise: zero residual 0.000053827 m; RL 0.220214263 m. Same illustrative
  outcome as before, not a new estimate of general grasp success.
- Both runtime logs list 21 hidden marker roots. First/mid-video previews,
  ffprobe, scene-only black detection, and 122 root regression tests passed.
- Initial capture attempt in `isaac_floating_hq_20260908/` stopped before recording
  due to a presentation-helper import path; corrected to the installed package.
  Failure log preserved, no original artifacts overwritten.

Reproduce from repository root with a new directory:

```bash
bash scripts/record_floating_comparison.sh --presentation-hq \
  logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  NEW_CAPTURE_DIRECTORY
bash scripts/compose_floating_comparison.sh --presentation-hq NEW_CAPTURE_DIRECTORY
```

Omit the explicit checkpoint to capture the shared current model instead; that
would be a new policy comparison, not reproduction of this video. The HQ composer
accepts the verified 37-frame 4K capture only and rejects unexpected inputs.
