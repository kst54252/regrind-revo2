# Retargeting presentation — model keypoints hidden

Final: `retargeting_skeleton_vs_isaac_presentation.mp4` (1280×720, 30 fps,
150 frames / 5 s; .33× playback and one-second final hold unchanged).
First-frame image: same stem plus `_poster.png`. No black scene intro detected.

Right panel hides 21 `kp_*` marker roots below `/Presentation/Hand` using USD
session visibility only. Left MANO/Revo2 skeleton points and bones remain visible.
Reuses the existing kinematic FK/mimic renderer; no policy or physics is run,
and no source asset or trajectory is edited. Original videos remain intact.

Source world SHA256, all 40 frame indices, camera eye/target, playback rate and
hold match `retargeting_dual_20260908_final/metadata.json`. Per-frame USD/FK
assertions passed; hidden paths are recorded in the new `metadata.json`.
123 root regressions passed, including explicit model-scope/skeleton preservation.

Capture from repository root:

```bash
bash scripts/record_retargeting_presentation.sh NEW_OUTPUT_DIRECTORY \
  --sequence 20200709_143747_left --hide-model-keypoints
```

The launcher produces the original white-caption composition. The final re-edit
uses the established presentation layout: crop 1920×864 at (0,118), uniformly
scale to 1280×576, 104 px header/40 px footer, background #3A3A3A, DejaVu Sans
28/24/19 px and cyan/orange MANO/FK labels. H.264 CRF 14, slow, yuv420p, faststart.
Only spatial formatting and marker visibility changed, not trajectory timing.
