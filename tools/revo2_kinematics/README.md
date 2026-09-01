# Revo2 kinematics and semantic keypoints

This directory contains the GUI-independent six-DoF Revo2 FK implementation,
its 21 semantic keypoints, and an interactive joint-slider viewer. Project
tests are collected in the top-level `tests/` directory.

```bash
# Run all tests from the project root.
./scripts/run_tests.sh

# Open the interactive FK viewer.
python tools/revo2_kinematics/visualize_revo2_kinematics.py

# Build a single-file browser viewer for all active/mimic joints and 21 points.
python tools/revo2_kinematics/visualize_revo2_keypoints_html.py

# Save a static validation image without opening a window.
python tools/revo2_kinematics/visualize_revo2_kinematics.py \
    --save outputs/visualizations/fk/revo2_fk_visualization.png \
    --no-show
```

Python interface:

```python
from tools.revo2_kinematics import Revo2Kinematics

fk = Revo2Kinematics()
keypoints = fk.get_keypoints([0.0] * 6)  # (21, 3)
lower, upper = fk.get_joint_limits()
mimic_q = fk.get_mimic_joint_positions([0.0] * 6)  # (5,)
```

The generated HTML is written to
`outputs/visualizations/revo2_kinematics/revo2_keypoints_mimic_interactive.html`.
It embeds all FK data and rendering code, so the one HTML file can be copied to
another machine and opened without this repository or an internet connection.

Set `REVO2_PROJECT_ROOT` before running the Isaac Sim export/validation scripts
if the repository is moved from its current path.
