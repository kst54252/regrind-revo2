# Revo2 kinematics and semantic keypoints

This directory contains the GUI-independent six-DoF Revo2 FK implementation,
its 21 semantic keypoints, and an interactive joint-slider viewer. Project
tests are collected in the top-level `tests/` directory.

```bash
# Run all tests from the project root.
./scripts/run_tests.sh

# Open the interactive FK viewer.
python tools/revo2_kinematics/visualize_revo2_kinematics.py

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
```

Set `REVO2_PROJECT_ROOT` before running the Isaac Sim export/validation scripts
if the repository is moved from its current path.
