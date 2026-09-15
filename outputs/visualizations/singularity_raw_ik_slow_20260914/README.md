# Raw IK singularity: slow kinematic view (2026-09-14)

- Episode 11, source steps 108–132 from `outputs/diagnostics/ik120_improvement_20260907/old20_baseline` (historical 5k policy).
- Use **raw solver q**, not rate-clipped q_cmd or recorded actual arm q.
- All 25 original arm/leader command knots are preserved without angle wrapping or branch changes.
- Between knots: joint-linear geometry, quintic scalar timing with zero endpoint velocity/acceleration. Motion 12.25 s plus 1.5 s holds at each end, 60 Hz, 916 samples.
- Maximum sampled speed 0.645983 rad/s, acceleration 1.99009 rad/s². These are visualization trajectory bounds, not measured actuator specifications.
- **Kinematic only**: physics paused, no policy/IK inference, no contact/grasp evaluation. Can poses are recorded post-step visual poses. The loop return to the first frame is a reset, not another singularity movement.
- Wrist reference in the file is FK of the interpolated q, explicitly for geometric readback validation. Original IK targets are stored separately. Between-knot deviation from their interpolated pose is at most 0.174 mm / 0.001229 rad; exact continuous Cartesian tracking is not claimed.

From repository root:

```bash
./scripts/run_isaac_replay.sh \
  --trajectory outputs/visualizations/singularity_raw_ik_slow_20260914/raw_ik_slow_reference.npz \
  --validation-output outputs/visualizations/singularity_raw_ik_slow_20260914/validation_gui.npz \
  --loop --no-demo-skeleton --width 1600 --height 1000 \
  --camera-eye 1.2 -1.5 0.85 --camera-target 0.18 0 0.15
```

Use the Replay window's Play/Pause/Reset/Go buttons. The generated `prepare.py` records preparation/provenance and refuses to overwrite its existing trajectory.

Executed headless Isaac check: 916/916 frames, no problem frames. Maximum arm/hand/follower readback error 1.20e-7 / 2.99e-8 / 1.50e-8 rad; FK wrist position error 3.52e-7 m, orientation error 9.68e-7 rad. This validates kinematic reconstruction, not controller precision. Details: `headless.log`, `validation_headless.npz`, `preparation.json`.

Existing regressions: 166 passed; optional report-path parser and unchanged default checks passed. Existing baseline report retained.
