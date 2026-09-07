---
name: diagnose-regrind-failure
description: Diagnose a reproducible failure in DexYCB preprocessing, Revo2 retargeting/FK, RB3 IK/replay, Isaac simulation, or RL execution. Use when a command or runtime result behaves incorrectly; diagnosis alone does not authorize a fix.
---

# Diagnose a REGRIND failure

1. Record the exact command, input, expected result, observed result, and first
   failing stage. Read the relevant architecture section and only the affected
   subsystem document; check current status for a matching known limitation.
2. Trace the actual wrapper, registration, config, loader, and call chain. Check
   the nearest working path and the boundary contracts relevant to the symptom.
3. Reproduce with the smallest existing test or diagnostic command. Verify its
   options in the repository before running it; inspect only named artifacts.
4. Identify the earliest violated contract and affected scope. Report verified
   observations, likely root cause, remaining hypotheses, and the smallest
   fix/validation route separately.

Do not infer Isaac/PhysX contact behavior from simulator-independent tests.
