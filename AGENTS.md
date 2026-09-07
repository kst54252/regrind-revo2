# Repository guidance

This repository adapts REGRIND to DexYCB, the six-actuator Revo2 hand, the
RB3-730 arm, Isaac replay, and floating-hand residual PPO for a tuna can.

## Navigate

- Read only the relevant section of [architecture](docs/architecture.md) first.
- Consult [current status](docs/current-status.md) when active, legacy, or
  incomplete behavior matters.
- Open [data pipeline](docs/DATA_PIPELINE.md), [RL task](docs/RL_TASK.md), or
  [Isaac replay](docs/ISAAC_SIM_REPLAY.md) only for that subsystem.
- Use [cleanup plan](docs/cleanup-plan.md) only for repository cleanup work.
- Use the [command/diagnostic index](scripts/README.md) for baseline vs opt-in
  experiments; do not load every historical report.

## Map

- `scripts/`: maintained user entry points; compatibility wrappers delegate here.
- `tools/`: preprocessing, transforms, standalone kinematics/IK, and replay.
- `regrind/source/regrind/regrind/`: package, assets, MDP terms, and task configs.
- `tests/`: primary simulator-independent regressions.
- `config/workcell/`: shared workcell geometry and mount configuration.

## Always follow

- Search with `rg`/`rg --files` before opening targeted sections of large files.
- Do not scan datasets, logs, checkpoints, generated outputs, binaries, meshes,
  or USD payloads unless the task requires a specific artifact. Treat `dataset/`
  as read-only.
- Preserve unrelated working-tree changes and explicit data conventions:
  quaternion metadata, RB3-then-Revo2 joint order, and six Revo2 actuators
  producing 21 semantic keypoints.
- Put repeated workflows behind root `scripts/`; do not add logic to
  compatibility wrappers.
- Main commands: `scripts/run_pipeline.sh`, `scripts/rl.sh train|play|play-arm`,
  `scripts/run_isaac_replay.sh`. Pipeline generation is not a smoke test.
- Preserve observation/action normalization and order, phase/dt, command vs
  actual state, and default controller/reward/termination semantics in cleanup.
- New experiments stay opt-in. Retire them only after reference checks and a
  recoverable snapshot; retain unique regression tests and reproduction evidence.
- Validate proportionally. Use targeted tests, then `./scripts/run_tests.sh` when
  dependencies allow; Isaac/physics behavior requires an Isaac run.
- Before handoff inspect the focused diff and status, and run `git diff --check`.
