# User commands

Run from the repository root. Start with the [supported command table](../README.md#어떤-경로를-실행할-것인가)
and [architecture](../docs/architecture.md). For regressions use
`./scripts/run_tests.sh`; use Isaac runs for physics claims.

Use `./scripts/rl.sh --help` for the shared sequence/reference options. The
older `train_rb3_revo2_ppo.sh`, `play_rb3_revo2_ppo.sh`, and
`run_rl_*.sh` names are compatibility wrappers; they contain no independent
launcher logic.

## Experiments and failure reproduction (opt-in)

Read only the report for the question at hand. It defines exact inputs,
conditions, commands and existing evidence; use new output paths when rerunning.
`simple` mode alone is not the selected controller configuration.

| Question / entry point | Source of truth |
|---|---|
| Actual wrist/mount frame; `diagnose_wrist_frames.sh` | [Frame diagnosis](../docs/WRIST_FRAME_DIAGNOSIS.md) |
| Policy → IK → command → PhysX; `rl.sh play-arm --arm-execution-trace`, `analyze_arm_execution.sh` | [Execution trace](../docs/ARM_EXECUTION_DIAGNOSIS.md) |
| Identical commands, can contact ON/OFF; `rl.sh play-arm --arm-contact-replay`, `analyze_arm_contact.sh` | [Contact comparison](../docs/ARM_CONTACT_COMPARISON.md) |
| Effort provenance and velocity targets; `analyze_arm_actuator.sh` / `analyze_arm_policy_velocity.sh` | [Actuators](../docs/ARM_ACTUATOR_DIAGNOSIS.md), [paired policy evaluation](../docs/ARM_VELOCITY_CONTACT_POLICY_VALIDATION.md) |
| Frozen policy: floating/legacy/simple; `evaluate_mounted_interface.sh`, `analyze_mounted_interface.sh` | [Minimal interface](../docs/MINIMAL_MOUNTED_INTERFACE.md) |
| Original commands versus measured-motion targets; `arm_transfer_recovery.sh`, `analyze_transfer_recovery.sh` | [Completed recovery experiment](../docs/ARM_TRANSFER_RECOVERY.md) |
| Static/slow precision; `benchmark_arm_precision.sh`, `compare_arm_precision.sh` | [Precision benchmark](../docs/ARM_PRECISION_BENCHMARK.md) |
| Selected candidate + fast IK GUI; `play_arm_fast.sh` | [Opt-in fast execution and comparison video](../docs/ARM_REALTIME_EXECUTION.md) |

Completed sweep-only launchers were retired; individual evaluators, analyzers,
tests and evidence remain. See the [cleanup record](../docs/cleanup-plan.md#supported-path-cleanup-2026-09-07)
for exact Git recovery and deferred items. Training, assets and controllers are
not cleaned up by these command wrappers.
