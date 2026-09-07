# User commands

Run from the repository root. Start with the [supported command table](../README.md#어떤-경로를-실행할-것인가)
and [architecture](../docs/architecture.md). For regressions use
`./scripts/run_tests.sh`; use Isaac runs for physics claims.

Use `./scripts/rl.sh --help` for the shared sequence/reference options. The
training, evaluation, zero-agent and debug entry points are consolidated in
`rl.sh train|play|zero|debug`. Old duplicate aliases were removed; see the
[migration table](../docs/cleanup-plan.md#readable-layout-cleanup).

`play_arm_candidate.sh` is the renamed experimental GUI launcher (not a real-time
guarantee). It preserves the previous transfer candidate by default; add
`--transfer-config config/experiments/rb3_smooth_bounded_ik.json` for the tested
120 Hz smooth bounded-IK candidate. Neither replaces `rl.sh play-arm`.

Offline `analyze_*` / `compare_*` launchers delegate to
[`tools/arm_diagnostics/`](../tools/arm_diagnostics/README.md). They require saved
experiment traces, not a running simulator.

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
| Selected candidate + fast IK GUI; `play_arm_candidate.sh` | [Opt-in fast execution and comparison video](../docs/ARM_REALTIME_EXECUTION.md) |
| 120 Hz wrist3-only gain comparison; existing evaluator/precision launchers (candidate rejected as replacement) | [Executed 40-placement comparison](../docs/ARM_IK120_IMPROVEMENT.md) |
| 120 Hz singularity-aware velocity/acceleration-bounded IK; `play_arm_candidate.sh ... --transfer-config config/experiments/rb3_smooth_bounded_ik.json` (opt-in, approximate pose) | [Verified 40-placement IK fix](../docs/ARM_IK_SINGULARITY_FIX.md) |

Completed sweep-only launchers were retired; individual evaluators, analyzers,
tests and evidence remain. See the [cleanup record](../docs/cleanup-plan.md#supported-path-cleanup-2026-09-07)
for exact Git recovery and deferred items. Training, assets and controllers are
not cleaned up by these command wrappers.
