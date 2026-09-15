# Offline arm diagnostics

Saved traces in → statistics/plots out, plus offline IK workspace screening.
Exception: `evaluate_semicircle` orchestrates real Isaac policy evaluations
through the existing mounted evaluator; it is not an offline-only smoke test.
Runtime capture, replay, FK/IK and benchmarks remain in `../rb3_revo2_ik/`.
Some shared pose/error helpers are imported by runtime capture code.

Use root `scripts/analyze_*.sh` / `scripts/compare_*.sh`, or run from the repository
root with `python -m tools.arm_diagnostics.<module> --help`.

| Analysis | Modules |
|---|---|
| IK / command / actual tracking | `analyze_ik_tracking`, `analyze_arm_execution`, `analyze_arm_tracking` |
| Actuator and contact isolation | `analyze_arm_actuator`, `analyze_arm_contact`, `analyze_arm_velocity` |
| Live policy and transfer comparisons | `analyze_mounted_interface`, `analyze_arm_policy_velocity`, `analyze_transfer_recovery`, `analyze_actual_motion` |
| Static/slow precision | `analyze_arm_precision` (library), `compare_arm_precision`, `compare_arm_command_precision` |
| Tabletop motion screening | `measure_tabletop_region` (reference XY sweep using existing mounted-wrist IK; not collision/grasp validation) |
| Tabletop branch continuity | `compare_tabletop_branches`; launcher `scripts/compare_tabletop_branches.sh` (offline look-ahead; [results and limits](../../docs/TABLETOP_OPERATING_REGION.md)) |
| Tabletop approach direction | `search_tabletop_yaw`; launcher `scripts/search_tabletop_yaw.sh` (fixed can center, whole-reference yaw and dense IK check) |
| Recorded singularity methods | `compare_singularity_methods`; launcher `scripts/compare_singularity_methods.sh` ([same-target versus rotated-task results](../../docs/ARM_IK_SINGULARITY_FIX.md#2026-09-11-같은-특이점-입력으로-여러-회피-방법-재비교)) |
| Analytic all-branch / SQP comparison | `compare_analytic_sqp`; launcher `scripts/compare_analytic_sqp.sh` ([executed results and failure semantics](../../docs/ARM_IK_SINGULARITY_FIX.md#2026-09-15-analytic-all-branch-ik와-sqp-비교)) |
| Whole-table SQP screening | `scan_tabletop_sqp`; launcher `scripts/scan_tabletop_sqp.sh` ([fixed-yaw recorded-target map](../../docs/TABLETOP_OPERATING_REGION.md#2026-09-15-책상-전체-sqp-조사)) |
| Semicircle actual grasps | `evaluate_semicircle`; launcher `scripts/evaluate_semicircle.sh` (fixed spatial denominator, existing frozen-policy evaluator; [executed results](../../docs/TABLETOP_OPERATING_REGION.md#2026-09-15-반원-영역-실제-파지-평가)) |

Select one report in the [diagnostic index](../../scripts/README.md) for exact
input schemas and reproduction conditions. Historical traces/checkpoints may
need separate local copies; missing data does not make an analyzer obsolete.
