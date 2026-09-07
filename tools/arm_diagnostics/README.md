# Offline arm diagnostics

Saved traces in → statistics/plots out. These modules do not launch Isaac.
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

Select one report in the [diagnostic index](../../scripts/README.md) for exact
input schemas and reproduction conditions. Historical traces/checkpoints may
need separate local copies; missing data does not make an analyzer obsolete.
