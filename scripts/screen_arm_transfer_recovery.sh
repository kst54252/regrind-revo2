#!/usr/bin/env bash
# Bounded fixed diagnostic suite. New root directory required by child runners.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
common=(--mode simple --stage recovery --recovery-kind R1 --headless
  --actual-source outputs/diagnostics/arm_transfer_recovery/F0
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt)
run() {
  local label="$1"
  shift
  bash scripts/arm_transfer_recovery.sh "${common[@]}" "$@" \
    --output "outputs/diagnostics/arm_transfer_recovery/${label}" > "/tmp/recovery_${label}.log" 2>&1
  test -f "outputs/diagnostics/arm_transfer_recovery/${label}/metadata.json"
}
run holds_base --episodes 1 --recovery-no-can --recovery-holds
run motion_base --episodes 1 --recovery-no-can
run slow_base --episodes 1 --recovery-no-can --recovery-speed 4
run holds_c3 --episodes 1 --recovery-no-can --recovery-holds --arm-gains-key c3
run motion_c3 --episodes 1 --recovery-no-can --arm-gains-key c3
run R2_c1_screen --episodes 5 --arm-gains-key c1
run R2_c2_screen --episodes 5 --arm-gains-key c2
run R2_c3_screen --episodes 5 --arm-gains-key c3
