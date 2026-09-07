#!/usr/bin/env bash
# Final two candidates. No further gain, limit, policy or reward changes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
for tau in 0.1 0.075; do
  label="live_response_${tau}_screen"
  bash scripts/arm_transfer_recovery.sh --mode simple --stage grasp --episodes 5 \
    --arm-gains-key c3 --arm-velocity-path --arm-response-physics --response-tau "$tau" \
    --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
    --output "outputs/diagnostics/arm_transfer_recovery/${label}" --headless > "/tmp/recovery_${label}.log" 2>&1
  test -f "outputs/diagnostics/arm_transfer_recovery/${label}/metadata.json"
done
