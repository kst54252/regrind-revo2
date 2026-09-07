#!/usr/bin/env bash
# Same initial state/controller/dynamic object; only residual action source differs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
root="${1:?Usage: bash scripts/record_arm_comparison.sh NEW_OUTPUT_DIRECTORY}"
[[ ! -e "$root" ]] || { echo "Output already exists: $root" >&2; exit 2; }
mkdir -p "$root"
common=(--mode simple --episodes 1 --headless --record-video --fast-ik
  --transfer-config config/experiments/rb3_transfer_recovery_candidate.json
  --states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt)
for condition in zero_agent residual_rl; do
  extra=()
  [[ "$condition" == zero_agent ]] && extra+=(--zero-actions)
  bash scripts/arm_transfer_recovery.sh "${common[@]}" "${extra[@]}" \
    --output "$root/$condition" > "$root/${condition}.log" 2>&1
  test -f "$root/$condition/metadata.json"
done
