#!/usr/bin/env bash
# Same initial state/controller/dynamic object; only residual action source differs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
cd "${PROJECT_ROOT}"
[[ $# -ge 1 && $# -le 2 ]] || die "Usage: bash scripts/record_arm_comparison.sh NEW_OUTPUT_DIRECTORY [CHECKPOINT]"
root="$1"
checkpoint="${2:-${DEFAULT_FLOATING_CHECKPOINT}}"
require_file "$checkpoint" "floating-hand checkpoint"
[[ ! -e "$root" ]] || { echo "Output already exists: $root" >&2; exit 2; }
mkdir -p "$root"
common=(--mode simple --episodes 1 --headless --record-video --fast-ik
  --transfer-config config/experiments/rb3_transfer_recovery_candidate.json
  --states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl
  --checkpoint "$checkpoint")
for condition in zero_agent residual_rl; do
  extra=()
  [[ "$condition" == zero_agent ]] && extra+=(--zero-actions)
  bash scripts/arm_transfer_recovery.sh "${common[@]}" "${extra[@]}" \
    --output "$root/$condition" > "$root/${condition}.log" 2>&1
  test -f "$root/$condition/metadata.json"
done
