#!/usr/bin/env bash
# Freeze the screened tau before any full/held-out policy evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
tau="${1:?Pass the selected, already-screened response tau}"
case "$tau" in 0.1|0.075) ;; *) exit 2 ;; esac
root=outputs/diagnostics/arm_transfer_recovery
common=(--episodes 20 --headless --checkpoint
  logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt)
run() {
  local label="$1"
  shift
  bash scripts/arm_transfer_recovery.sh "${common[@]}" "$@" --output "${root}/${label}" > "/tmp/recovery_${label}.log" 2>&1
  test -f "${root}/${label}/metadata.json"
}
run R0_full --mode simple --stage actual --actual-source "${root}/F0"
run R1_full --mode simple --stage recovery --recovery-kind R1 --actual-source "${root}/F0"
run R2_full --mode simple --stage recovery --recovery-kind R1 --actual-source "${root}/F0" --arm-gains-key c3 --arm-velocity-path
run live_old20_baseline --mode legacy
run live_old20_candidate --mode simple --arm-gains-key c3 --arm-velocity-path --arm-response-physics --response-tau "$tau"
# Capture real new resets from the unmodified legacy policy. No candidate saw these placements.
run live_new20_v2_baseline --mode legacy --new-state-seed 20260908 --save-state-bank "${root}/heldout_initial_states_v2.jsonl"
run live_new20_v2_candidate --mode simple --arm-gains-key c3 --arm-velocity-path --arm-response-physics --response-tau "$tau" \
  --states "${root}/heldout_initial_states_v2.jsonl"
