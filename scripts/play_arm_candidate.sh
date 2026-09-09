#!/usr/bin/env bash
# Actual frozen policy + dynamic contact; optimized, not a 1x-rate guarantee.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
cd "${PROJECT_ROOT}"
output="${1:?Usage: bash scripts/play_arm_candidate.sh NEW_OUTPUT_DIRECTORY [--match-recording RUN_DIRECTORY | extra evaluator options]}"
shift
profile_args=(--arm-controller video)
for argument in "$@"; do
  case "$argument" in
    --match-recording|--match-recording=*|--transfer-config|--transfer-config=*|--arm-controller|--arm-controller=*) profile_args=();;
  esac
done
exec bash scripts/evaluate_mounted_interface.sh --mode simple --episodes 20 \
  --visualizer kit --realtime-view --fast-ik \
  --transfer-config config/experiments/rb3_transfer_recovery_candidate.json \
  --states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl \
  --checkpoint "${DEFAULT_FLOATING_CHECKPOINT}" \
  --output "$output" "${profile_args[@]}" "$@"
