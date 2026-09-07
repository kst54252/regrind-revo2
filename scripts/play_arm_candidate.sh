#!/usr/bin/env bash
# Actual frozen policy + dynamic contact; optimized, not a 1x-rate guarantee.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
output="${1:?Usage: bash scripts/play_arm_candidate.sh NEW_OUTPUT_DIRECTORY [extra evaluator options]}"
shift
exec bash scripts/evaluate_mounted_interface.sh --mode simple --episodes 20 \
  --visualizer kit --realtime-view --fast-ik \
  --transfer-config config/experiments/rb3_transfer_recovery_candidate.json \
  --states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt \
  --output "$output" "$@"
