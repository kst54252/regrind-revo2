#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

sequence="${DEFAULT_SEQUENCE}"
checkpoint="${PROJECT_ROOT}/logs/rsl_rl/floating_revo2_tuna/2026-09-03_14-56-00_floating_full_3000/model_2999.pt"
output_dir="${PROJECT_ROOT}/outputs/floating/random_can_replay"
physics_object=true
extra_replay_args=()

usage() {
    cat <<'EOF'
Usage: ./scripts/random_can_full_replay.sh [options]

Samples a strict-IK-safe table position, runs the floating Revo2 policy,
solves RB3 strict IK, then opens the complete workcell replay.

Options:
  --sequence NAME       Reference sequence (default: 20200709_143747_left)
  --checkpoint PATH     Floating-hand PPO checkpoint
  --output-dir DIR      Generated rollout/reference directory
  --kinematic-object    Follow the reference can pose instead of contact physics
  --speed VALUE         Isaac replay speed multiplier
  --physics-hz HZ       Isaac physics/update frequency (default: 120)
  --robot-control MODE  kinematic (default) or position
  --terminal-hold SEC   Hold final target before physics pauses (default: 0.5)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence) sequence="$2"; shift 2 ;;
        --checkpoint) checkpoint="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --kinematic-object) physics_object=false; shift ;;
        --speed) extra_replay_args+=(--speed "$2"); shift 2 ;;
        --physics-hz) extra_replay_args+=(--physics-hz "$2"); shift 2 ;;
        --robot-control) extra_replay_args+=(--robot-control "$2"); shift 2 ;;
        --terminal-hold) extra_replay_args+=(--terminal-hold "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

require_file "${checkpoint}" "floating-hand checkpoint"
mkdir -p "${output_dir}"
rollout="${output_dir}/${sequence}_random_rollout.h5"
reference="${output_dir}/${sequence}_random_reference_12dof.h5"

"${SCRIPT_DIR}/rl.sh" play \
    --sequence "${sequence}" \
    --random-placement \
    --checkpoint "${checkpoint}" \
    --num_envs 1 \
    --headless \
    --rollout-path "${rollout}"

ik_args=(--rollout "${rollout}" --out "${reference}")
if [[ "${physics_object}" == true ]]; then
    # The floating rollout starts from the measured 11-degree-leaning DexYCB
    # pose. Its dynamic can settles upright by frame 2. Starting full-robot
    # replay from that stable state prevents a second fall from shifting the
    # can away from the learned grasp.
    ik_args+=(--drop-leading-frames 2 --level-object-on-table)
fi
"${SCRIPT_DIR}/floating_to_rb3.sh" "${ik_args[@]}"

replay_args=(--trajectory "${reference}" --demo-skeleton --no-loop)
[[ "${physics_object}" == true ]] && replay_args+=(--physics-object)
exec "${PROJECT_ROOT}/tools/rb3_revo2_ik/run_replay_gui.sh" \
    "${replay_args[@]}" "${extra_replay_args[@]}"
