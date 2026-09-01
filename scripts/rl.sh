#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: ./scripts/rl.sh COMMAND [options]

Commands:
  train    Train PPO (16-env smoke task by default; pass --full for full task)
  play     Replay a trained policy in the deterministic GUI task
  zero     Replay the reference with zero residual actions
  debug    Inspect observation, reward, RSI, and finite-value checks

Common project options:
  --sequence NAME    Use outputs/isaac/dexycb/NAME/rb3_revo2_reference.h5
  --reference PATH   Use an explicit reference file (overrides --sequence)

Zero options handled here:
  --gui              Open the Kit viewer
  --skeleton         Draw the source MANO21 skeleton

All unrecognized options are forwarded to the underlying Isaac Lab script.
EOF
}

[[ $# -gt 0 ]] || { usage; exit 2; }
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi
command_name="$1"
shift

sequence="${DEFAULT_SEQUENCE}"
reference=""
full_training=false
gui=false
skeleton=false
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence)
            [[ $# -ge 2 ]] || die "--sequence requires a value"
            sequence="$2"
            shift 2
            ;;
        --reference)
            [[ $# -ge 2 ]] || die "--reference requires a value"
            reference="$2"
            shift 2
            ;;
        env.commands.reference.trajectory_path=*)
            reference="${1#*=}"
            shift
            ;;
        --full)
            full_training=true
            shift
            ;;
        --gui)
            gui=true
            shift
            ;;
        --skeleton)
            skeleton=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            passthrough+=("$1")
            shift
            ;;
    esac
done

[[ -n "${reference}" ]] || reference="$(reference_for_sequence "${sequence}")"
require_file "${reference}" "reference trajectory"
setup_regrind_python

case "${command_name}" in
    train)
        task="Regrind-RB3-Revo2-TunaCan-Smoke-v0"
        if [[ "${full_training}" == true ]]; then
            task="Regrind-RB3-Revo2-TunaCan-v0"
        fi
        exec "${ISAAC_PYTHON}" \
            "${PROJECT_ROOT}/regrind/scripts/rsl_rl/train.py" \
            --task "${task}" \
            "${passthrough[@]}" \
            "env.commands.reference.trajectory_path=${reference}"
        ;;
    play)
        play_args=()
        headless=false
        for argument in "${passthrough[@]}"; do
            [[ "${argument}" == "--headless" ]] && headless=true
        done
        if [[ "${headless}" == false ]]; then
            play_args+=(--viz kit --max_visible_envs 1)
        fi
        exec "${ISAAC_PYTHON}" \
            "${PROJECT_ROOT}/regrind/scripts/rsl_rl/play.py" \
            --task Regrind-RB3-Revo2-TunaCan-Play-v0 \
            "${play_args[@]}" \
            "${passthrough[@]}" \
            "env.commands.reference.trajectory_path=${reference}"
        ;;
    zero)
        zero_args=(--reference "${reference}")
        [[ "${gui}" == true ]] && zero_args+=(--viz kit --max_visible_envs 1)
        [[ "${skeleton}" == true ]] && zero_args+=(--show_skeleton)
        exec "${ISAAC_PYTHON}" \
            "${PROJECT_ROOT}/regrind/scripts/validate_rb3_revo2_zero_residual.py" \
            "${zero_args[@]}" \
            "${passthrough[@]}"
        ;;
    debug)
        object_keypoints="${PROJECT_ROOT}/007_tuna_fish_can/object_points_50.npy"
        require_file "${object_keypoints}" "tuna object keypoints"
        exec "${ISAAC_PYTHON}" \
            "${PROJECT_ROOT}/regrind/scripts/debug_rb3_revo2_mdp.py" \
            --reference "${reference}" \
            --object-keypoints "${object_keypoints}" \
            "${passthrough[@]}"
        ;;
    *)
        usage
        die "unknown RL command: ${command_name}"
        ;;
esac
