#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: ./scripts/rl.sh COMMAND [options]

Commands:
  train    Train floating Revo2 PPO (16-env smoke task by default; --full uses 4096 env config)
  train-arm
           Fine-tune the same 67-D/12-D policy through the selected mounted arm controller
           Start: --transfer-init FLOATING.pt (fresh optimizer); resume: --resume --checkpoint TRANSFER.pt
           Separate logs; video default 1 env, full gravity, randomization/RSI OFF.
           --training-seconds 3600 stops after a completed PPO update. See docs/RL_TASK.md.
  play     Replay a floating-hand policy in the deterministic GUI task
  play-arm Approved video arm controller + compliant fingertips (1 env, stored placement bank)
           --arm-controller baseline retains the previous strict-IK play path
  zero     Replay the floating-hand reference with zero residual actions
  debug    Inspect observation, reward, RSI, and finite-value checks

Common project options:
  --sequence NAME    Prefer rb3_revo2_reference_stable.h5 when present, otherwise
                     use outputs/isaac/dexycb/NAME/rb3_revo2_reference.h5
  --reference PATH   Use an explicit reference file (overrides --sequence)
  --arm-controller video|baseline
                     Shared arm play/transfer controller; default video (floating unchanged)
  --output PATH      Fresh arm video-controller evaluation directory
  --legacy-arm-rl    Select the former combined RB3+Revo2 RL task
  --random-placement Enable random can/reference XY in evaluation (training default: ON)

Play options:
  Primary-sequence play/play-arm default to the pinned 10000-update floating
  checkpoint in scripts/_common.sh (override with --checkpoint or
  REGRIND_FLOATING_CHECKPOINT). Missing defaults fail instead of selecting another run.
  --rollout-path P   Save one floating-hand policy episode for downstream RB3 IK
  --arm-tracking-path P
                     Baseline controller only: first episode target/measured telemetry NPZ
  --rb3-stiffness-scale S
  --rb3-damping-scale S
  --rb3-effort-scale S
                     Baseline controller only: runtime gain/limit comparisons

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
arm_controller="video"
arm_output=""
full_training=false
gui=false
skeleton=false
legacy_arm_rl=false
random_placement=false
rollout_path=""
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm-controller)
            [[ $# -ge 2 ]] || die "--arm-controller requires video or baseline"
            arm_controller="$2"
            [[ "$arm_controller" == video || "$arm_controller" == baseline ]] || die "Unknown arm controller"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || die "--output requires a directory"
            if [[ "$command_name" == play-arm ]]; then arm_output="$2"; else passthrough+=("$1" "$2"); fi
            shift 2
            ;;
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
        --legacy-arm-rl)
            legacy_arm_rl=true
            shift
            ;;
        --random-placement)
            random_placement=true
            shift
            ;;
        --rollout-path)
            [[ $# -ge 2 ]] || die "--rollout-path requires a value"
            rollout_path="$2"
            shift 2
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

[[ -n "${reference}" ]] || reference="$(training_reference_for_sequence "${sequence}")"
require_file "${reference}" "reference trajectory"
setup_regrind_python
if [[ "${random_placement}" == true && "${legacy_arm_rl}" == true ]]; then
    die "--random-placement must run in floating-hand mode; solve strict IK for the sampled rollout before arm replay"
fi

case "${command_name}" in
    train|train-arm)
        task="Regrind-Floating-Revo2-TunaCan-Smoke-v0"
        if [[ "${command_name}" == "train-arm" ]]; then
            passthrough+=(--arm-controller "$arm_controller")
            [[ "${legacy_arm_rl}" == false ]] || die "train-arm cannot be combined with --legacy-arm-rl"
            task="Regrind-RB3-Revo2-TunaCan-Online-Smoke-v0"
        elif [[ "${legacy_arm_rl}" == true ]]; then
            task="Regrind-RB3-Revo2-TunaCan-Smoke-v0"
        fi
        if [[ "${full_training}" == true && "${command_name}" == "train-arm" ]]; then
            task="Regrind-RB3-Revo2-TunaCan-Online-v0"
        elif [[ "${full_training}" == true && "${legacy_arm_rl}" == false ]]; then
            task="Regrind-Floating-Revo2-TunaCan-v0"
        elif [[ "${full_training}" == true ]]; then
            task="Regrind-RB3-Revo2-TunaCan-v0"
        fi
        [[ "${random_placement}" == true ]] && passthrough+=("env.commands.reference.randomize_object_xy=true")
        exec "${ISAAC_PYTHON}" \
            "${PROJECT_ROOT}/regrind/scripts/rsl_rl/train.py" \
            --task "${task}" \
            "${passthrough[@]}" \
            "env.commands.reference.trajectory_path=${reference}"
        ;;
    play|play-arm)
        play_args=()
        if [[ "${legacy_arm_rl}" == false && "${sequence}" == "20200709_143747_left" ]] && \
            ! has_policy_selection "${passthrough[@]}"; then
            require_file "${DEFAULT_FLOATING_CHECKPOINT}" "default 10000-update floating checkpoint (use --checkpoint or REGRIND_FLOATING_CHECKPOINT)"
            play_args+=(--checkpoint "${DEFAULT_FLOATING_CHECKPOINT}")
        fi
        headless=false
        for argument in "${passthrough[@]}"; do
            [[ "${argument}" == "--headless" ]] && headless=true
        done
        if [[ "${headless}" == false ]]; then
            play_args+=(--visualizer kit --max_visible_envs 1)
        fi
        if [[ "${command_name}" == play-arm && "$arm_controller" == video ]]; then
            [[ "${legacy_arm_rl}" == false ]] || die "Use --arm-controller baseline for --legacy-arm-rl"
            [[ "$sequence" == "${DEFAULT_SEQUENCE}" ]] || die "Video default requires the validated sequence/state bank; custom sequences use baseline"
            [[ -n "$arm_output" ]] || arm_output="outputs/diagnostics/arm_play_$(date +%Y%m%d_%H%M%S)_$$"
            [[ "$headless" == true ]] && play_args+=(--visualizer none)
            echo "[arm execution] Approved video controller + compliant fingertips; original floating policy unless explicitly overridden." >&2
            exec bash "${SCRIPT_DIR}/evaluate_mounted_interface.sh" --mode simple --arm-controller video \
                --transfer-evaluation --episodes 20 \
                --states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl \
                --expected-reference "$reference" --output "$arm_output" \
                "${play_args[@]}" "${passthrough[@]}"
        fi
        task="Regrind-Floating-Revo2-TunaCan-Play-v0"
        if [[ "${command_name}" == "play-arm" ]]; then
            echo "[arm execution] Explicit previous strict-IK baseline selected; approved video/rubber controller is not used." >&2
            task="Regrind-RB3-Revo2-TunaCan-Online-Play-v0"
        elif [[ "${legacy_arm_rl}" == true ]]; then
            task="Regrind-RB3-Revo2-TunaCan-Play-v0"
        fi
        if [[ -n "${rollout_path}" ]]; then
            [[ "${command_name}" == "play" && "${legacy_arm_rl}" == false ]] || \
                die "--rollout-path is only valid for floating-hand RL"
            play_args+=(--rollout-path "${rollout_path}" --num_envs 1)
        fi
        [[ "${random_placement}" == true ]] && passthrough+=("env.commands.reference.randomize_object_xy=true")
        exec "${ISAAC_PYTHON}" \
            "${PROJECT_ROOT}/regrind/scripts/rsl_rl/play.py" \
            --task "${task}" \
            "${play_args[@]}" \
            "${passthrough[@]}" \
            "env.commands.reference.trajectory_path=${reference}"
        ;;
    zero)
        if [[ "${legacy_arm_rl}" == true ]]; then
            zero_args=(--reference "${reference}")
            [[ "${gui}" == true ]] && zero_args+=(--visualizer kit --max_visible_envs 1)
            [[ "${skeleton}" == true ]] && zero_args+=(--show_skeleton)
            exec "${ISAAC_PYTHON}" \
                "${PROJECT_ROOT}/regrind/scripts/validate_rb3_revo2_zero_residual.py" \
                "${zero_args[@]}" \
                "${passthrough[@]}"
        fi
        zero_args=(
            --task Regrind-Floating-Revo2-TunaCan-Play-v0
            --reference "${reference}"
            --num_envs 1
        )
        [[ "${random_placement}" == true ]] && zero_args+=(--random-placement)
        [[ "${gui}" == true ]] && zero_args+=(--visualizer kit --max_visible_envs 1)
        exec "${ISAAC_PYTHON}" \
            "${PROJECT_ROOT}/regrind/scripts/zero_agent.py" \
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
