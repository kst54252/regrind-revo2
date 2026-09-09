#!/usr/bin/env bash
# Fresh Isaac physical recordings. Neither side contains an RB3 articulation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
presentation_args=()
if [[ "${1:-}" == "--presentation-hq" ]]; then
    shift
    presentation_args=(--hide-revo2-keypoints
        env.video_recorder.window_width=3840 env.video_recorder.window_height=2160)
fi
case $# in
    1) checkpoint="${DEFAULT_FLOATING_CHECKPOINT}"; record_dir="$1" ;;
    2) checkpoint="$1"; record_dir="$2" ;;
    *) die "Usage: bash scripts/record_floating_comparison.sh [--presentation-hq] [CHECKPOINT] NEW_OUTPUT_DIRECTORY" ;;
esac
require_file "$checkpoint" checkpoint
[[ ! -e "$record_dir" ]] || die "Output already exists: $record_dir"
mkdir -p "$record_dir"
common=(--sequence 20200709_143747_left --checkpoint "$checkpoint" --headless
  --num_envs 1 --seed 42 --video --video_length 38 --rollout-frames 38
  'env.viewer.eye=(0.85,0.65,0.55)' 'env.viewer.lookat=(0.4,-0.02,0.14)'
  'env.viewer.resolution=(960,540)'
  env.commands.reference.randomize_object_xy=False
  env.terminations.time_out=None env.terminations.success=None
  env.terminations.object_deviation=None
  env.terminations.hand_far_from_object=None)
for condition in retargeting_only residual_rl; do
  action_args=()
  [[ "$condition" == retargeting_only ]] && action_args+=(--zero_actions)
  bash "$SCRIPT_DIR/rl.sh" play "${common[@]}" "${presentation_args[@]}" "${action_args[@]}" \
    --video-output-dir "$record_dir/$condition" \
    --rollout-path "$record_dir/${condition}_states.h5" \
    > "$record_dir/${condition}.log" 2>&1
done
