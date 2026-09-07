#!/usr/bin/env bash
# Render the same saved retargeting as semantic skeletons and real Isaac USD meshes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
setup_regrind_python
[[ $# -ge 1 ]] || die "Usage: bash scripts/record_retargeting_presentation.sh NEW_OUTPUT_DIRECTORY [--sequence NAME]"
output="$1"
shift
[[ ! -e "$output" ]] || die "Output already exists: $output"
"${ISAAC_PYTHON}" -m tools.revo2_kinematics.record_retargeting_presentation --output "$output" "$@"
ffmpeg -hide_banner -loglevel warning -n -i "$output/skeleton.mp4" -i "$output/isaac.mp4" -filter_complex "
 [0:v]fps=30,tpad=stop_mode=clone:stop_duration=1,pad=960:1080:0:100:color=white,
 drawtext=text='MANO to Revo2 retargeting':fontsize=32:fontcolor=0x172235:x=(w-tw)/2:y=22,
 drawtext=text='Human MANO':fontsize=23:fontcolor=0x1659c9:x=230:y=66,
 drawtext=text='Revo2 FK':fontsize=23:fontcolor=0xe65c0a:x=510:y=66[l];
 [1:v]fps=30,tpad=stop_mode=clone:stop_duration=1,pad=960:1080:0:100:color=white,
 drawtext=text='Revo2 model in Isaac Sim':fontsize=32:fontcolor=0x172235:x=(w-tw)/2:y=22,
 drawtext=text='Same wrist pose and 6 joint angles':fontsize=23:fontcolor=0x465367:x=(w-tw)/2:y=66[r];
 [l][r]hstack=inputs=2,drawbox=x=959:y=0:w=2:h=ih:color=0xd5dce5:t=fill,
 drawtext=text='Synchronized views  |  0.33x playback  |  Kinematic visualization - no RL or physics':fontsize=27:fontcolor=0x465367:x=(w-tw)/2:y=1026[v]
 " -map '[v]' -an -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart "$output/retargeting_skeleton_vs_isaac.mp4"
ffprobe -v error -show_entries format=duration:stream=width,height,nb_frames,avg_frame_rate -of json "$output/retargeting_skeleton_vs_isaac.mp4"
