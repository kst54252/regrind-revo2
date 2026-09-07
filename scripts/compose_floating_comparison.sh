#!/usr/bin/env bash
# Compose two genuine Isaac captures, identical 0.25x playback and freeze padding.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: bash scripts/compose_floating_comparison.sh RECORD_DIRECTORY" >&2; exit 2; }
record_dir="$1"
left_files=("$record_dir/retargeting_only/"*.mp4)
right_files=("$record_dir/residual_rl/"*.mp4)
[[ ${#left_files[@]} -eq 1 && ${#right_files[@]} -eq 1 ]] || { echo 'Expected exactly one recording per side' >&2; exit 2; }
left="${left_files[0]}"
right="${right_files[0]}"
output="$record_dir/isaac_floating_retargeting_vs_residual_rl.mp4"
[[ -f "$left" && -f "$right" && ! -e "$output" ]] || { echo 'Missing inputs or output already exists' >&2; exit 2; }
ffmpeg -hide_banner -loglevel warning -n -i "$left" -i "$right" -filter_complex "
 [0:v]scale=960:540,setpts=4*(PTS-STARTPTS),fps=30,tpad=start_mode=clone:start_duration=1:stop_mode=clone:stop_duration=5,trim=duration=10,setpts=PTS-STARTPTS,
 drawbox=x=0:y=0:w=iw:h=58:color=black@0.8:t=fill,
 drawtext=text='Floating hand - Retargeting only':fontcolor=white:fontsize=28:x=(w-tw)/2:y=16[l];
 [1:v]scale=960:540,setpts=4*(PTS-STARTPTS),fps=30,tpad=start_mode=clone:start_duration=1:stop_mode=clone:stop_duration=5,trim=duration=10,setpts=PTS-STARTPTS,
 drawbox=x=0:y=0:w=iw:h=58:color=black@0.8:t=fill,
 drawtext=text='Floating hand - Retargeting + Residual RL':fontcolor=white:fontsize=28:x=(w-tw)/2:y=16[r];
 [l][r]hstack=inputs=2,
 drawbox=x=958:y=0:w=4:h=ih:color=black:t=fill,
 drawbox=x=0:y=500:w=iw:h=40:color=black@0.8:t=fill,
 drawtext=text='Isaac Sim physics  |  Same initial state  |  0.25x playback':fontcolor=white:fontsize=22:x=22:y=510,
 drawtext=text='INITIAL FRAME HOLD':fontcolor=white:fontsize=22:x=w-tw-22:y=510:enable='lt(t,1)',
 drawtext=text='FINAL FRAME HOLD':fontcolor=white:fontsize=22:x=w-tw-22:y=510:enable='gte(t,5.8)'[v]
 " -map '[v]' -an -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -r 30 -frames:v 300 -movflags +faststart "$output"
ffprobe -v error -show_entries format=duration:stream=width,height,avg_frame_rate,nb_frames -of json "$output"
