#!/usr/bin/env bash
# Compose two genuine Isaac captures, identical 0.25x playback and freeze padding.
set -euo pipefail
hq=false
if [[ "${1:-}" == "--presentation-hq" ]]; then hq=true; shift; fi
[[ $# -eq 1 ]] || { echo "Usage: bash scripts/compose_floating_comparison.sh [--presentation-hq] RECORD_DIRECTORY" >&2; exit 2; }
record_dir="$1"
left_files=("$record_dir/retargeting_only/"*.mp4)
right_files=("$record_dir/residual_rl/"*.mp4)
[[ ${#left_files[@]} -eq 1 && ${#right_files[@]} -eq 1 ]] || { echo 'Expected exactly one recording per side' >&2; exit 2; }
left="${left_files[0]}"
right="${right_files[0]}"
output="$record_dir/isaac_floating_retargeting_vs_residual_rl.mp4"
[[ -f "$left" && -f "$right" && ! -e "$output" ]] || { echo 'Missing inputs or output already exists' >&2; exit 2; }
if [[ "$hq" == true ]]; then
    output="$record_dir/isaac_floating_retargeting_vs_residual_rl_presentation_hq.mp4"
    [[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
    # This recipe preserves the verified 37-frame 4K capture's timestamps.
    # Refuse other inputs rather than silently trimming a valid first frame.
    for input in "$left" "$right"; do
        shape=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 "$input")
        [[ "$shape" == '3840,2160,30/1,37' ]] || { echo "Expected 37-frame 4K/30 capture: $shape" >&2; exit 2; }
        first=$(ffmpeg -hide_banner -i "$input" -vf blackframe=amount=99:threshold=24 -frames:v 1 -an -f null - 2>&1)
        [[ "$first" == *'frame:0 pblack:100'* ]] || { echo 'Expected one unrendered first frame; inspect before trimming' >&2; exit 2; }
    done
    ffmpeg -hide_banner -loglevel warning -n -i "$left" -i "$right" -filter_complex "
     [0:v]trim=start_frame=1,setpts=2*(PTS-STARTPTS),fps=30,crop=1920:1728:960:256,scale=960:864:flags=lanczos,tpad=stop_mode=clone:stop_duration=2[l];
     [1:v]trim=start_frame=1,setpts=2*(PTS-STARTPTS),fps=30,crop=1920:1728:960:256,scale=960:864:flags=lanczos,tpad=stop_mode=clone:stop_duration=2[r];
     [l][r]hstack=inputs=2,pad=1920:1080:0:156:color=0x3a3a3a,
     drawbox=x=957:y=81:w=6:h=939:color=0x3a3a3a:t=fill,
     drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Floating Revo2 + Tuna Can':fontsize=42:fontcolor=white:x=(w-tw)/2:y=23,
     drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Retargeting Only':fontsize=36:fontcolor=white:x=(960-tw)/2:y=98,
     drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Retargeting + Residual RL':fontsize=36:fontcolor=white:x=960+(960-tw)/2:y=98,
     drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Isaac Sim Physics  |  Same Initial State  |  0.5x Playback':fontsize=28:fontcolor=white:x=30:y=1037,
     drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='FINAL FRAME HOLD':fontsize=27:fontcolor=white:x=w-tw-30:y=1037:enable='gte(t,2.4)'[v]
    " -map '[v]' -frames:v 103 -an -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p -movflags +faststart "$output"
    ffprobe -v error -show_entries format=duration:stream=width,height,nb_frames,avg_frame_rate -of json "$output"
    exit 0
fi
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
