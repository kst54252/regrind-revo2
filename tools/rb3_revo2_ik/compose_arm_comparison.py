"""Verify paired simulation captures, then compose 1x zero-agent/RL video."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--motion-duration',type=float,help='Retiming only: both motions span this many seconds, without presentation holds')
    parser.add_argument('--floating-style',action='store_true',help='Match floating comparison: 0.25x playback, 1s initial hold, 10s total, 1920x540 at 30fps')
    args=parser.parse_args();root=args.directory
    if args.floating_style and args.motion_duration is not None:
        parser.error('--floating-style and --motion-duration are mutually exclusive')
    if args.motion_duration is not None and (not np.isfinite(args.motion_duration) or args.motion_duration<=0):
        raise ValueError('Motion duration must be positive and finite')
    names=('zero_agent','residual_rl')
    metas=[json.loads((root/n/'metadata.json').read_text()) for n in names]
    a,b=metas
    for key in ('checkpoint_sha256','reference','physics_dt','control_dt','gains','limits',
                'gravity','robot_spawn','response_tau_s','fast_ik','arm_velocity_path','arm_response_physics'):
        if a[key]!=b[key]:raise ValueError(f'Comparison invariant differs: {key}')
    for key in ('all_q','all_v','wrist_pos','wrist_quat','object_state','phase'):
        np.testing.assert_array_equal(a['initial_states'][0][key],b['initial_states'][0][key])
    if not a['zero_actions'] or a['policy_loaded'] or b['zero_actions'] or not b['frozen_policy_verified']:
        raise ValueError('Incorrect zero/frozen-policy execution modes')
    for line in (root/'zero_agent'/'physics.jsonl').open():
        row=json.loads(line)
        np.testing.assert_array_equal(row['action'],np.zeros(12))
    sources=[];lengths=[];counts=[]
    for n,m in zip(names,metas):
        videos=list((root/n/'video').glob('*.mp4'))
        if len(videos)!=1:raise ValueError('Expected exactly one native capture per condition')
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,r_frame_rate','-of','json',str(videos[0])]))['streams'][0]
        numerator,denominator=map(int,info['r_frame_rate'].split('/'))
        if abs(numerator/denominator-1/m['control_dt'])>1e-8:raise ValueError('Capture FPS does not match simulation time')
        count=int(info['nb_frames'])
        if count<3 or len(m['ends'])!=1:raise ValueError('Incomplete episode capture')
        counts.append(count-1);lengths.append((count-1)*m['control_dt']);sources.append(videos[0])
    # Gym auto-resets before final rendering. Remove that single reset image,
    # then freeze the last genuine pre-reset frame. Never extend physics commands.
    retimed=args.motion_duration is not None
    duration=args.motion_duration if retimed else max(lengths)+3.0
    initial_hold=0 if retimed else 1
    filters=[]
    headers=('ZERO AGENT - Retargeting only','TRAINED POLICY - Retargeting + Residual RL')
    for i,(count,length,title,m) in enumerate(zip(counts,lengths,headers,metas)):
        success=bool(m['ends'][0]['termination']['success'])
        outcome='SUCCESS' if success else 'FAILED'
        scale=duration/length if retimed else 1
        presentation_length=length*scale
        filters.append(f"[{i}:v]trim=end_frame={count},setpts={scale}*(PTS-STARTPTS),fps=30,"
            f"tpad=start_mode=clone:start_duration={initial_hold}:stop_mode=clone:stop_duration={max(0,duration-initial_hold-presentation_length)+1/30},"
            f"trim=duration={duration},drawbox=x=0:y=0:w=iw:h=60:color=black@0.85:t=fill,"
            f"drawtext=text='{title}':fontcolor=white:fontsize=30:x=(w-tw)/2:y=17,"
            f"drawtext=text='INITIAL FRAME HOLD':fontcolor=white:box=1:boxcolor=black@0.7:fontsize=24:x=20:y=80:enable='lt(t,{initial_hold})',"
            f"drawtext=text='{outcome}{' - RETIMED' if retimed else ' - EPISODE ENDED - FRAME HOLD'}':fontcolor=white:box=1:boxcolor=black@0.7:fontsize=23:x=20:y=80:enable='gte(t,{duration-.4 if retimed else 1+length})'[v{i}]")
    timing_label=f'Both motions slowed to {duration:g}s - individually retimed' if retimed else '1x simulation-time recording - NOT live'
    filters.append("[v0][v1]hstack=inputs=2,drawbox=x=1278:y=0:w=4:h=ih:color=black:t=fill,"
        "drawbox=x=0:y=678:w=iw:h=42:color=black@0.85:t=fill,"
        f"drawtext=text='Isaac Sim recording | {timing_label}':fontcolor=white:fontsize=26:x=(w-tw)/2:y=687[out]")
    suffix=f'_slow_{duration:g}s' if retimed else ''
    if args.floating_style:
        # Presentation only. Keep identical time scale on both sides, including
        # the earlier failed episode; never stretch it to the successful one.
        duration=10.0;initial_hold=1.0;scale=4.0;suffix='_floating_style'
        if initial_hold+max(lengths)*scale>duration:
            raise ValueError('Capture too long for 10s floating style; refusing to cut motion')
        filters=[]
        titles=('RB3 + Revo2 - Retargeting only','RB3 + Revo2 - Retargeting + Residual RL')
        for i,(count,length,title) in enumerate(zip(counts,lengths,titles)):
            filters.append(f"[{i}:v]trim=end_frame={count},scale=960:540,setsar=1,"
                f"setpts=4*(PTS-STARTPTS),fps=30,"
                f"tpad=start_mode=clone:start_duration=1:stop_mode=clone:stop_duration=10,"
                f"trim=duration=10,setpts=PTS-STARTPTS,"
                f"drawbox=x=0:y=0:w=iw:h=58:color=black@0.8:t=fill,"
                f"drawtext=text='{title}':fontcolor=white:fontsize=28:x=(w-tw)/2:y=16[v{i}]")
        hold_starts=[initial_hold+(count-1)*m['control_dt']*scale for count,m in zip(counts,metas)]
        early_side='LEFT' if hold_starts[0]<=hold_starts[1] else 'RIGHT'
        filters.append("[v0][v1]hstack=inputs=2,drawbox=x=958:y=0:w=4:h=ih:color=black:t=fill,"
            "drawbox=x=0:y=500:w=iw:h=40:color=black@0.8:t=fill,"
            "drawtext=text='Isaac Sim physics  |  Same initial state  |  0.25x playback':"
            "fontcolor=white:fontsize=22:x=22:y=510,"
            "drawtext=text='INITIAL FRAME HOLD':fontcolor=white:fontsize=22:x=w-tw-22:y=510:enable='lt(t,1)',"
            f"drawtext=text='{early_side} FRAME HOLD':fontcolor=white:fontsize=22:x=w-tw-22:y=510:enable='gte(t,{min(hold_starts)})*lt(t,{max(hold_starts)})',"
            f"drawtext=text='FINAL FRAME HOLD':fontcolor=white:fontsize=22:x=w-tw-22:y=510:enable='gte(t,{max(hold_starts)})'[out]")
    output=root/f'isaac_rb3_retargeting_vs_residual_rl{suffix}.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-n','-i',str(sources[0]),'-i',str(sources[1]),
        '-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-crf','18','-preset','medium',
        '-pix_fmt','yuv420p','-r','30','-movflags','+faststart',str(output)],check=True)
    report=dict(initial_states_bitwise_equal=True,controller_and_physics_equal=True,
        playback_rate={n:length/duration if retimed else 1 for n,length in zip(names,lengths)},
        individually_retimed=retimed,initial_hold_s=initial_hold,final_hold_s=0 if retimed else 2,
        dropped_terminal_reset_image_per_side=1,
        termination={n:m['ends'][0]['termination'] for n,m in zip(names,metas)},
        sources=[str(s) for s in sources],output=str(output),duration_s=duration)
    if args.floating_style:
        report.update(presentation_style='floating comparison',playback_rate={n:.25 for n in names},
            final_hold_s={n:duration-initial_hold-length*4 for n,length in zip(names,lengths)},
            final_frame_first_visible_s={n:initial_hold+(count-1)*m['control_dt']*4
                for n,count,m in zip(names,counts,metas)},
            width=1920,height=540,fps=30,
            note='Same physical-time playback rate; earlier termination freezes sooner. Holds are not continued simulation.')
    with (root/f'comparison{suffix}.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
