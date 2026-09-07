"""Verify frozen inputs/physics and compare completed precision runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    meta=json.loads((path/'metadata.json').read_text())
    summary=json.loads((path/'summary.json').read_text())
    with (path/'trace.jsonl').open() as f:rows=[json.loads(line) for line in f]
    expected=round(meta['phases'][-1]['end_s']/meta['benchmark']['physics_dt'])
    if len(rows)!=expected:raise ValueError(f'Incomplete run {path}')
    return meta,summary,rows


def verify(a,b):
    ma,sa,ra=a;mb,sb,rb=b
    for key in ('benchmark_sha256','held_out','phases','initial_q','initial_v','initial_root','initial_object',
                'physics','robot_spawn','user_joint_names','backend_joint_names','body_names','contact_bodies'):
        if ma[key]!=mb[key]:raise ValueError(f'Changed comparison condition: {key}')
    for key in ('effort','velocity','position_limits','drive_type','mass','inertia','gravity_disabled'):
        if ma['runtime'][key]!=mb['runtime'][key]:raise ValueError(f'Changed native runtime {key}')
    hand=[i for i in range(len(ma['backend_joint_names'])) if i not in ma['backend_arm_ids']]
    for key in ('kp','kd'):
        np.testing.assert_array_equal(np.array(ma['runtime'][key])[hand],np.array(mb['runtime'][key])[hand])
    if len(ra)!=len(rb):raise ValueError('Different sample count')
    for x,y in zip(ra,rb):
        for key in ('step','time_s','command_time_s','q_target','v_target','a_target','target_pos','target_quat_xyzw','phase'):
            if x[key]!=y[key]:raise ValueError(f'Changed target/time {key}')


def compare(root,out):
    cases={name:read(root/name) for name in ('baseline','c1','c2','c3','baseline_heldout','best_heldout')}
    for name in ('c1','c2','c3'):verify(cases['baseline'],cases[name])
    verify(cases['baseline_heldout'],cases['best_heldout'])
    np.testing.assert_array_equal(cases['c3'][0]['gains']['kp'],cases['best_heldout'][0]['gains']['kp'])
    np.testing.assert_array_equal(cases['c3'][0]['gains']['kd'],cases['best_heldout'][0]['gains']['kd'])
    report=dict(identical_conditions_verified=True,unique_gain_configurations=4,selected='c3',held_out_gain_change=False,
                results={n:s['headline'] for n,(m,s,r) in cases.items()},engineering_excursions={})
    for name,(m,s,rows) in cases.items():
        v=np.array([r['v_actual'] for r in rows]);acc=(v-np.array([r['v_pre'] for r in rows]))/m['benchmark']['physics_dt']
        motion=np.array([m['phases'][r['phase']]['moving'] for r in rows])
        report['engineering_excursions'][name]=dict(
            actual_speed_over_05_samples=int(np.sum(np.any(abs(v)>.5,axis=1))),
            actual_acceleration_over_1_all_samples=int(np.sum(np.any(abs(acc)>1,axis=1))),
            actual_acceleration_over_1_motion_samples=int(np.sum(np.any(abs(acc[motion])>1,axis=1))))
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    for i,pair in enumerate((('baseline','c3'),('baseline_heldout','best_heldout'))):
        for name in pair:
            rows=cases[name][2];t=[r['time_s'] for r in rows]
            ax[i,0].semilogy(t,[r['position_error_m']*1000 for r in rows],label=name)
            ax[i,1].plot(t,np.rad2deg([r['rotation_error_rad'] for r in rows]),label=name)
        ax[i,0].axhline(1,color='gray',ls='--',label='static 1 mm')
        ax[i,1].axhline(.5,color='gray',ls='--',label='0.5 deg')
        for j in range(2):
            for phase in cases[pair[0]][0]['phases']:
                if phase['name'].startswith('hold_'):ax[i,j].axvspan(phase['end_s']-1,phase['end_s'],color='green',alpha=.12)
            ax[i,j].grid(alpha=.25);ax[i,j].legend();ax[i,j].set_xlabel('Unshifted physical time [s]')
        ax[i,0].set_ylabel(('Selection' if i==0 else 'Held-out')+' wrist error [mm, log scale]')
        ax[i,1].set_ylabel('Rotation error [deg]')
    fig.suptitle('Frozen static / analytical slow benchmark; identical limits and physics\nGreen: fixed final 1-second hold windows; no policy or grasp claims')
    fig.tight_layout();fig.savefig(out/'before_after.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs_dir',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();compare(args.runs_dir,args.output_dir)
