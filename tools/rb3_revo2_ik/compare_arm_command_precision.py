"""Compare frozen command modes with unshifted/original and submitted targets."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(first, second, output):
    output.mkdir(parents=True,exist_ok=False)
    def load(path):
        meta=json.loads((path/'metadata.json').read_text())
        rows=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
        return meta,rows,json.loads((path/'summary.json').read_text())
    baseline,improved=load(first),load(second)
    m0,r0,_=baseline;m1,r1,_=improved
    for key in ('benchmark_sha256','held_out','gains','physics','robot_spawn','runtime',
                'initial_q','initial_v','initial_root','initial_object','phases',
                'contact_body_paths','known_command_delay_s','command_substeps'):
        if m0[key]!=m1[key]:raise ValueError('Comparison mismatch: '+key)
    if len(r0)!=len(r1):raise ValueError('Sample count mismatch')
    for key in ('time_s','command_time_s','original_q','original_pos','original_quat_xyzw'):
        np.testing.assert_array_equal([r[key] for r in r0],[r[key] for r in r1])
    n=m0['command_substeps']
    np.testing.assert_allclose([r['q_target'] for r in r0[n-1::n]],
                               [r['q_target'] for r in r1[n-1::n]],atol=1e-14)
    def stats(x):
        x=np.abs(x)
        return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
    result=dict(equal_initial_state_physics_gains_original_targets=True,
                same_control_boundary_positions=True,
                common_delay_s=m0['known_command_delay_s'],
                distinction='Offline analytical derivative oracle; not a live RL filter or grasp validation.',modes={})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
    for meta,rows,summary in (baseline,improved):
        a=lambda key:np.asarray([r[key] for r in rows]);t=a('time_s')
        motion=np.array([meta['phases'][r['phase']]['moving'] for r in rows])
        dt=meta['benchmark']['physics_dt'];q=a('q_target');q0=np.asarray(meta['initial_q'])[meta['arm_ids']]
        path_v=np.diff(np.vstack([q0,q]),axis=0)/dt
        path_a=np.diff(np.vstack([np.zeros(q.shape[1]),path_v]),axis=0)/dt
        actual_acc=(a('v_actual')-a('v_pre'))/dt
        item=dict(headline=summary['headline'],motion={},
                  velocity_target_max_rad_s=np.max(abs(a('submitted_velocity')),axis=0).tolist(),
                  path_speed_max_rad_s=np.max(abs(path_v),axis=0).tolist(),
                  path_acceleration_fd_max_rad_s2=np.max(abs(path_a),axis=0).tolist(),
                  actual_motion_speed_max_rad_s=np.max(abs(a('v_actual')[motion]),axis=0).tolist(),
                  actual_motion_acceleration_max_rad_s2=np.max(abs(actual_acc[motion]),axis=0).tolist(),
                  differentiation='Backward raw joint differences/dt; no wrapping or time shifts. FD acceleration is not instantaneous linear-corner acceleration.',
                  holds=summary['holds'])
        for key in ('original_position_error_m','original_rotation_error_rad','position_error_m',
                    'rotation_error_rad','command_deviation_position_m','command_deviation_rotation_rad'):
            item['motion'][key]=stats(a(key)[motion])
        error=abs(a('q_actual')-q)
        item['motion_joint_error_max_rad']=np.max(error[motion],axis=0).tolist()
        item['original_motion_joint_error_max_rad']=np.max(abs(a('q_actual')[motion]-a('original_q')[motion]),axis=0).tolist()
        item['finite']=bool(all(np.isfinite(a(k)).all() for k in ('q_target','v_target','original_q','q_actual','v_actual','original_position_error_m')))
        result['modes'][meta['command_mode']]=item
        label=meta['command_mode']
        for ax,y,title in zip(axes.flat,
            [a('original_position_error_m')*1000,a('position_error_m')*1000,
             a('command_deviation_position_m')*1000,np.rad2deg(a('original_rotation_error_rad')),
             np.max(abs(actual_acc),axis=1),np.max(abs(a('contact_force_N')),axis=(1,2))],
            ['Original-target error [mm]','Submitted-target tracking [mm]','Command deviation [mm]',
             'Original rotation error [deg]','Max actual joint acceleration [rad/s²]','Contact force component peak [N]']):
            ax.plot(t,y,label=label);ax.set_ylabel(title);ax.grid(alpha=.3)
    for ax in axes.flat:ax.legend()
    axes[2,0].set_xlabel('Physics time [s]');axes[2,1].set_xlabel('Physics time [s]')
    fig.suptitle('Fixed gains, common 33.33 ms delay; no post-hoc alignment')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['motion'] for k,v in result['modes'].items()},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path);parser.add_argument('smooth',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compare(args.baseline,args.smooth,args.output)
