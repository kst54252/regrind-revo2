"""Fixed-window precision/stability scoring; no lag shifting or target fitting."""
import json
import numpy as np


def summarize(meta,rows,out):
    c=meta['benchmark'];dt=c['physics_dt'];a=lambda k:np.asarray([r[k] for r in rows])
    t=a('time_s');q=a('q_actual');v=a('v_actual');acc=(v-a('v_pre'))/dt
    pe=a('position_error_m');re=a('rotation_error_rad');err=q-a('q_target')
    def stats(values):
        x=np.abs(values)
        return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
    motion=np.array([meta['phases'][r['phase']]['moving'] for r in rows]);holds={};static=np.zeros(len(rows),bool)
    stable=True
    for j,phase in enumerate(meta['phases']):
        if not phase['name'].startswith('hold_'):continue
        full=a('phase')==j;last=full&(t>phase['end_s']-c['hold_window_s']+1e-8);static|=last
        ptp=np.ptp(q[last],axis=0);speed=np.max(abs(v[last]),axis=0)
        quiet=bool(np.all(ptp<=c['stability_hold_peak_to_peak_rad']) and np.all(speed<=c['stability_hold_speed_rad_s']))
        stable &= quiet
        passed=(pe[full]<=c['static_position_target_m'])&(np.rad2deg(re[full])<=c['rotation_target_deg'])
        remains=np.logical_and.accumulate(passed[::-1])[::-1];indices=np.flatnonzero(remains)
        direction=np.sign(np.asarray(phase['end_q'])-meta['phases'][j-1]['start_q'])
        overshoot=np.maximum(0,np.max(err[full]*direction,axis=0))
        holds[phase['name']]=dict(position_m=stats(pe[last]),rotation_deg=stats(np.rad2deg(re[last])),
            joint_error_mean_rad=np.mean(abs(err[last]),axis=0).tolist(),joint_error_max_rad=np.max(abs(err[last]),axis=0).tolist(),
            joint_peak_to_peak_rad=ptp.tolist(),joint_peak_speed_rad_s=speed.tolist(),stable=quiet,
            settling_time_to_engineering_tolerance_s=float(t[full][indices[0]]-phase['start_s']) if len(indices) else None,
            joint_overshoot_rad=overshoot.tolist())
    bounds=np.array(meta['runtime']['position_limits'])[meta['backend_arm_ids']]
    velocity_limits=np.array(meta['runtime']['velocity'])[meta['backend_arm_ids']]
    finite=all(np.isfinite(a(k)).all() for k in ('q_actual','v_actual','actual_pos','actual_quat_xyzw','contact_force_N'))
    pos_violation=np.any((q<bounds[:,0]-1e-6)|(q>bounds[:,1]+1e-6),axis=1)
    vel_violation=np.any(abs(v)>velocity_limits+1e-4,axis=1)
    contact=np.linalg.norm(a('contact_force_N'),axis=-1)
    contact_rows=np.any(contact>c['contact_rejection_N'],axis=1)
    static_pass=bool(np.max(pe[static])<=c['static_position_target_m'] and np.max(np.rad2deg(re[static]))<=c['rotation_target_deg'])
    # Require max rotation <= 0.5 deg during motion (stricter than P95).
    motion_pass=bool(np.percentile(pe[motion],95)<=c['motion_p95_position_target_m'] and np.max(np.rad2deg(re[motion]))<=c['rotation_target_deg'])
    valid=finite and stable and not np.any(pos_violation|vel_violation|contact_rows)
    headline=dict(static_max_mm=float(np.max(pe[static])*1000),static_max_deg=float(np.max(np.rad2deg(re[static]))),
        motion_p95_mm=float(np.percentile(pe[motion],95)*1000),motion_max_mm=float(np.max(pe[motion])*1000),
        motion_p95_deg=float(np.percentile(np.rad2deg(re[motion]),95)),motion_max_deg=float(np.max(np.rad2deg(re[motion]))),
        static_target_met=static_pass,motion_target_met=motion_pass,stable=stable,valid=bool(valid),
        contact_peak_N=float(contact.max()),position_limit_violations=int(pos_violation.sum()),velocity_limit_violations=int(vel_violation.sum()),
        finite=finite,drive_saturation='UNKNOWN')
    result=dict(headline=headline,holds=holds,motion_position_m=stats(pe[motion]),motion_rotation_deg=stats(np.rad2deg(re[motion])),
        joint_error_mean_rad=np.mean(abs(err[motion]),axis=0).tolist(),joint_error_max_rad=np.max(abs(err),axis=0).tolist(),
        actual_speed_max_rad_s=np.max(abs(v),axis=0).tolist(),actual_acceleration_max_rad_s2=np.max(abs(acc),axis=0).tolist(),
        motion_speed_max_rad_s=np.max(abs(v[motion]),axis=0).tolist(),motion_acceleration_max_rad_s2=np.max(abs(acc[motion]),axis=0).tolist(),
        command_speed_max_rad_s=np.max(abs(a('v_target')),axis=0).tolist(),command_acceleration_max_rad_s2=np.max(abs(a('a_target')),axis=0).tolist(),
        contact_times_s=t[contact_rows].tolist(),position_limit_times_s=t[pos_violation].tolist(),velocity_limit_times_s=t[vel_violation].tolist(),
        contact_peak_by_body_N=dict(zip(meta['contact_bodies'],contact.max(0).tolist())),
        hand_actual_deviation_max_rad=float(np.max(abs(a('all_q_actual')[:,[i for i in range(len(meta['user_joint_names'])) if i not in meta['arm_ids']]]-c['hand_position_target_rad']))))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,2,figsize=(13,10),sharex=True)
    ax[0,0].plot(t,pe*1000);ax[0,0].set_ylabel('Wrist error [mm]');ax[0,0].axhline(1,color='gray',ls='--')
    ax[0,1].plot(t,np.rad2deg(re));ax[0,1].set_ylabel('Rotation error [deg]');ax[0,1].axhline(.5,color='gray',ls='--')
    for j,n in enumerate(c['joint_names']):
        ax[1,0].plot(t,err[:,j],label=n);ax[1,1].plot(t,v[:,j],label=n);ax[2,0].plot(t,acc[:,j],label=n)
    ax[1,0].set_ylabel('Joint error [rad]');ax[1,1].set_ylabel('Actual speed [rad/s]');ax[2,0].set_ylabel('Actual acceleration [rad/s²]')
    ax[2,1].plot(t,contact.max(1));ax[2,1].set_ylabel('Peak body contact [N]')
    for x in ax.flat:
        x.grid(alpha=.25)
        for phase in meta['phases']:
            if phase['name'].startswith('hold_'):x.axvspan(phase['end_s']-1,phase['end_s'],color='green',alpha=.1)
    ax[1,0].legend(fontsize=8);ax[2,0].set_xlabel('Physical time [s]');ax[2,1].set_xlabel('Physical time [s]')
    fig.suptitle(f"{meta['candidate']} / {'held-out' if meta['held_out'] else 'selection'}; fixed benchmark; no time shift")
    fig.tight_layout();fig.savefig(out/'precision.png',dpi=140);plt.close(fig)
    return result
