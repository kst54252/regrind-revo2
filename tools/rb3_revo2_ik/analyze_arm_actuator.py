"""Per-joint response evidence; never classify approximate/reaction torque as drive saturation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

if __package__:
    from .analyze_arm_contact import read_run, verify_pair
else:
    from analyze_arm_contact import read_run, verify_pair


def intervals(mask, dt, times):
    ids = np.flatnonzero(mask)
    runs = np.split(ids, np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
    longest = max(runs,key=len) if runs else []
    return dict(fraction=float(np.mean(mask)), longest_s=len(longest)*dt,
                start_s=float(times[longest[0]]-dt) if len(longest) else None,
                end_s=float(times[longest[-1]]) if len(longest) else None)


def proximity(values, lower, upper, dt, times, threshold=.95):
    # Only a meaningful finite bound can support a near-limit statistic.
    if not np.isfinite([lower,upper]).all() or lower>=0 or upper<=0:
        return None
    return intervals((values <= threshold*lower)|(values >= threshold*upper),dt,times)


def raw_path_speed(commands, previous, dt):
    # Preserve winding and actual discontinuities. No angle modulo operation.
    return np.diff(np.vstack([previous,commands]),axis=0)/dt


def scalar_stats(value,times):
    value = np.abs(value)
    return dict(mean=float(value.mean()),p95=float(np.percentile(value,95)),
                max=float(value.max()),max_time_s=float(times[value.argmax()]))


def analyze(present,absent,baseline_dir,out):
    meta_on,on = read_run(present)
    meta_off,off = read_run(absent)
    verify_pair(meta_on,on,meta_off,off)
    results = dict(threshold=.95, large_step_threshold_rad=.1,
                   drive_saturation="UNKNOWN", actual_drive_effort_available=False,
                   projected_effort_compared_to_limit=False, conditions={})
    chart_data = {}
    for label,meta,rows in (("present",meta_on,on),("absent",meta_off,off)):
        probe = meta["actuator_diagnostic"]
        _,baseline = read_run(baseline_dir/f"arm_contact_{label}_20260907.jsonl")
        if len(rows)!=len(baseline): raise ValueError("baseline length differs")
        reproduction = {}
        for key in ("all_joint_pos","all_joint_vel","actual_base_pos","actual_base_quat_xyzw",
                    "object_root_state","all_position_targets","all_velocity_targets","all_effort_targets"):
            delta = np.asarray([r[key] for r in rows])-np.asarray([r[key] for r in baseline])
            reproduction[key] = float(np.abs(delta).max())
            if reproduction[key] != 0: raise ValueError(f"Instrumentation changed {label}/{key}")
        signals = [r["actuator"] for r in rows]
        if not all(s["force_pd_units_verified"] for s in signals):
            raise ValueError("Force-PD effort analysis requires verified rotational force drives; do not interpret acceleration-drive PD as N m")
        a = lambda k: np.asarray([s[k] for s in signals],dtype=float)
        t = a("state_time_s"); dt=meta["physics_dt"]
        for signal in ("q_cmd","q_actual","velocity_actual","computed_effort_approx",
                       "applied_effort_approx","submitted_effort_Nm","submitted_velocity_rad_s"):
            if not np.isfinite(a(signal)).all(): raise ValueError(f"Nonfinite {label}/{signal}")
        for s in signals:
            if s["limits"]!=probe["runtime"]: raise ValueError("runtime limits changed during replay")
        runtime = probe["runtime"]
        model_zero = bool(np.all(np.asarray(runtime["drive_model"])==0))
        vp=raw_path_speed(a("q_cmd"),probe["previous_q_cmd_seed"],dt)
        np.testing.assert_allclose(vp,a("v_path"),atol=2e-6)
        err=a("q_cmd")-a("q_actual")
        approx=a("computed_effort_approx"); clipped=a("applied_effort_approx")
        pd=a("pd_p_pre_Nm")+a("pd_d_pre_Nm")+a("submitted_effort_Nm")
        condition=dict(metadata=probe,reproduction_max_deltas=reproduction,joints={},
            approximate_pd_identity_max_delta=float(np.abs(pd-approx).max()),
            effort_model="symmetric static configured limit; no native envelope coefficients" if model_zero else "nonzero envelope: scalar effort proximity unsupported",
            C={phase:{"position_m":scalar_stats(a("C_position_m")[mask],t[mask]),
                      "rotation_rad":scalar_stats(a("C_rotation_rad")[mask],t[mask])}
               for phase,mask in {"all":np.ones(len(t),bool),"startup":t<=4*dt,"work":t>4*dt}.items()},
            peaks={},projected_status=sorted({s["projected_status"] for s in signals}))
        for j,name in enumerate(probe["joint_names"]):
            vl=runtime["velocity_limits"][j]; el=runtime["effort_limits"][j]
            joint=dict(tracking_error_rad=scalar_stats(err[:,j],t),
                path_speed_rad_s=scalar_stats(vp[:,j],t),actual_speed_rad_s=scalar_stats(a("velocity_actual")[:,j],t),
                submitted_velocity_rad_s=scalar_stats(a("submitted_velocity_rad_s")[:,j],t),
                submitted_effort_Nm=scalar_stats(a("submitted_effort_Nm")[:,j],t),
                approximation_Nm=scalar_stats(approx[:,j],t),
                actual_speed_near_limit=proximity(a("velocity_actual")[:,j],-vl,vl,dt,t),
                path_speed_near_limit=proximity(vp[:,j],-vl,vl,dt,t),
                velocity_target_near_limit=proximity(a("submitted_velocity_rad_s")[:,j],-vl,vl,dt,t),
                submitted_effort_near_limit=proximity(a("submitted_effort_Nm")[:,j],-el,el,dt,t) if model_zero else None,
                approximation_near_limit_NOT_drive=proximity(approx[:,j],-el,el,dt,t) if model_zero else None,
                approximation_clipped_NOT_drive=intervals(np.abs(approx[:,j]-clipped[:,j])>1e-6,dt,t),
                max_command_step_rad=float(np.abs(a("command_step")[:,j]).max()),
                large_step_times_s=t[np.abs(a("command_step")[:,j])>.1].tolist(),
                drive_effort_proximity=None,confirmed_drive_clipping="UNKNOWN")
            condition["joints"][name]=joint
        for peak in (.575,.900):
            i=int(np.argmin(abs(t-peak)))
            condition["peaks"][str(peak)]=dict(time_s=float(t[i]),reference_frame=rows[i]["reference_frame"],
                tracking_error_rad=err[i].tolist(),
                signals={key:signals[i][key] for key in ("v_path","velocity_actual","submitted_velocity_rad_s",
                    "pd_p_pre_Nm","pd_d_pre_Nm","submitted_effort_Nm","computed_effort_approx","applied_effort_approx",
                    "C_position_m","C_rotation_rad")})
        results["conditions"][label]=condition
        chart_data[label]=(t,err,vp,signals)
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    with (out/"per_joint.csv").open("x",newline="") as f:
        w=csv.writer(f); w.writerow(["condition","joint","error_mean_rad","error_p95_rad","error_max_rad","path_speed_max_rad_s","speed_max_rad_s","approx_effort_max_Nm","approx_near_fraction_NOT_drive","approx_clipped_fraction_NOT_drive","drive_saturation"])
        for label,result in results["conditions"].items():
            for name,j in result["joints"].items():
                w.writerow([label,name,*[j["tracking_error_rad"][s] for s in ("mean","p95","max")],
                    j["path_speed_rad_s"]["max"],j["actual_speed_rad_s"]["max"],j["approximation_Nm"]["max"],
                    j["approximation_near_limit_NOT_drive"]["fraction"] if j["approximation_near_limit_NOT_drive"] else None,
                    j["approximation_clipped_NOT_drive"]["fraction"],"UNKNOWN"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for label,(t,err,vp,signals) in chart_data.items():
        a=lambda k:np.asarray([s[k] for s in signals])
        fig,axes=plt.subplots(6,3,figsize=(15,17),sharex=True)
        for j,name in enumerate(meta_on["actuator_diagnostic"]["joint_names"]):
            axes[j,0].plot(t,err[:,j],label="q_cmd - q_actual")
            axes[j,0].set_ylabel(f"{name}\nerror [rad]")
            axes[j,1].plot(t,vp[:,j],label="position-path slope")
            axes[j,1].plot(t,a("velocity_actual")[:,j],label="actual speed")
            axes[j,1].plot(t,a("submitted_velocity_rad_s")[:,j],":",label="submitted velocity")
            v=meta_on["actuator_diagnostic"]["runtime"]["velocity_limits"][j]
            for bound in (-v,v):axes[j,1].axhline(bound,color="gray",linestyle="--")
            axes[j,1].set_ylabel("rad/s")
            axes[j,2].plot(t,a("computed_effort_approx")[:,j],label="PD approximation")
            axes[j,2].plot(t,a("applied_effort_approx")[:,j],"--",label="clipped approximation")
            axes[j,2].plot(t,a("submitted_effort_Nm")[:,j],":",label="submitted feedforward")
            e=meta_on["actuator_diagnostic"]["runtime"]["effort_limits"][j]
            for bound in (-e,e):axes[j,2].axhline(bound,color="gray",linestyle="--")
            axes[j,2].set_ylabel("Approx./command [N m]\nNOT measured drive torque")
        for ax in axes.flat:
            for peak in (.575,.9):ax.axvline(peak,color="black",alpha=.2)
            ax.grid(alpha=.25)
        for ax in axes[0]:ax.legend(fontsize=7)
        for ax in axes[-1]:ax.set_xlabel("Episode time [s]")
        fig.suptitle(f"Can {label}: arm response, original timing/settings\nEffort lines are approximations/commands; drive saturation UNKNOWN")
        fig.tight_layout(); fig.savefig(out/f"response_{label}.png",dpi=140); plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for row,j in enumerate((1,3,5)):
            for k in ("pd_p_pre_Nm","pd_d_pre_Nm","computed_effort_approx"):
                axes[row,0].plot(t,a(k)[:,j],label=k)
            axes[row,0].set_ylabel(f"{meta_on['arm_names'][j]}\nPre-step approximation [N m]")
            axes[row,1].plot(t,a("C_position_m")*1000,label="C position [mm]")
            axes[row,1].plot(t,np.rad2deg(a("C_rotation_rad")),label="C rotation [deg]")
        for ax in axes.flat:
            for peak in (.575,.9):ax.axvline(peak,color="black",alpha=.2)
            ax.grid(alpha=.25);ax.legend(fontsize=7)
        for ax in axes[-1]:ax.set_xlabel("Episode time [s]")
        fig.suptitle(f"Can {label}: pre-step P/D cancellation vs post-step wrist error\nApproximate P and D are NOT independent measured torques")
        fig.tight_layout();fig.savefig(out/f"pd_terms_{label}.png",dpi=140);plt.close(fig)
    print(json.dumps({k:v["joints"] for k,v in results["conditions"].items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("present",type=Path);p.add_argument("absent",type=Path)
    p.add_argument("--baseline-dir",type=Path,default=Path("outputs/diagnostics"))
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();analyze(args.present,args.absent,args.baseline_dir,args.output_dir)
