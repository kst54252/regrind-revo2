"""Single-variable zero versus recorded-path velocity-target replay analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

if __package__:
    from .analyze_arm_contact import read_run
    from .analyze_arm_actuator import raw_path_speed, scalar_stats
else:
    from analyze_arm_contact import read_run
    from analyze_arm_actuator import raw_path_speed, scalar_stats


def load_velocity_path(path, source_meta, source_rows, condition="absent"):
    meta, rows = read_run(path)
    if meta["condition"] != condition or meta.get("velocity_path_source"):
        raise ValueError("Velocity source must be a baseline actuator trace with matching contact condition")
    if meta["physics_dt"] != source_meta["physics_dt"] or len(rows)!=len(source_rows):
        raise ValueError("Velocity source duration/timestep mismatch")
    if meta["arm_names"] != source_meta["joint_names"]:
        raise ValueError("Velocity source arm order mismatch")
    order = [meta["joint_names"].index(n) for n in source_meta["user_joint_names"]]
    for row, source in zip(rows,source_rows):
        if row["source_physics_step"] != source["physics_step"] or row["time_s"] != source["state_time_s"]:
            raise ValueError("Velocity source physics-step alignment mismatch")
        for key in ("all_position_targets","all_velocity_targets","all_effort_targets"):
            np.testing.assert_array_equal(np.asarray(row[key])[order],source[key])
    velocity=np.asarray([r["actuator"]["v_path"] for r in rows])
    commands=np.asarray([r["actuator"]["q_cmd"] for r in rows])
    expected=raw_path_speed(commands,source_rows[0]["previous_accepted_q"],meta["physics_dt"])
    if not np.isfinite(velocity).all():raise ValueError("Nonfinite velocity path")
    np.testing.assert_allclose(velocity,expected,atol=2e-6)
    return velocity


def sign_changes(values, deadband):
    signs=np.sign(np.asarray(values)[np.abs(values)>deadband])
    return int(np.count_nonzero(np.diff(signs)))


def analyze(zero,path,original,out):
    mz,z=read_run(zero);mp,p=read_run(path);mo,o=read_run(original)
    if mz.get("velocity_path_source") or not mp.get("velocity_path_source"):
        raise ValueError("Expected zero baseline then velocity-path intervention")
    if not mz["condition"] == mp["condition"] == mo["condition"]:
        raise ValueError("Contact conditions must match")
    if not len(z)==len(p)==len(o):raise ValueError("Different replay durations")
    for key in ("source","episode","physics_dt","joint_names","arm_names","initial", "actual_stiffness",
                "actual_damping","actual_effort_limits","actual_velocity_limits"):
        if not mz[key]==mp[key]==mo[key]:raise ValueError(f"Changed metadata: {key}")
    for key in ("runtime","config","cli_scales","gravity_config","self_collision_config","link_gravity_disabled"):
        if not mz["actuator_diagnostic"][key]==mp["actuator_diagnostic"][key]==mo["actuator_diagnostic"][key]:
            raise ValueError(f"Changed actuator/physics metadata: {key}")
    ids=[mz["joint_names"].index(n) for n in mz["arm_names"]]
    fingers=[j for j in range(len(mz["joint_names"])) if j not in ids]
    baseline_deltas={}
    for key in ("all_joint_pos","all_joint_vel","actual_base_pos","actual_base_quat_xyzw","object_root_state"):
        delta=float(np.abs(np.asarray([r[key] for r in z])-np.asarray([r[key] for r in o])).max())
        baseline_deltas[key]=delta
        if delta!=0:raise ValueError(f"Zero replay changed from original: {key}")
    for a,b,c in zip(z,p,o):
        for key in ("time_s","physics_step","source_physics_step","reference_frame","interpolation_step", "all_position_targets",
                    "all_effort_targets","backend_position_targets"):
            if not a[key]==b[key]==c[key]:raise ValueError(f"Changed input {key}")
        np.testing.assert_array_equal(np.asarray(a["all_velocity_targets"])[fingers],np.asarray(b["all_velocity_targets"])[fingers])
        np.testing.assert_array_equal(b["actuator"]["submitted_velocity_rad_s"],c["actuator"]["v_path"])
        np.testing.assert_array_equal(a["actuator"]["submitted_velocity_rad_s"],np.zeros(6))
        np.testing.assert_array_equal(a["actuator"]["submitted_effort_Nm"],b["actuator"]["submitted_effort_Nm"])
        if a["actuator"]["limits"]!=b["actuator"]["limits"]:raise ValueError("Runtime limits changed")
        for key in ("command_time_s", "state_time_s", "physics_step"):
            if a["actuator"][key] != b["actuator"][key]:
                raise ValueError(f"Actuator timing changed: {key}")
    t=np.array([r["time_s"] for r in z]);dt=mz["physics_dt"]
    summary=dict(only_arm_velocity_target_changed=True, baseline_reproduction_deltas=baseline_deltas,
        source_velocity_trace=mp["velocity_path_source"],conditions={},changes={},
        diagnostic_note="Reversal/zero-crossing counts on a moving 0.9s trajectory are not alone evidence of instability.")
    all_signals={}
    for label,rows in (("zero",z),("v_path",p)):
        signals=[r["actuator"] for r in rows]
        a=lambda k:np.asarray([s[k] for s in signals],dtype=float)
        err=a("q_cmd")-a("q_actual");v=a("velocity_actual")
        accel=(v-np.vstack([signals[0]["velocity_pre"],v[:-1]]))/dt
        for key in ("q_actual","velocity_actual","submitted_velocity_rad_s","C_position_m","C_rotation_rad"):
            if not np.isfinite(a(key)).all():raise ValueError(f"Nonfinite {label}/{key}")
        c=dict(C={phase:{"position_m":scalar_stats(a("C_position_m")[mask],t[mask]),
                         "rotation_rad":scalar_stats(a("C_rotation_rad")[mask],t[mask])}
                  for phase,mask in {"all":np.ones(len(t),bool),"startup":t<=4*dt,"work":t>4*dt}.items()},
               joints={},peaks={}, object_initial_state=mz["initial"]["object_root_state"],
               object_final_state=rows[-1]["object_root_state"],
               object_max_rise_m=float(max(r["object_root_state"][2] for r in rows)-mz["initial"]["object_root_state"][2]),
               termination_events=[{"time_s":r["time_s"],"terms":r["termination_check"]} for r in rows
                                   if r.get("termination_check") and any(r["termination_check"].values())])
        for j,name in enumerate(mz["arm_names"]):
            limit=np.array(signals[0]["limits"]["position_limits"])[j]
            c["joints"][name]=dict(error_rad=scalar_stats(err[:,j],t),speed_rad_s=scalar_stats(v[:,j],t),
                max_acceleration_rad_s2=float(np.abs(accel[:,j]).max()),
                max_actual_step_rad=float(np.abs(np.diff(a("q_actual")[:,j])).max()),
                error_total_variation_rad=float(np.abs(np.diff(err[:,j])).sum()),
                error_sign_changes_above_005rad=sign_changes(err[:,j],.005),
                actual_velocity_reversals_above_05rad_s=sign_changes(v[:,j],.05),
                position_limit_margin_rad=float(np.minimum(a("q_actual")[:,j]-limit[0],limit[1]-a("q_actual")[:,j]).min()),
                speed_near_limit_fraction=float(np.mean(np.abs(v[:,j])>=.95*signals[0]["limits"]["velocity_limits"][j])))
        for time in (.575,.9):
            i=int(np.argmin(abs(t-time)))
            c["peaks"][str(time)]={"position_m":float(a("C_position_m")[i]),"rotation_rad":float(a("C_rotation_rad")[i]),
                                   "joint_error_rad":err[i].tolist(),"actual_velocity_rad_s":v[i].tolist()}
        summary["conditions"][label]=c;all_signals[label]=(signals,err,v)
    for kind in ("position_m","rotation_rad"):
        summary["changes"][kind]={stat:100*(summary["conditions"]["v_path"]["C"]["all"][kind][stat]/summary["conditions"]["zero"]["C"]["all"][kind][stat]-1) for stat in ("mean","p95","max")}
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(13,14),sharex=True)
    for label,(signals,err,v) in all_signals.items():
        a=lambda k:np.asarray([s[k] for s in signals])
        axes[0,0].plot(t,a("C_position_m")*1000,label=label)
        axes[0,1].plot(t,np.rad2deg(a("C_rotation_rad")),label=label)
        for row,j in enumerate((3,5),1):
            axes[row,0].plot(t,err[:,j],label=label)
            axes[row,1].plot(t,v[:,j],label=label+" actual")
        rows=z if label=="zero" else p
        obj=np.asarray([r["object_root_state"][:3] for r in rows])
        axes[3,0].plot(t,(obj[:,2]-mz["initial"]["object_root_state"][2])*1000,label=label)
        axes[3,1].plot(t,np.linalg.norm(obj[:,:2]-np.asarray(mz["initial"]["object_root_state"])[:2],axis=1)*1000,label=label)
    axes[0,0].set_ylabel("Wrist position error [mm]");axes[0,1].set_ylabel("Wrist rotation error [deg]")
    for row,j in enumerate((3,5),1):
        axes[row,0].set_ylabel(mz["arm_names"][j]+" joint error [rad]")
        axes[row,1].set_ylabel(mz["arm_names"][j]+" speed [rad/s]")
        axes[row,1].plot(t,np.asarray([s["v_path"] for s in all_signals["zero"][0]])[:,j],"--",color="gray",label="recorded v_path")
    for ax in axes.flat:
        for peak in (.575,.9):ax.axvline(peak,color="black",alpha=.15)
        ax.grid(alpha=.25);ax.legend()
    for ax in axes[-1]:ax.set_xlabel("Episode time [s]")
    axes[3,0].set_ylabel("Can rise from initial [mm]");axes[3,1].set_ylabel("Can XY displacement [mm]")
    fig.suptitle(f"Can {mz['condition']}, identical position/hand commands and physical settings\nOnly arm velocity target: zero vs recorded same-step v_path")
    fig.tight_layout();fig.savefig(out/"velocity_target_comparison.png",dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("zero",type=Path);p.add_argument("path",type=Path)
    p.add_argument("--original",type=Path,default=Path("outputs/diagnostics/arm_actuator_absent_20260907.jsonl"))
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();analyze(args.zero,args.path,args.original,args.output_dir)
