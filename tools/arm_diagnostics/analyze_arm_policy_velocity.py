"""Paired live-policy outcomes; do not force actions or episode lengths to match."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_arm_execution import analyze as analyze_execution
    from .analyze_arm_velocity import sign_changes
    from tools.rb3_revo2_ik.paired_arm_states import check_state
else:
    from analyze_arm_execution import analyze as analyze_execution
    from analyze_arm_velocity import sign_changes
    from tools.rb3_revo2_ik.paired_arm_states import check_state


def read_trace(path):
    with Path(path).open() as stream:
        records = [json.loads(line) for line in stream]
    if records[-1]["event"] != "trace_end":
        raise ValueError(f"Incomplete execution: {path}")
    return records[0], records


def compare(zero, path, output):
    mz,z=read_trace(zero); mp,p=read_trace(path)
    if mz["velocity_target_mode"] != "zero" or mp["velocity_target_mode"] != "v_path":
        raise ValueError("Expected live zero/v_path modes")
    if Path(mp["paired_initial_state_source"]).resolve()!=zero.resolve():
        raise ValueError("Changed run must restore this baseline state bank")
    for key in ("joint_names","user_joint_names","physics_dt","command_dt","decimation","checkpoint","reference",
                "interpolation_substeps","model_config","base_position","base_quaternion_xyzw","gravity","self_collisions",
                "all_stiffness","all_damping","all_effort_limits","all_velocity_limits","termination_config",
                "scale_pos","scale_rot","raw_clip","hydra_overrides","seed"):
        if mz[key]!=mp[key]:raise ValueError(f"Paired config differs: {key}")
    zi={r["episode"]:r for r in z if r["event"]=="initialization"}
    pi={r["episode"]:r for r in p if r["event"]=="initialization"}
    ze={r["episode"]:r for r in z if r["event"]=="episode_end"}
    pe={r["episode"]:r for r in p if r["event"]=="episode_end"}
    if set(ze)!=set(pe) or len(ze)!=20:raise ValueError("Expected 20 completed paired episodes")
    ids=[mz["user_joint_names"].index(n) for n in mz["joint_names"]]
    fingers=[j for j in range(len(mz["user_joint_names"])) if j not in ids]
    initial_deltas={}
    for ep in ze:
        zs=[r for r in z if r["event"]=="physics_sample" and r["episode"]==ep]
        ps=[r for r in p if r["event"]=="physics_sample" and r["episode"]==ep]
        initial_deltas[ep]=dict(reset=check_state(zi[ep],pi[ep]), first_physics=check_state(zs[0]["before_state"],ps[0]["before_state"]))
    # Deliberately compare only contracts, never force the two policies' actions
    # or position targets equal: each is responding to its actual observations.
    for meta,records in ((mz,z),(mp,p)):
        initial={r["episode"]:r for r in records if r["event"]=="initialization"}
        prior={}
        for row in records:
            if row["event"]!="physics_sample":continue
            ep=row["episode"]
            q=np.asarray(row["q_cmd"],dtype=np.float32)
            # before_state is sampled after apply_actions, so its target buffer
            # already contains q_cmd. The first predecessor is the RESET target.
            previous=prior.get(ep,np.asarray(initial[ep]["applied_arm_target"],dtype=np.float32))
            expected=(q-previous)/np.float32(meta["physics_dt"])
            prior[ep]=q
            if len(row["backend_velocity_calls"])!=1:raise ValueError("Velocity target overwritten")
            sent=np.asarray(row["backend_velocity_calls"][0])
            np.testing.assert_allclose(sent,expected if meta["velocity_target_mode"]=="v_path" else 0.,atol=2e-5,rtol=2e-6)
            np.testing.assert_array_equal(np.asarray(row["all_velocity_targets"])[fingers],0.)
            np.testing.assert_array_equal(row["all_effort_targets"],np.zeros(len(mz["user_joint_names"])))
            for key in ("all_joint_pos","all_joint_vel","object_root_state","actual_base_pos","actual_base_quat_xyzw","policy_action"):
                if not np.isfinite(row[key]).all():raise ValueError(f"Nonfinite {key}")
    output.mkdir(parents=True,exist_ok=False)
    result=dict(initial_state_max_deltas=initial_deltas, configurations_match=True,
                velocity_delivery_verified=True, trials=[],conditions={},
                note="Twenty paired placements, not a general success-rate estimate; success uses the unchanged reference-end criterion.")
    arrays={}
    for mode,file,records in (("zero",zero,z),("v_path",path,p)):
        analyze_execution(file,output/mode)
        summary=json.loads((output/mode/"summary.json").read_text())
        detail={e["episode"]:e for e in summary["episodes"]}
        global_errors=[];global_speed=[];global_accel=[]
        for ep in ze:
            rows=[r for r in records if r["event"]=="physics_sample" and r["episode"]==ep]
            a=lambda k:np.asarray([r[k] for r in rows])
            err=a("q_cmd")-a("q_actual");v=a("q_velocity")
            accel=(v-np.vstack([rows[0]["before_state"]["q_velocity"],v[:-1]]))/mz["physics_dt"]
            detail[ep].update(samples=len(rows),duration_s=rows[-1]["state_time_s"],
                joint_error_mean_rad=np.abs(err).mean(0).tolist(),joint_error_max_rad=np.abs(err).max(0).tolist(),
                speed_max_rad_s=np.abs(v).max(0).tolist(),acceleration_max_rad_s2=np.abs(accel).max(0).tolist(),
                velocity_reversals=[sign_changes(v[:,j],.05) for j in range(6)],
                joint_error_total_variation_rad=np.abs(np.diff(err,axis=0)).sum(0).tolist(),
                position_limit_violation=bool(np.any(a("q_actual")<np.asarray(mz["actual_position_limits"])[:,0]-1e-6) or
                                              np.any(a("q_actual")>np.asarray(mz["actual_position_limits"])[:,1]+1e-6)),
                speed_near_limit_fraction=np.mean(np.abs(v)>=.95*np.asarray(mz["actual_velocity_limits"]),axis=0).tolist())
            global_errors.extend(err);global_speed.extend(v);global_accel.extend(accel)
        result["conditions"][mode]=dict(successes=summary["successes"],failed_episodes=summary["failed_episodes"],
            statistics=summary["statistics"],solver=summary["solver"],delivery=summary["delivery"],episodes=detail,
            joint_error_mean_rad=np.abs(global_errors).mean(0).tolist(),joint_error_max_rad=np.abs(global_errors).max(0).tolist(),
            speed_max_rad_s=np.abs(global_speed).max(0).tolist(),acceleration_max_rad_s2=np.abs(global_accel).max(0).tolist())
        arrays[mode]=detail
    for ep in ze:
        x,y=arrays["zero"][ep],arrays["v_path"][ep]
        result["trials"].append(dict(episode=ep,initial_object_xyz=zi[ep]["object_pos"],
            zero_success=x["termination"]["success"],v_path_success=y["termination"]["success"],
            zero_termination=x["termination"],v_path_termination=y["termination"],
            zero_final_lift_m=x["final_rise_m"],v_path_final_lift_m=y["final_rise_m"]))
    result["failure_to_success"]=[r["episode"] for r in result["trials"] if not r["zero_success"] and r["v_path_success"]]
    result["success_to_failure"]=[r["episode"] for r in result["trials"] if r["zero_success"] and not r["v_path_success"]]
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (output/"placements.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result["trials"][0]));writer.writeheader();writer.writerows(result["trials"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for mode in arrays:
        entries=list(arrays[mode].values()); x=[e["episode"] for e in entries]
        axes[0,0].plot(x,[e["statistics"]["C_tracking_position_m"]["mean"]*1000 for e in entries],"o-",label=mode)
        axes[0,1].plot(x,[np.rad2deg(e["statistics"]["C_tracking_rotation_rad"]["mean"]) for e in entries],"o-",label=mode)
        axes[1,0].plot(x,[e["final_rise_m"]*1000 for e in entries],"o-",label=mode)
        axes[1,1].plot(x,[e["acceleration_max_rad_s2"][5] for e in entries],"o-",label=mode)
    for ax,label in zip(axes.flat,("Mean wrist error [mm]","Mean wrist rotation error [deg]","Final can lift [mm]","Peak wrist3 acceleration [rad/s²]")):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend();ax.set_xticks(list(ze));ax.set_xlabel("Placement index (0-based)")
    fig.suptitle("Same 20 explicit reset states; independent live policy + IK in each condition")
    fig.tight_layout();fig.savefig(output/"paired_policy_comparison.png",dpi=150);plt.close(fig)
    print(json.dumps({k:result[k] for k in ("failure_to_success","success_to_failure","trials")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zero",type=Path);parser.add_argument("path",type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args();compare(args.zero,args.path,args.output_dir)
