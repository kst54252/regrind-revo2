"""Analyze recorded-command can-present/absent experiments without Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .rb3_kinematics import RB3730Kinematics
    from .analyze_arm_execution import pose_errors
else:
    from rb3_kinematics import RB3730Kinematics
    from analyze_arm_execution import pose_errors


def read_run(path):
    with Path(path).open() as stream:
        records = [json.loads(line) for line in stream]
    if records[-1]["event"] != "complete":
        raise ValueError(f"Incomplete contact replay: {path}")
    return records[0], [r for r in records if r["event"] == "sample"]


def verify_pair(meta_on, on, meta_off, off):
    if meta_on["condition"] != "present" or meta_off["condition"] != "absent":
        raise ValueError("Expected present and absent, respectively")
    for key in ("source", "episode", "physics_dt", "joint_names", "actual_stiffness",
                "actual_damping", "actual_effort_limits", "actual_velocity_limits"):
        if meta_on[key] != meta_off[key]:
            raise ValueError(f"Different paired metadata: {key}")
    if len(on) != len(off) or not on:
        raise ValueError("Paired samples must have identical nonzero lengths")
    for a, b in zip(on, off):
        for key in ("time_s", "source_physics_step", "reference_frame", "interpolation_step",
                    "all_position_targets", "all_velocity_targets", "all_effort_targets",
                    "backend_position_targets"):
            if a[key] != b[key]:
                raise ValueError(f"Pair differs in {key} at time {a['time_s']}")


def stats(values, rows, mask):
    ids = np.flatnonzero(mask)
    i = int(ids[np.argmax(values[ids])])
    return dict(mean=float(values[ids].mean()), p95=float(np.percentile(values[ids], 95)),
                max=float(values[i]), at_time_s=rows[i]["time_s"],
                at_reference_frame=rows[i]["reference_frame"])


def analyze(present, absent, out):
    meta, on = read_run(present)
    off_meta, off = read_run(absent)
    verify_pair(meta, on, off_meta, off)
    with Path(meta["source"]).open() as stream:
        source = [json.loads(line) for line in stream]
    original = [r for r in source if r["event"] == "physics_sample" and r["episode"] == meta["episode"]]
    if [r["physics_step"] for r in original] != [r["source_physics_step"] for r in on]:
        raise ValueError("Source timeline differs from paired replay")
    srcmeta = next(r for r in source if r["event"] == "metadata")
    ids = [meta["joint_names"].index(n) for n in meta["arm_names"]]
    kin = RB3730Kinematics(model_config=meta["model_config"], base_position=meta["base_position"],
                           base_quaternion_xyzw=meta["base_quaternion_xyzw"])
    a = lambda rows, key: np.asarray([r[key] for r in rows], dtype=float)
    target_pos, target_quat = kin.forward_batch(a(on, "all_position_targets")[:, ids])
    times = a(on, "time_s")
    errors = {}
    for name, rows in (("present", on), ("absent", off), ("source", original)):
        errors[name+"_position_m"], errors[name+"_rotation_rad"] = pose_errors(
            target_pos, target_quat, a(rows, "actual_base_pos"), a(rows, "actual_base_quat_xyzw"))
    for name, left, right in (("paired_actual_difference", on, off), ("reproduction_difference", on, original)):
        errors[name+"_position_m"], errors[name+"_rotation_rad"] = pose_errors(
            a(left, "actual_base_pos"), a(left, "actual_base_quat_xyzw"),
            a(right, "actual_base_pos"), a(right, "actual_base_quat_xyzw"))
    if not all(np.isfinite(v).all() for v in errors.values()):
        raise ValueError("Nonfinite pose errors")
    masks = {"startup": times <= srcmeta["command_dt"], "work": times > srcmeta["command_dt"],
             "all": np.ones(len(on), dtype=bool)}
    obj0 = np.asarray(meta["initial"]["object_root_state"])
    summary = dict(episode=meta["episode"], samples=len(on), source=meta["source"],
                   present=str(present), absent=str(absent), target_arrays_identical=True,
                   gains_limits_dt_identical=True,
                   initial_state_max_deltas={"present":meta["initial_state_max_deltas"],
                                             "absent":off_meta["initial_state_max_deltas"]},
                   policy_calls_in_test=0, ik_calls_in_test=0,
                   statistics={phase:{key:stats(value,on,mask) for key,value in errors.items()}
                               for phase,mask in masks.items() if mask.any()},
                   absent_min_can_to_base_distance_m=float(np.linalg.norm(
                       a(off,"object_root_state")[:,:3]-a(off,"actual_base_pos"),axis=1).min()),
                   contact_scope="Only can removed from reach; table and robot self-contact unchanged.",
                   torque_conclusion="Unconfirmed: no measured drive torque collected.")
    for label, rows in (("present",on),("absent",off)):
        joints = a(rows,"all_joint_pos")[:,ids]
        qerr = np.abs(joints-a(rows,"all_position_targets")[:,ids])
        summary[label+"_arm_joint_error_max_rad"] = qerr.max(axis=0).tolist()
        summary[label+"_arm_speed_max_rad_s"] = np.abs(a(rows,"all_joint_vel")[:,ids]).max(axis=0).tolist()
    summary["present_can_max_rise_m"] = float((a(on,"object_root_state")[:,2]-obj0[2]).max())
    out.mkdir(parents=True, exist_ok=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    header = ["time_s","reference_frame"] + list(errors)
    np.savetxt(out/"errors.csv",np.column_stack([times,a(on,"reference_frame")]+list(errors.values())),
               delimiter=",",header=",".join(header),comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True)
    for col, (suffix,scale,unit) in enumerate((("position_m",1000,"mm"),("rotation_rad",180/np.pi,"deg"))):
        for label,color in (("source","gray"),("present","tab:red"),("absent","tab:blue")):
            axes[0,col].plot(times,errors[label+"_"+suffix]*scale,label=label,color=color,
                             linestyle="--" if label=="source" else "-")
        axes[0,col].set_ylabel(f"C: commanded FK vs actual [{unit}]")
        axes[0,col].legend()
    axes[1,0].plot(times,errors["paired_actual_difference_position_m"]*1000,color="purple")
    axes[1,0].set_ylabel("Present vs absent actual wrist gap [mm]")
    axes[1,1].plot(times,(a(on,"object_root_state")[:,2]-obj0[2])*1000,color="tab:red",label="present")
    axes[1,1].plot(times,(a(original,"object_pos")[:,2]-obj0[2])*1000,"--",color="gray",label="source")
    axes[1,1].set_ylabel("Can rise [mm]; absent can is out of reach")
    axes[1,1].legend()
    for ax in axes.flat:
        ax.axvspan(0,srcmeta["command_dt"],color="gray",alpha=.12)
        ax.grid(alpha=.3)
    for ax in axes[1]: ax.set_xlabel("Episode simulation time [s]")
    fig.suptitle(f"Episode {meta['episode']}: identical recorded arm + hand commands; no policy feedback\nOnly intervention: can moved +10 m X in absent run; gains, limits and dt unchanged")
    fig.tight_layout()
    fig.savefig(out/"contact_comparison.png",dpi=160)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("present",type=Path)
    parser.add_argument("absent",type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args = parser.parse_args()
    analyze(args.present,args.absent,args.output_dir)
