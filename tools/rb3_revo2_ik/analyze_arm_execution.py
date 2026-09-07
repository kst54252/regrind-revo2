"""Separate IK, command processing, and physical tracking errors in JSONL traces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__:
    from .rb3_kinematics import RB3730Kinematics
    from .analyze_arm_tracking import quaternion_error_xyzw
else:
    from rb3_kinematics import RB3730Kinematics
    from analyze_arm_tracking import quaternion_error_xyzw


def pose_errors(p1, q1, p2, q2):
    return np.linalg.norm(p1-p2, axis=1), quaternion_error_xyzw(q1, q2)


def statistics(values, samples, mask):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return None
    selected = values[indices]
    i = int(indices[np.argmax(selected)])
    row = samples[i]
    return dict(mean=float(np.mean(selected)), p95=float(np.percentile(selected, 95)),
                max=float(values[i]), at=dict(episode=row["episode"],
                    time_s=row["state_time_s"], physics_step=row["physics_step"],
                    command_id=row["command_id"], reference_frame=row["reference_frame_at_command"],
                    interpolation_step=row["interpolation_step"]))


def analyze(path, out):
    out.mkdir(parents=True, exist_ok=False)
    with path.open() as stream:
        records = [json.loads(line) for line in stream]
    meta = next(r for r in records if r["event"] == "metadata")
    samples = [r for r in records if r["event"] == "physics_sample"]
    if not samples:
        raise ValueError("no physics samples")
    kin = RB3730Kinematics(model_config=meta["model_config"], base_position=meta["base_position"],
                           base_quaternion_xyzw=meta["base_quaternion_xyzw"])
    # Every logged array is already mapped by name. Still enforce the contract
    # here so future traces with a different order cannot silently compare.
    order = [meta["joint_names"].index(n) for n in kin.joint_names]
    a = lambda key: np.asarray([s[key] for s in samples], dtype=float)
    ik_q, cmd_q, actual_q = [a(k)[:, order] for k in ("q_ik", "q_cmd", "q_actual")]
    ik_pos, ik_quat = kin.forward_batch(ik_q)
    cmd_pos, cmd_quat = kin.forward_batch(cmd_q)
    actual_fk_pos, actual_fk_quat = kin.forward_batch(actual_q)
    errors = {}
    for name, poses in {
        "A_ik": (a("ik_input_pos"), a("ik_input_quat_xyzw"), ik_pos, ik_quat),
        "B_command": (ik_pos, ik_quat, cmd_pos, cmd_quat),
        "C_tracking": (cmd_pos, cmd_quat, a("actual_base_pos"), a("actual_base_quat_xyzw")),
        "total": (a("ik_input_pos"), a("ik_input_quat_xyzw"), a("actual_base_pos"), a("actual_base_quat_xyzw")),
        "FK_runtime_consistency": (actual_fk_pos, actual_fk_quat, a("actual_base_pos"), a("actual_base_quat_xyzw")),
    }.items():
        errors[name+"_position_m"], errors[name+"_rotation_rad"] = pose_errors(*poses)
    if not all(np.isfinite(v).all() for v in errors.values()):
        raise ValueError("nonfinite error data; inspect raw solver/trace status")

    # Distinguish first policy interval from work; this is an analysis label,
    # NOT an inserted hold, settle step, or claim about physical contact phase.
    startup = np.asarray([s["episode_physics_step"] <= meta["decimation"] for s in samples])
    masks = {"startup_first_control_interval": startup, "work": ~startup,
             "all": np.ones(len(samples), dtype=bool)}
    summary = dict(trace=str(path.resolve()), metadata=meta, samples=len(samples),
                   phase_definition="Reset snapshots separate; first control interval = startup; remaining samples = work. No explicit warmup exists.",
                   statistics={region: {k: statistics(v, samples, mask) for k, v in errors.items()}
                               for region, mask in masks.items()})
    summary["solver"] = {
        "failed_command_ids": sorted({s["command_id"] for s in samples if not s["ik_success"]}),
        "fallback_command_ids": sorted({s["command_id"] for s in samples if s["fallback"] != "none"}),
        "optimizer_failure_ids": sorted({s["command_id"] for s in samples if not s["solver_success"]}),
    }
    # Compare all observed delivery boundaries, including native setter inputs
    # after mapping the backend's joint names back into the canonical order.
    delivery = {}
    for key in ("q_pre_send", "q_post_actuator", "q_pre_physics_buffer", "q_after_physics_buffer"):
        delivery[key+"_max_delta_rad"] = float(np.max(np.abs(a(key)[:, order]-cmd_q)))
    for kind in ("position", "velocity"):
        counts = [len(s[f"backend_{kind}_calls"]) for s in samples]
        delivery[f"backend_{kind}_calls_min_max"] = [min(counts), max(counts)]
    if any(not s["backend_position_calls"] for s in samples):
        raise ValueError("missing actual simulator setter observation")
    sent = np.asarray([s["backend_position_calls"][-1] for s in samples])[:, order]
    delivery["actual_backend_setter_max_delta_rad"] = float(np.max(np.abs(sent-cmd_q)))
    delivery["backend_overwrite_samples"] = [s["physics_step"] for s in samples
        if any(not np.array_equal(v, s["q_cmd"]) for v in s["backend_position_calls"])]
    delivery["velocity_target_max_abs_rad_s"] = float(max(
        np.abs(v).max() for s in samples for v in s["backend_velocity_calls"]))
    summary["delivery"] = delivery
    summary["initialization"] = [r for r in records if r["event"] == "initialization"]
    summary["episodes"] = []
    init = {r["episode"]: r for r in summary["initialization"]}
    ends = {r["episode"]: r for r in records if r["event"] == "episode_end"}
    episode_ids = np.asarray([s["episode"] for s in samples])
    for episode in sorted(set(episode_ids.tolist())):
        ids = np.flatnonzero(episode_ids == episode)
        rows = [samples[i] for i in ids]
        dz = np.asarray([r["object_pos"][2] for r in rows])-init[episode]["object_pos"][2]
        raised = np.flatnonzero(dz > 0.02)
        detail = dict(episode=episode, start=init[episode]["object_pos"],
                      termination=ends.get(episode, {}).get("termination", {}),
                      max_rise_m=float(dz.max()), final_rise_m=float(dz[-1]),
                      first_rise_over_2cm_s=rows[raised[0]]["state_time_s"] if len(raised) else None,
                      contact_time="unconfirmed", grasp_time="unconfirmed",
                      statistics={k: statistics(v, samples, episode_ids == episode) for k, v in errors.items()})
        summary["episodes"].append(detail)
    summary["successes"] = sum(e["termination"].get("success", False) for e in summary["episodes"])
    summary["failed_episodes"] = [e["episode"] for e in summary["episodes"]
                                   if e["termination"] and not e["termination"].get("success", False)]
    # Last substep is the same sampling boundary used in old control-rate logs.
    last = np.asarray([s["interpolation_step"] == meta["interpolation_substeps"] for s in samples])
    summary["last_substep_statistics"] = {k: statistics(v, samples, last) for k, v in errors.items()}
    with (out/"summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    with (out/"errors.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "physics_step", "command_id", "reference_frame",
                                              "time_s", "region", "interpolation_step"] + list(errors))
        writer.writeheader()
        for i, row in enumerate(samples):
            writer.writerow(dict(episode=row["episode"], physics_step=row["physics_step"],
                                 command_id=row["command_id"], reference_frame=row["reference_frame_at_command"],
                                 time_s=row["state_time_s"], interpolation_step=row["interpolation_step"],
                                 region="startup" if startup[i] else "work",
                                 **{k: float(v[i]) for k,v in errors.items()}))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for episode in [summary["failed_episodes"][0] if summary["failed_episodes"] else 0]:
        mask = episode_ids == episode
        times = a("state_time_s")[mask]
        fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
        for index, stage in enumerate(("A_ik", "B_command", "C_tracking")):
            for col, (suffix, scale, units) in enumerate((("position_m", 1000, "mm"), ("rotation_rad", 180/np.pi, "deg"))):
                ax = axes[index, col]
                ax.plot(times, errors[stage+"_"+suffix][mask]*scale)
                ax.set_ylabel(f"{stage} [{units}]")
                ax.grid(alpha=0.3)
                ax.axvspan(0, meta["command_dt"], alpha=0.12, color="gray")
        dz = a("object_pos")[mask, 2]-init[episode]["object_pos"][2]
        axes[3, 0].plot(times, dz*1000)
        axes[3, 0].set_ylabel("Object rise [mm]")
        axes[3, 1].plot(times, a("q_velocity")[mask])
        axes[3, 1].set_ylabel("Measured joint speed [rad/s]")
        axes[3, 1].legend(meta["joint_names"], fontsize=7, ncol=3)
        ep = next(e for e in summary["episodes"] if e["episode"] == episode)
        for ax in axes.flat:
            if ep["first_rise_over_2cm_s"] is not None:
                ax.axvline(ep["first_rise_over_2cm_s"], color="green", linestyle="--", alpha=0.5)
        for ax in axes[-1]:
            ax.set_xlabel("Episode simulation time [s]")
        fig.suptitle(f"Episode {episode}: same-substep A/B/C errors\nGray = startup interval; green = object rise >2cm (not confirmed grasp/contact)")
        fig.tight_layout()
        fig.savefig(out/f"stage_errors_episode_{episode:03d}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for e in summary["episodes"]:
        mask = episode_ids == e["episode"]
        ax.plot(a("state_time_s")[mask], (a("object_pos")[mask, 2]-init[e["episode"]]["object_pos"][2])*1000,
                color="red" if e["episode"] in summary["failed_episodes"] else "steelblue", alpha=0.55,
                label=f"failed episode {e['episode']}" if e["episode"] in summary["failed_episodes"] else None)
    ax.set(xlabel="Episode simulation time [s]", ylabel="Object rise [mm]",
           title="Actual object rise: blue = task success, red = task failure")
    ax.grid(alpha=0.3)
    if summary["failed_episodes"]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/"object_rise_all_episodes.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("samples", "successes", "failed_episodes", "solver", "delivery", "statistics")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.trace, args.output_dir)
