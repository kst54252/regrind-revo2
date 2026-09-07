"""Isolate can contact using recorded, identical arm AND hand drive targets.

Invoked only by play.py's explicit diagnostic flag. Earlier episodes are run
normally to preserve PhysX history; the selected episode uses no policy or IK.
The absent condition moves only the can +10 m in world X once, without changing
collision properties, gravity, drive parameters, or robot/table geometry.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rb3_revo2_ik.trace_arm_execution import array, serial


def read_episode(path, episode):
    with Path(path).open() as stream:
        records = [json.loads(line) for line in stream]
    meta = next(r for r in records if r["event"] == "metadata")
    rows = [r for r in records if r["event"] == "physics_sample" and r["episode"] == episode]
    if not meta.get("full_state") or not rows:
        raise ValueError("Need a full-state trace containing the requested episode")
    required = ("all_position_targets", "all_velocity_targets", "all_effort_targets")
    for row in rows:
        for key in required:
            value = np.asarray(row[key])
            if value.shape != (len(meta["user_joint_names"]),) or not np.isfinite(value).all():
                raise ValueError(f"Invalid recorded {key}")
    return meta, rows


def run(wrapped, obs, policy, args, checkpoint):
    env = wrapped.unwrapped
    command = env.command_manager.get_term("reference")
    robot, obj = command.robot, command.object
    meta, rows = read_episode(args.arm_contact_replay, args.arm_contact_episode)
    output = Path(args.arm_contact_output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if env.num_envs != 1 or env._physics_handles_decimation:
        raise ValueError("This experiment requires one standard PhysX environment")
    if env.event_manager.active_terms:
        raise ValueError("Contact replay requires the existing deterministic Play config (no events)")
    if Path(checkpoint).resolve() != Path(meta["checkpoint"]).resolve():
        raise ValueError("Prefix must use the recorded checkpoint")
    if Path(command.reference.path).resolve() != Path(meta["reference"]).resolve():
        raise ValueError("Reference differs from source trace")
    if env.cfg.seed != meta["seed"] or env.physics_dt != meta["physics_dt"] or env.step_dt != meta["command_dt"]:
        raise ValueError("Seed or timing differs from source trace")
    names = robot.joint_names
    order = [meta["user_joint_names"].index(n) for n in names]
    arm_ids = [names.index(n) for n in meta["joint_names"]]
    backend_ids = [robot.backend_joint_names.index(n) for n in names]
    velocity_path = None
    if args.arm_velocity_path_trace:
        from tools.arm_diagnostics.analyze_arm_velocity import load_velocity_path
        velocity_path = load_velocity_path(args.arm_velocity_path_trace, meta, rows, args.arm_contact_condition)
    for field, key in (("joint_stiffness", "actual_stiffness"), ("joint_damping", "actual_damping"),
                       ("joint_effort_limits", "actual_effort_limits"), ("joint_vel_limits", "actual_velocity_limits")):
        np.testing.assert_array_equal(array(getattr(robot.data, field))[0, arm_ids], meta[key])

    # Preserve reset/contact history rather than approximating a mid-run PhysX
    # snapshot with only positions. Prefix policy calls stop before the test.
    completed = 0
    with torch.inference_mode():
        for _ in range((args.arm_contact_episode + 1) * command.reference.frames * 4):
            if completed == args.arm_contact_episode:
                break
            obs, _, dones, _ = wrapped.step(policy(obs))
            completed += int(bool(dones[0]))
        else:
            raise RuntimeError("Prefix did not reach requested episode")

    def state():
        return dict(all_joint_pos=array(robot.data.joint_pos)[0],
                    all_joint_vel=array(robot.data.joint_vel)[0],
                    robot_root_state=array(robot.data.root_state_w)[0],
                    object_root_state=array(obj.data.root_state_w)[0],
                    actual_base_pos=array(command.current_hand_wrist_pos)[0]+array(env.scene.env_origins)[0],
                    actual_base_quat_xyzw=array(command.current_hand_wrist_quat)[0])

    initial = state()
    expected = rows[0]["before_state"]
    deltas = {}
    for key in ("all_joint_pos", "all_joint_vel", "robot_root_state", "object_root_state"):
        reference = np.asarray(expected[key])
        if key.startswith("all_joint"):
            reference = reference[order]
        deltas[key] = float(np.max(np.abs(initial[key]-reference)))
        if deltas[key] > 1e-6:
            raise RuntimeError(f"Prefix state not reproduced: {key} max delta {deltas[key]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    def write(stream, event, **data):
        stream.write(json.dumps(dict(event=event, **data), default=serial)+"\n")
        stream.flush()

    # Intercept actual simulator position setter to verify replayed delivery.
    setter = robot.root_view.set_dof_position_targets
    sent = []
    def observe(data, *a, **kw):
        sent.append(array(data)[0, backend_ids])
        return setter(data, *a, **kw)
    robot.root_view.set_dof_position_targets = observe
    probe = None
    try:
        if args.arm_actuator_diagnostic:
            from tools.rb3_revo2_ik.arm_actuator_probe import ArmActuatorProbe
            probe = ArmActuatorProbe(env, meta, rows, args)
        with output.open("x") as stream, torch.inference_mode():
            write(stream, "metadata", source=str(Path(args.arm_contact_replay).resolve()),
                  condition=args.arm_contact_condition, episode=args.arm_contact_episode,
                  joint_names=names, arm_names=meta["joint_names"], physics_dt=env.physics_dt,
                  model_config=meta["model_config"], base_position=meta["base_position"],
                  base_quaternion_xyzw=meta["base_quaternion_xyzw"],
                  initial_state_max_deltas=deltas, initial=initial,
                  physics_steps=len(rows), prefix_episodes=completed,
                  policy_calls_in_test=0, ik_calls_in_test=0,
                  torque_note="Effort targets are feedforward commands, not measured drive torques.",
                  intervention="can +10 m world X" if args.arm_contact_condition == "absent" else "none",
                  actual_stiffness=array(robot.data.joint_stiffness)[0],
                  actual_damping=array(robot.data.joint_damping)[0],
                  actual_effort_limits=array(robot.data.joint_effort_limits)[0],
                  actual_velocity_limits=array(robot.data.joint_vel_limits)[0],
                  actuator_diagnostic=probe.metadata if probe else None,
                  velocity_path_source=str(Path(args.arm_velocity_path_trace).resolve()) if velocity_path is not None else None)
            if args.arm_contact_condition == "absent":
                pose = torch.as_tensor(initial["object_root_state"][:7], device=env.device).unsqueeze(0).clone()
                pose[:, 0] += 10.0
                obj.write_root_link_pose_to_sim_index(root_pose=pose)
            for sample_index, row in enumerate(rows):
                sent.clear()
                targets = {}
                for key, method in (("all_position_targets", robot.set_joint_position_target_index),
                                    ("all_velocity_targets", robot.set_joint_velocity_target_index),
                                    ("all_effort_targets", robot.set_joint_effort_target_index)):
                    values = torch.as_tensor(np.asarray(row[key])[order], dtype=torch.float32, device=env.device).unsqueeze(0)
                    if key == "all_velocity_targets" and velocity_path is not None:
                        values[0, arm_ids] = torch.as_tensor(velocity_path[sample_index], dtype=torch.float32, device=env.device)
                    method(target=values)
                    targets[key] = array(values)[0]
                env._sim_step_counter += 1
                if probe:
                    probe.before_send(row)
                env.scene.write_data_to_sim()
                if probe:
                    probe.after_send()
                if len(sent) != 1:
                    raise RuntimeError(f"Expected one simulator position setter, got {len(sent)}")
                np.testing.assert_array_equal(sent[0], targets["all_position_targets"])
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                # Evaluate the existing terms at the original control boundaries,
                # without terminating, resetting or extending recorded commands.
                termination_check = None
                if row["interpolation_step"] == meta["interpolation_substeps"]:
                    saved_frame = command.time_steps.clone()
                    saved_length = env.episode_length_buf.clone()
                    try:
                        command.time_steps[:] = row["reference_frame_at_command"]
                        env.episode_length_buf[:] = (sample_index + 1) // meta["interpolation_substeps"]
                        termination_check = {}
                        for name in env.termination_manager.active_terms:
                            cfg = env.termination_manager.get_term_cfg(name)
                            termination_check[name] = bool(cfg.func(env, **cfg.params)[0])
                    finally:
                        command.time_steps.copy_(saved_frame)
                        env.episode_length_buf.copy_(saved_length)
                write(stream, "sample", time_s=row["state_time_s"],
                      source_physics_step=row["physics_step"], physics_step=env._sim_step_counter,
                      reference_frame=row["reference_frame_at_command"],
                      interpolation_step=row["interpolation_step"], backend_position_targets=sent[0],
                      **targets, **state(), termination_check=termination_check,
                      actuator=probe.after_step(row) if probe else None)
            write(stream, "complete", samples=len(rows))
    finally:
        if probe:
            probe.close()
        robot.root_view.set_dof_position_targets = setter
    print(f"[contact comparison] {args.arm_contact_condition}: {output}; initial deltas={deltas}")
