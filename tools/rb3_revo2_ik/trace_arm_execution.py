"""Opt-in observers on the real play-arm call path; never change commands/state.

Callbacks always invoke each original bound method exactly once. No extra sim
steps, IK solves, actuator computations or random draws. GPU reads can increase
wall-clock execution time, not simulated dt. Records are flushed for crash use.
"""
from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp


def array(value):
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, wp.array):
        value = wp.to_torch(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class ArmExecutionTrace:
    def __init__(self, env, path, checkpoint, cli, hydra, full_state=False):
        self.env = env
        self.full_state = full_state
        self.full_commands = {}
        self.command = env.command_manager.get_term("reference")
        self.action = env.action_manager.get_term("root_pose")
        self.robot = self.command.robot
        self.kin = self.action._kinematics
        if env._physics_handles_decimation:
            raise RuntimeError("substep trace requires the existing per-step PhysX loop")
        if getattr(self.robot, "_has_newton_actuators", False):
            raise RuntimeError("trace supports the current standard Isaac implicit actuator path")
        self.names = list(self.kin.joint_names)
        self.ids = [self.robot.joint_names.index(n) for n in self.names]
        self.backend_ids = [self.robot.backend_joint_names.index(n) for n in self.names]
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8")
        self.started = time.perf_counter()
        self.episode = 0
        self.episode_step = 0
        self.command_id = -1
        self.context = None
        self.in_process = False
        self.pending = False
        self.writes = []
        self.position_setter_calls = []
        self.velocity_setter_calls = []
        self.restores = []
        actuator = self.robot.actuators["rb3_arm"]
        methods = {
            "process_actions": self.action.process_actions,
            "inverse": self.kin.inverse,
            "apply_actions": self.action.apply_actions,
            "write_data_to_sim": self.robot.write_data_to_sim,
            "backend_position_setter": self.robot.root_view.set_dof_position_targets,
            "physics_step": env.sim.step, "scene_update": env.scene.update,
            "env_step": env.step, "reference_update": self.command._update_command,
            "actuator_compute": actuator.compute,
        }
        locations = {}
        for name, method in methods.items():
            try:
                locations[name] = {"file": inspect.getsourcefile(method), "line": inspect.getsourcelines(method)[1]}
            except (TypeError, OSError):
                locations[name] = {"file": "native/unavailable"}
        self.emit("metadata", joint_names=self.names, user_joint_names=self.robot.joint_names,
                  backend_joint_names=self.robot.backend_joint_names, user_ids=self.ids, backend_ids=self.backend_ids,
                  physics_dt=env.physics_dt, command_dt=env.step_dt, decimation=env.cfg.decimation,
                  interpolation_substeps=self.action.cfg.interpolation_substeps,
                  checkpoint=str(checkpoint), reference=str(self.command.reference.path),
                  cli=cli, hydra_overrides=hydra, source_locations=locations,
                  seed=env.cfg.seed, base_position=self.kin.base_position,
                  base_quaternion_xyzw=__import__('scipy').spatial.transform.Rotation.from_matrix(self.kin.base_rotation).as_quat(),
                  model_config=str(self.kin.model_config_path),
                  actuator_type=type(actuator).__module__ + "." + type(actuator).__name__,
                  actual_stiffness=self.joints(self.robot.data.joint_stiffness),
                  actual_damping=self.joints(self.robot.data.joint_damping),
                  actual_effort_limits=self.joints(self.robot.data.joint_effort_limits),
                  actual_velocity_limits=self.joints(self.robot.data.joint_vel_limits),
                  actual_position_limits=array(self.robot.data.soft_joint_pos_limits)[0, self.ids],
                  reference_pose_source=self.action.cfg.base_action_source,
                  scale_pos=self.action.cfg.scale_pos, scale_rot=self.action.cfg.scale_rot,
                  raw_clip=self.action.cfg.raw_clip,
                  quaternion_order="xyzw", pose_frame="world / right_hand_base_link",
                  env_origin=array(env.scene.env_origins)[0], full_state=full_state,
                  velocity_target_mode=self.action.cfg.velocity_target_mode,
                  paired_initial_state_source=getattr(getattr(env, "_paired_arm_states", None), "path", None),
                  gravity=env.cfg.sim.gravity,
                  self_collisions=self.robot.cfg.spawn.articulation_props.enabled_self_collisions,
                  all_stiffness=array(self.robot.data.joint_stiffness)[0],
                  all_damping=array(self.robot.data.joint_damping)[0],
                  all_effort_limits=array(self.robot.data.joint_effort_limits)[0],
                  all_velocity_limits=array(self.robot.data.joint_vel_limits)[0],
                  termination_config={name:dict(function=env.termination_manager.get_term_cfg(name).func.__name__,
                                               params=env.termination_manager.get_term_cfg(name).params)
                                      for name in env.termination_manager.active_terms},
                  torque_note="No torque measurement collected. ImplicitActuator computed/applied_effort are approximate PD estimates, not measured PhysX motor torque.",
                  contact_note="No new contact sensor; contact/grasp time unconfirmed. Can rise is only a kinematic proxy.")
        self.initialization()
        self.wrap(self.kin, "inverse", self.inverse)
        self.wrap(self.action, "process_actions", self.process)
        self.wrap(self.action, "apply_actions", self.apply)
        self.wrap(self.robot, "set_joint_position_target_index", self.joint_write)
        self.wrap(self.robot, "write_data_to_sim", self.send)
        self.wrap(self.robot.root_view, "set_dof_position_targets", self.backend_position)
        self.wrap(self.robot.root_view, "set_dof_velocity_targets", self.backend_velocity)
        self.wrap(env.sim, "step", self.physics)
        self.wrap(env.scene, "update", self.update)
        self.wrap(env, "_reset_idx", self.reset)
        self.wrap(self.command, "_update_command", self.command_update)

    def wrap(self, obj, name, observer):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return observer(original, *args, **kwargs)
        setattr(obj, name, wrapper)
        self.restores.append((obj, name, original))

    def joints(self, value):
        return array(value)[0, self.ids]

    def initialization(self):
        state = self.state()
        bank = getattr(self.env, "_paired_arm_states", None)
        deltas = bank.verify(self.episode, state) if bank else None
        self.emit("initialization", **state, paired_max_deltas=deltas)

    def emit(self, event, **fields):
        record = dict(event=event, wall_elapsed_s=time.perf_counter()-self.started,
                      physics_step=int(self.env._sim_step_counter), episode=self.episode,
                      command_id=self.command_id, **fields)
        self.file.write(json.dumps(record, default=serial) + "\n")
        self.file.flush()

    def state(self):
        origin = array(self.env.scene.env_origins)[0]
        state = dict(q_actual=self.joints(self.robot.data.joint_pos),
                    q_velocity=self.joints(self.robot.data.joint_vel),
                    actual_base_pos=array(self.command.current_hand_wrist_pos)[0]+origin,
                    actual_base_quat_xyzw=array(self.command.current_hand_wrist_quat)[0],
                    object_pos=array(self.command.current_object_pos)[0]+origin,
                    object_quat_xyzw=array(self.command.current_object_quat)[0],
                    reference_frame=int(self.command.time_steps[0]),
                    placement_offset=array(self.command.placement_offset)[0],
                    applied_arm_target=array(self.action.applied_joint_target)[0])
        if self.full_state:
            state.update(all_joint_pos=array(self.robot.data.joint_pos)[0],
                         all_joint_vel=array(self.robot.data.joint_vel)[0],
                         robot_root_state=array(self.robot.data.root_state_w)[0],
                         object_root_state=array(self.command.object.data.root_state_w)[0])
        return state

    def process(self, original, actions):
        self.command_id += 1
        self.context = dict(reference_frame_at_command=int(self.command.time_steps[0]),
                            command_physics_step=int(self.env._sim_step_counter),
                            policy_wrist_action=array(actions)[0],
                            policy_action=array(self.env.action_manager.action)[0],
                            reference_pos=array(self.command.target_hand_wrist_pos)[0],
                            reference_quat_xyzw=array(self.command.target_hand_wrist_quat)[0],
                            warm_start_valid=bool(self.action._warm_start_valid[0]),
                            previous_accepted_q=array(self.action.last_joint_target)[0])
        self.in_process = True
        try:
            result = original(actions)
        finally:
            self.in_process = False
        self.context["q_accepted"] = array(self.action.last_joint_target)[0]
        self.context["fallback"] = ("none" if self.context["ik_accepted"] else
                                    "hold_previous_valid" if self.context["warm_start_valid"] else "hold_measured")
        self.emit("command", **self.context)
        return result

    def inverse(self, original, position, quaternion, **kwargs):
        result = original(position, quaternion, **kwargs)
        if self.in_process:
            origin = array(self.env.scene.env_origins)[0]
            self.context.update(ik_input_pos=np.asarray(position).copy()+origin,
                                ik_input_quat_xyzw=np.asarray(quaternion).copy(),
                                q_ik=result.q.copy(), ik_success=bool(result.success),
                                solver_success=bool(result.optimizer_success),
                                ik_finite=bool(result.finite), ik_limit_violation=bool(result.joint_limit_violation),
                                ik_accepted=bool(result.success and result.finite and not result.joint_limit_violation),
                                ik_message=result.message, ik_nfev=result.nfev)
        return result

    def joint_write(self, original, *args, **kwargs):
        before = self.joints(self.robot.data.joint_pos_target)
        result = original(*args, **kwargs)
        after = self.joints(self.robot.data.joint_pos_target)
        self.writes.append(dict(caller=inspect.currentframe().f_back.f_back.f_code.co_name,
                                before=before, after=after))
        return result

    def apply(self, original):
        self.writes = []
        self.position_setter_calls = []
        self.velocity_setter_calls = []
        result = original()
        self.q_cmd = array(self.action.applied_joint_target)[0]
        return result

    def send(self, original):
        self.pre_send = self.joints(self.robot.data.joint_pos_target)
        if self.full_state:
            self.full_commands = {
                "all_position_targets": array(self.robot.data.joint_pos_target)[0],
                "all_velocity_targets": array(self.robot.data.joint_vel_target)[0],
                "all_effort_targets": array(self.robot.data.joint_effort_target)[0],
            }
        result = original()
        self.post_actuator = self.joints(self.robot._joint_pos_target_sim)
        return result

    def backend_position(self, original, data, *args, **kwargs):
        self.position_setter_calls.append(array(data)[0, self.backend_ids])
        return original(data, *args, **kwargs)

    def backend_velocity(self, original, data, *args, **kwargs):
        self.velocity_setter_calls.append(array(data)[0, self.backend_ids])
        return original(data, *args, **kwargs)

    def physics(self, original, *args, **kwargs):
        if self.context is None:
            return original(*args, **kwargs)
        self.before_state = self.state()
        bank = getattr(self.env, "_paired_arm_states", None)
        if bank and self.episode_step == 0:
            self.emit("paired_first_physics", deltas=bank.verify(self.episode, self.before_state, first_physics=True))
        self.pre_physics_target = self.joints(self.robot.data.joint_pos_target)
        self.pending = True
        return original(*args, **kwargs)

    def update(self, original, dt):
        result = original(dt)
        if self.pending:
            self.pending = False
            self.episode_step += 1
            self.emit("physics_sample", episode_physics_step=self.episode_step,
                      state_time_s=self.episode_step*self.env.physics_dt,
                      before_state=self.before_state, **self.state(), **self.context,
                      q_cmd=self.q_cmd, interpolation_step=self.action._interpolation_step,
                      q_pre_send=self.pre_send, q_post_actuator=self.post_actuator,
                      q_pre_physics_buffer=self.pre_physics_target,
                      q_after_physics_buffer=self.joints(self.robot.data.joint_pos_target),
                      backend_position_calls=self.position_setter_calls,
                      backend_velocity_calls=self.velocity_setter_calls,
                      position_buffer_writes=self.writes, **self.full_commands)
        return result

    def reset(self, original, env_ids):
        self.emit("episode_end", **self.state(), termination={
            name: bool(self.env.termination_manager.get_term(name)[0])
            for name in self.env.termination_manager.active_terms})
        result = original(env_ids)
        self.episode += 1
        self.episode_step = 0
        self.context = None
        self.initialization()
        return result

    def command_update(self, original):
        before = int(self.command.time_steps[0])
        result = original()
        self.emit("reference_update", before_frame=before, after_frame=int(self.command.time_steps[0]))
        return result

    def after_env_step(self, dones):
        self.emit("env_step_return", done=bool(dones[0]), **self.state())

    def close(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        self.emit("trace_end")
        self.file.close()
        print(f"[arm execution trace] {self.path}")
