"""Read-only observations of the installed implicit arm actuator/PhysX API.

No drive-only torque is claimed: PD fields are approximations, submitted forces
are feedforward commands, and projected joint force is a reaction projection.
"""
from __future__ import annotations

from importlib import metadata
import inspect
from pathlib import Path

import numpy as np

from tools.rb3_revo2_ik.trace_arm_execution import array
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.analyze_arm_execution import pose_errors


class ArmActuatorProbe:
    def __init__(self, env, source_meta, rows, args):
        self.env = env
        self.command = env.command_manager.get_term("reference")
        self.robot = self.command.robot
        self.view = self.robot.root_view
        self.act = self.robot.actuators["rb3_arm"]
        self.names = source_meta["joint_names"]
        self.ids = [self.robot.joint_names.index(n) for n in self.names]
        self.bids = [self.robot.backend_joint_names.index(n) for n in self.names]
        self.aids = [self.act.joint_names.index(n) for n in self.names]
        self.previous_cmd = np.asarray(rows[0]["previous_accepted_q"])
        self.kin = RB3730Kinematics(model_config=source_meta["model_config"],
            base_position=source_meta["base_position"], base_quaternion_xyzw=source_meta["base_quaternion_xyzw"])
        self.projected_status = "available (projected joint reaction, not drive-only)"
        self.restores = []
        self.calls = {}
        self.runtime = self.parameters()
        versions = {}
        for name in ("isaacsim", "isaaclab", "isaaclab_physx", "isaacsim-kernel"):
            try: versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError: versions[name] = "not installed"
        import isaaclab
        root = next((p for p in Path(isaaclab.__file__).parents if (p/"VERSION").is_file()), None)
        versions["lab_checkout_VERSION"] = (root/"VERSION").read_text().strip() if root else None
        methods = {"implicit_compute": self.act.compute, "approximate_clip": self.act._clip_effort,
                   "write_data_to_sim": self.robot.write_data_to_sim,
                   "submit_feedforward": self.view.set_dof_actuation_forces,
                   "submit_velocity": self.view.set_dof_velocity_targets,
                   "projected_joint_effort": self.view.get_dof_projected_joint_forces}
        locations = {name: dict(file=inspect.getsourcefile(fn), line=inspect.getsourcelines(fn)[1])
                     for name,fn in methods.items()}
        from pxr import UsdGeom
        self.metadata = dict(versions=versions, backend=type(self.robot).__module__, device=str(env.device),
            actuator_class=type(self.act).__module__+"."+type(self.act).__name__,
            implicit=bool(self.act.is_implicit_model), joint_names=self.names, user_ids=self.ids,
            backend_ids=self.bids, actuator_group_ids=self.aids, runtime=self.runtime,
            config={k:getattr(self.act.cfg,k) for k in ("stiffness","damping","effort_limit_sim","velocity_limit_sim","effort_limit","velocity_limit")},
            config_source="regrind/source/regrind/regrind/robots/rb3_revo2.py",
            cli_scales=[args.rb3_stiffness_scale,args.rb3_damping_scale,args.rb3_effort_scale],
            gravity_config=env.cfg.sim.gravity, auto_gravity_from_checkpoint=args.auto_gravity_from_ckpt,
            stage_meters_per_unit=UsdGeom.GetStageMetersPerUnit(env.sim.stage),
            self_collision_config=self.robot.cfg.spawn.articulation_props.enabled_self_collisions,
            link_gravity_disabled=array(self.view.get_disable_gravities())[0],
            body_names=self.robot.body_names, source_locations=locations,
            drive_effort_status="UNAVAILABLE: no verified drive-only measurement in inspected APIs",
            saturation_status="UNKNOWN",
            differentiation="Raw unwrapped-coordinate backward difference, no modulo; first predecessor is recorded previous_accepted_q at reset",
            previous_q_cmd_seed=self.previous_cmd,
            timing="q_pre/PD before physics k; native setters before k; q_post/projected effort after physics k. Timestamp is end of k.",
            provenance={
                "pd_p_pre_Nm":"approximate force-PD P, not solved implicit-drive torque",
                "pd_d_pre_Nm":"approximate force-PD D, not solved implicit-drive torque",
                "computed_effort_approx":"ImplicitActuator.compute approximate P+D+feedforward at pre-step state",
                "applied_effort_approx":"ActuatorBase._clip_effort clipped approximation; NOT the implicit drive effort submitted to PhysX",
                "submitted_effort_Nm":"explicitly submitted actuation/feedforward effort, intercepted native setter; positive along positive joint coordinate",
                "projected_joint_effort_Nm":"post-step joint-reaction projection along DOF motion; NOT verified drive-only; never compared to drive maxForce",
            })
        for key,name in (("velocity", "set_dof_velocity_targets"),("effort","set_dof_actuation_forces")):
            fn = getattr(self.view,name)
            def wrapper(data,*a,_key=key,_fn=fn,**kw):
                self.calls[_key].append(array(data)[0,self.bids])
                return _fn(data,*a,**kw)
            setattr(self.view,name,wrapper)
            self.restores.append((name,fn))

    def parameters(self):
        methods = {"drive_types":"get_drive_types", "dof_types":"get_dof_types",
                   "stiffness":"get_dof_stiffnesses", "damping":"get_dof_dampings",
                   "effort_limits":"get_dof_max_forces", "velocity_limits":"get_dof_max_velocities",
                   "position_limits":"get_dof_limits", "drive_model":"get_dof_drive_model_properties"}
        return {key:array(getattr(self.view,method)())[0,self.bids] for key,method in methods.items()}

    def before_send(self, row):
        self.calls = {"velocity":[],"effort":[]}
        self.q_pre = array(self.robot.data.joint_pos)[0,self.ids]
        self.v_pre = array(self.robot.data.joint_vel)[0,self.ids]
        self.q_cmd = array(self.robot.data.joint_pos_target)[0,self.ids]
        self.v_path = (self.q_cmd-self.previous_cmd)/self.env.physics_dt
        self.command_step = self.q_cmd-self.previous_cmd
        self.previous_cmd = self.q_cmd.copy()
        self.limits = self.parameters()

    def after_send(self):
        if any(len(v)!=1 for v in self.calls.values()):
            raise RuntimeError("Expected exactly one native velocity and effort setter per step")
        self.approx = array(self.act.computed_effort)[0,self.aids]
        self.clipped_approx = array(self.act.applied_effort)[0,self.aids]
        self.ff = self.calls["effort"][0]
        self.vtarget = self.calls["velocity"][0]
        self.p = self.limits["stiffness"]*(self.q_cmd-self.q_pre)
        self.d = self.limits["damping"]*(self.vtarget-self.v_pre)

    def after_step(self, row):
        projected = None
        if self.projected_status.startswith("available"):
            try:
                projected = array(self.view.get_dof_projected_joint_forces())[0,self.bids]
            except Exception as error:
                self.projected_status = f"unavailable: {type(error).__name__}: {error}"
        actual = array(self.command.current_hand_wrist_pos)[0]+array(self.env.scene.env_origins)[0]
        quat = array(self.command.current_hand_wrist_quat)[0]
        fkpos,fkquat = self.kin.forward(self.q_cmd)
        pe,re = pose_errors(fkpos[None],fkquat[None],actual[None],quat[None])
        force_pd = np.all(self.limits["drive_types"]==1) and np.all(self.limits["dof_types"]==0)
        return dict(q_cmd=self.q_cmd, q_pre=self.q_pre, velocity_pre=self.v_pre,
            q_actual=array(self.robot.data.joint_pos)[0,self.ids],
            velocity_actual=array(self.robot.data.joint_vel)[0,self.ids],
            command_time_s=row["state_time_s"]-self.env.physics_dt, state_time_s=row["state_time_s"],
            physics_step=self.env._sim_step_counter, v_path=self.v_path, command_step=self.command_step,
            submitted_velocity_rad_s=self.vtarget, submitted_effort_Nm=self.ff,
            pd_p_pre_Nm=self.p if force_pd else None, pd_d_pre_Nm=self.d if force_pd else None,
            computed_effort_approx=self.approx, applied_effort_approx=self.clipped_approx,
            force_pd_units_verified=bool(force_pd), limits=self.limits,
            projected_joint_effort_Nm=projected, projected_status=self.projected_status,
            drive_effort_Nm=None, drive_saturation="UNKNOWN", C_position_m=float(pe[0]), C_rotation_rad=float(re[0]))

    def close(self):
        for name,fn in reversed(self.restores):
            setattr(self.view,name,fn)
