"""Opt-in paired evaluation: saved placements, existing reset, verified full states.

No policy actions are replayed. Hidden PhysX contact caches are not claimed to
be serializable; this verifies the explicit reset state and first-step state.
"""
import json
from pathlib import Path

import numpy as np


STATE_KEYS = ("all_joint_pos", "all_joint_vel", "robot_root_state", "object_root_state",
              "actual_base_pos", "actual_base_quat_xyzw", "applied_arm_target", "placement_offset")


def check_state(actual, expected, atol=1e-6):
    deltas = {}
    for key in STATE_KEYS:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid paired state {key}")
        deltas[key] = float(np.max(np.abs(a-b)))
        if deltas[key] > atol:
            raise ValueError(f"Paired initial state mismatch: {key} delta={deltas[key]}")
    if actual["reference_frame"] != expected["reference_frame"]:
        raise ValueError("Paired reference frame mismatch")
    return deltas


class PairedArmStates:
    def __init__(self, env, path, episodes):
        self.path = str(Path(path).resolve())
        with Path(path).open() as stream:
            records = [json.loads(line) for line in stream]
        self.meta = records[0]
        if records[-1]["event"] != "trace_end" or not self.meta.get("full_state"):
            raise ValueError("A complete full-state execution trace is required")
        self.initials = {r["episode"]:r for r in records if r["event"] == "initialization"}
        self.first = {}
        for r in records:
            if r["event"] == "physics_sample":
                self.first.setdefault(r["episode"], r["before_state"])
        if episodes <= 0 or len(self.first) < episodes:
            raise ValueError("Requested episodes exceed recorded complete initial states")
        self.command = env.command_manager.get_term("reference")
        if env.event_manager.active_terms or self.command.cfg.rsi_enabled or self.command.cfg.enable_reset_perturbation:
            raise ValueError("Paired evaluation requires the existing deterministic PLAY reset")
        if self.meta["user_joint_names"] != self.command.robot.joint_names:
            raise ValueError("Paired robot joint name/order mismatch")
        if Path(self.meta["reference"]).resolve() != Path(self.command.reference.path).resolve():
            raise ValueError("Paired reference mismatch")
        self.index = 0
        self.original = self.command._sample_placement
        self.command._sample_placement = self.sample
        env._paired_arm_states = self

    def sample(self, env_ids):
        import torch
        # Consume the normal reset draws, then use the saved common placement.
        self.original(env_ids)
        if len(env_ids) != 1 or self.index not in self.initials:
            raise ValueError("Paired placement bank exhausted or multiple environments requested")
        self.command.placement_offset[env_ids] = torch.as_tensor(
            self.initials[self.index]["placement_offset"], device=self.command.device)
        self.index += 1

    def verify(self, episode, state, first_physics=False):
        expected = self.first[episode] if first_physics else self.initials[episode]
        return check_state(state, expected)
