"""Motion I/O contracts. Official read-only SDK probes live in sdk_readers.py."""
from typing import Protocol

import numpy as np

from .contracts import JointCommand, Pose, RobotState, reorder


class RobotBackend(Protocol):
    kind: str

    def read_state(self) -> RobotState:
        """Fresh actual encoders/wrist/object; bounded read, never desired state."""
        ...

    def send(self, command: JointCommand) -> None:
        """Submit BOTH arm and six hand leaders, or raise on any partial failure.

        Future drivers must enforce sequence/deadline, rad/sign/zero conversion,
        per-device acknowledgement and an independent command-loss watchdog.
        """
        ...

    def stop(self, reason: str) -> None:
        """Request a documented controlled stop of both devices; may raise.

        Not an emergency-stop or a guarantee that a physical arm has stopped.
        """
        ...


class MockBackend:
    """Ideal joint follower for CONTRACT tests only; no dynamics/grasp evidence."""
    kind = "mock"

    def __init__(self, kin, joint_names, initial_q, clock, frame="mock_world"):
        self.kin, self.joint_names = kin, tuple(joint_names)
        self.q = np.array(initial_q, dtype=float, copy=True)
        self.dq = np.zeros(12)
        self.clock, self.frame = clock, frame
        self.sequence = 0
        self.pending = None
        self.sent = []
        self.stop_reasons = []

    def read_state(self):
        position, quaternion = self.kin.forward(self.q[:6])
        return RobotState(self.sequence, self.clock(), self.joint_names, self.q, self.dq,
                          Pose(position, quaternion, self.frame), "mock_fk")

    def send(self, command):
        if self.clock() >= command.valid_until:
            raise ValueError("Mock driver rejected expired command")
        if self.sent and command.sequence <= self.sent[-1].sequence:
            raise ValueError("Mock driver rejected out-of-order command")
        self.pending = reorder(command.position, command.joint_names, self.joint_names)
        self.sent.append(command)

    def advance(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid mock timestep")
        if self.pending is not None:
            self.dq = (self.pending - self.q) / dt
            self.q = self.pending.copy()
            self.pending = None
        else:
            self.dq.fill(0)
        self.sequence += 1

    def stop(self, reason):
        self.pending = None
        self.stop_reasons.append(reason)


class SimulationCallbacks:
    """Opt-in binding to an existing simulator's actual reader / command writer.

    Caller owns physics stepping and sole ownership of actuator writes. No
    implicit state teleport, mimic rewrite, effort or gain override happens here.
    This is an injection point, not an already-validated Isaac adapter.
    """
    kind = "simulation"

    def __init__(self, read_state, send, stop):
        self.read_state, self.send, self.stop = read_state, send, stop
