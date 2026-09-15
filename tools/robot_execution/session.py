"""A bounded, opt-in execution seam downstream of the existing action decoder.

No scheduler, policy history, actuator tuning or hardware-enabling switch.
The caller owns 30 Hz policy/phase and control tick scheduling. Missed deadlines
fault instead of stretching phase or sending a burst of delayed commands.
"""
from dataclasses import dataclass
import time

import numpy as np

from .contracts import JointCommand, joint_order, reorder, vector
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics


class ExecutionFault(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionLimits:
    lower: np.ndarray
    upper: np.ndarray
    max_speed: np.ndarray
    max_acceleration: np.ndarray
    max_tracking_error: np.ndarray
    period: float
    max_state_age: float

    def __post_init__(self):
        for name in ("lower", "upper", "max_speed", "max_acceleration", "max_tracking_error"):
            object.__setattr__(self, name, vector(getattr(self, name), 12, name))
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid joint position bounds")
        for name in ("max_speed", "max_acceleration", "max_tracking_error"):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f"{name} must be positive")
        for name in ("period", "max_state_age"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")


class ExecutionSession:
    def __init__(self, backend, kinematics, joint_names, limits, *, frame,
                 solver=None, clock=time.monotonic, require_object=True):
        self.backend, self.kin, self.limits = backend, kinematics, limits
        self.names = joint_order(joint_names)
        if self.names[:6] != tuple(kinematics.joint_names):
            raise ValueError("First six names must be the model's RB3 order")
        if self.names[6:] != tuple(Revo2Kinematics.joint_names):
            raise ValueError("Last six names must be the existing Revo2 leader order")
        if kinematics.mounted_wrist_frame.rsplit("/", 1)[-1] != "right_hand_base_link":
            raise ValueError("IK must target the mounted Revo2 base, not flange/link6")
        lower, upper = kinematics.get_joint_limits()
        if np.any(limits.lower[:6] < lower) or np.any(limits.upper[:6] > upper):
            raise ValueError("Configured arm bounds exceed the existing model bounds")
        self.solver = solver or kinematics
        self.clock, self.frame, self.require_object = clock, frame, require_object
        self.status = "idle"
        self.command_sequence = 0
        self.last_state = None
        self.last_command = None
        self.last_path_speed = np.zeros(12)
        self.last_send_time = None
        self.last_result = None
        self.last_target = None
        self.stop_error = None

    def _fault(self, message):
        self.status = "faulted"
        try:
            self.backend.stop(message)
        except Exception as exc:
            # Do not report a device as stopped when the stop request failed.
            self.stop_error = repr(exc)
            message += f"; STOP REQUEST FAILED: {exc}"
        raise ExecutionFault(message)

    def _fresh(self, sample_time, now, label):
        if not 0 <= now - sample_time <= self.limits.max_state_age:
            raise ValueError(f"{label}: stale or future sample")

    def _read(self):
        state = self.backend.read_state()
        now = self.clock()
        self._fresh(state.sample_time, now, "robot")
        if not state.ready or state.protective_stop:
            raise ValueError("Robot not ready or protective stop active")
        if state.wrist.frame != self.frame:
            raise ValueError("Measured wrist world frame mismatch")
        if self.backend.kind != "mock" and state.wrist_source == "mock_fk":
            raise ValueError("Mock state cannot represent an actual robot")
        if self.require_object and state.object_pose is None:
            raise ValueError("Fresh measured object pose required; reference pose is not a substitute")
        if state.object_pose is not None:
            self._fresh(state.object_sample_time, now, "object")
        if self.last_state is not None:
            if state.sequence <= self.last_state.sequence or state.sample_time <= self.last_state.sample_time:
                raise ValueError("Repeated/out-of-order measured state")
        q = reorder(state.position, state.joint_names, self.names)
        dq = reorder(state.velocity, state.joint_names, self.names)
        if np.any(q < self.limits.lower) or np.any(q > self.limits.upper):
            raise ValueError("Actual joint limit violation")
        if np.any(abs(dq) > self.limits.max_speed + 1e-8):
            raise ValueError("Actual joint speed above configured bound")
        self.last_state = state
        return state, q

    def start(self):
        if self.status != "idle":
            raise ExecutionFault("Create a new session after stop/fault; no automatic restart")
        # No live-device opt-in until SDK, calibration, watchdog and stop are verified.
        if self.backend.kind not in {"mock", "simulation"}:
            raise ExecutionFault("Hardware execution is not commissioned; no device I/O performed")
        try:
            _, q = self._read()
            self.last_command = q.copy()  # actual encoders, never reset/teleport to a reference
            self.last_send_time = self.clock()
            self.status = "running"
        except Exception as exc:
            self._fault(str(exc))

    def step(self, target_factory):
        """Read actual state -> caller's decoder -> existing IK -> checked submit.

        target_factory(state) must return a DecodedTarget. It can reuse a held
        decoded target between policy ticks; it must not update phase at IK rate.
        Only accepted sends advance command history. Failure latches until close.
        """
        if self.status != "running":
            raise ExecutionFault("Session is not running")
        try:
            started = self.clock()
            if not np.isfinite(started):
                raise ValueError("Invalid monotonic clock")
            elapsed = started - self.last_send_time
            if not .9 * self.limits.period <= elapsed <= 1.1 * self.limits.period:
                raise ValueError("Control cadence missed or non-monotonic clock")
            state, actual = self._read()
            target = target_factory(state)
            self.last_target = target
            if target.wrist.frame != self.frame:
                raise ValueError("Target and actual wrist frames differ")
            if set(target.hand_names) != set(self.names[6:]):
                raise ValueError("Target hand joint names are not the six configured leaders")
            if self.clock() >= target.valid_until:
                raise ValueError("Expired decoded target")
            result = self.solver.inverse(target.wrist.position, target.wrist.quaternion_xyzw,
                                         initial_q=self.last_command[:6], neutral_q=actual[:6], max_nfev=300)
            self.last_result = result
            if not (result.success and result.finite and not result.joint_limit_violation):
                raise ValueError(f"IK rejected: {result.message}")
            hand = target.hand_position[[target.hand_names.index(n) for n in self.names[6:]]]
            command_q = vector(np.r_[result.q, hand], 12, "IK/hand command")
            if np.any(command_q < self.limits.lower) or np.any(command_q > self.limits.upper):
                raise ValueError("Command joint limit violation")
            if np.any(abs(command_q - actual) > self.limits.max_tracking_error):
                raise ValueError("Command/actual tracking gap exceeds bound")
            # Raw coordinates, no angle wrapping or silent target clipping.
            path_speed = (command_q - self.last_command) / self.limits.period
            acceleration = (path_speed - self.last_path_speed) / self.limits.period
            if np.any(abs(path_speed) > self.limits.max_speed + 1e-8):
                raise ValueError("Command step/path speed exceeds bound")
            if np.any(abs(acceleration) > self.limits.max_acceleration + 1e-8):
                raise ValueError("Command path acceleration exceeds bound")
            now = self.clock()
            deadline = min(target.valid_until, started + self.limits.period)
            self._fresh(state.sample_time, now, "robot before send")
            if state.object_pose is not None:
                self._fresh(state.object_sample_time, now, "object before send")
            command = JointCommand(self.command_sequence, state.sequence, now, deadline,
                                   self.names, command_q, np.zeros(12), target.reference_frame)
            self.backend.send(command)
            if self.clock() >= deadline:
                raise ValueError("Command send/acknowledgement missed deadline")
            self.last_command = command_q.copy()
            self.last_path_speed = path_speed
            self.last_send_time = started
            self.command_sequence += 1
            return command
        except Exception as exc:
            self._fault(str(exc))

    def close(self):
        if self.status == "running":
            try:
                self.backend.stop("session closed")
            except Exception as exc:
                self.stop_error = repr(exc)
                self.status = "faulted"
                raise ExecutionFault(f"STOP REQUEST FAILED: {exc}") from exc
            self.status = "closed"
