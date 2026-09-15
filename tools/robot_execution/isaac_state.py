"""Read the maintained mounted command's runtime state without importing Kit.

This is opt-in and read-only: it is not installed into normal train/play.
Call only AFTER physics and the existing scene/runtime-buffer update. The caller
supplies that acquisition's sequence/time (host monotonic seconds); never stamp
stale PhysX buffers as newly measured just because a consumer requested them.
"""
import numpy as np

from .contracts import Pose, RobotState, joint_order


def _array(value):
    if hasattr(value, 'torch'):
        value = value.torch
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=float, copy=True)


class IsaacMountedStateReader:
    def __init__(self, env, reference_command, joint_names, *, env_index=0, frame='isaac_world'):
        self.env, self.reference = env, reference_command
        self.robot = reference_command.robot
        self.names = joint_order(joint_names)
        if len(set(self.robot.joint_names)) != len(self.robot.joint_names):
            raise ValueError('Duplicate articulation joint names')
        missing = set(self.names) - set(self.robot.joint_names)
        if missing:
            raise ValueError(f'Missing articulation joints: {sorted(missing)}')
        self.ids = [self.robot.joint_names.index(name) for name in self.names]
        if not 0 <= env_index < len(_array(env.scene.env_origins)):
            raise ValueError('Invalid environment index')
        self.index, self.frame = env_index, frame

    def sample(self, sequence, sample_time):
        i, command = self.index, self.reference
        origin = _array(self.env.scene.env_origins)[i]
        # Same data/coordinate path as trace_arm_execution.ArmExecutionTrace.state:
        # current_* are PhysX runtime properties, not USD-cache or IK target poses.
        wrist = Pose(_array(command.current_hand_wrist_pos)[i] + origin,
                     _array(command.current_hand_wrist_quat)[i], self.frame)
        obj = Pose(_array(command.current_object_pos)[i] + origin,
                   _array(command.current_object_quat)[i], self.frame)
        return RobotState(sequence, sample_time, self.names,
            _array(self.robot.data.joint_pos)[i, self.ids],
            _array(self.robot.data.joint_vel)[i, self.ids], wrist, 'physx_body_pose', obj, sample_time)
