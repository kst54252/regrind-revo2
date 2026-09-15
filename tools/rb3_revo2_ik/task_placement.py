"""Opt-in rigid task placement for single-environment frozen-policy evaluation.

XYZW quaternions; yaw around the original can, then the existing translation.
This changes neither policy action coordinates nor normalization/history owners.
Only the diagnostic evaluator installs it; ordinary train/play are unchanged.
"""
import math
import random
import torch


class ShuffledPlacementCycle:
    """Stateless episode lookup: each cycle visits every saved placement once.

    Repeated initialization/probing of the same episode never consumes RNG
    state. This changes only the reset schedule, never policy/control timing.
    """
    def __init__(self, bank, count, seed):
        if count < 1 or any(i not in bank for i in range(count)):
            raise ValueError('Need consecutive saved placement indices')
        self.bank, self.count, self.seed = bank, count, seed

    def __getitem__(self, episode):
        if episode < 0:
            raise IndexError('Episode must be nonnegative')
        cycle, offset = divmod(episode, self.count)
        order = list(range(self.count))
        random.Random(self.seed + cycle).shuffle(order)
        return self.bank[order[offset]]


class TaskPlacement:
    def __init__(self, center, yaw_deg):
        if not math.isfinite(yaw_deg):
            raise ValueError('Finite yaw required')
        self.center = center.clone()
        a = math.radians(yaw_deg)
        self.rotation = center.new_tensor([[math.cos(a), -math.sin(a), 0],
                                           [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        self.yaw_deg = float(yaw_deg)

    def world_vector(self, v):
        return v @ self.rotation.T

    def canonical_vector(self, v):
        return v @ self.rotation

    def canonical_position(self, p, offset):
        while offset.ndim < p.ndim:
            offset = offset.unsqueeze(-2)
        return self.center + self.canonical_vector(p - offset - self.center)

    def world_position(self, p, offset):
        while offset.ndim < p.ndim:
            offset = offset.unsqueeze(-2)
        return self.center + self.world_vector(p - self.center) + offset

    def _quat(self, q, sign):
        # Left multiplication by the world-Z yaw quaternion, XYZW.
        s = sign * math.sin(math.radians(self.yaw_deg) / 2)
        c = math.cos(math.radians(self.yaw_deg) / 2)
        x, y, z, w = q.unbind(-1)
        return torch.stack((c*x-s*y, c*y+s*x, c*z+s*w, c*w-s*z), -1)

    def canonical_quat(self, q):
        return self._quat(q, -1)

    def world_quat(self, q):
        return self._quat(q, 1)


def place_reference(command, yaw_deg, arm_q=None):
    """Reset-only hook. Always derive from the original, never the last episode.

    Geometry/reward reference is transformed; hand angles and phase are not.
    An explicit arm branch seeds the existing reset IK, not tracking teleports.
    """
    if command.num_envs != 1:
        raise ValueError('Task placement diagnostic currently requires num_envs=1')
    names = ['reference_'+body+'_'+field for body in ('object', 'wrist')
             for field in ('pos', 'quat', 'lin_vel', 'ang_vel')]
    if not hasattr(command, '_unplaced_reference'):
        command._unplaced_reference = {n: getattr(command, n).clone() for n in
                                      names + ['reference_joint_pos']}
    original = command._unplaced_reference
    transform = TaskPlacement(original['reference_object_pos'][0], yaw_deg)
    command.task_placement = transform
    zero = torch.zeros_like(transform.center)
    for name in names:
        value = original[name]
        if name.endswith('_pos'):
            placed = transform.world_position(value, zero)
        elif name.endswith('_quat'):
            placed = transform.world_quat(value)
        else:
            placed = transform.world_vector(value)
        getattr(command, name).copy_(placed)
    command.reference_joint_pos.copy_(original['reference_joint_pos'])
    if arm_q is not None and command.cfg.joint_reference == 'combined':
        from tools.rb3_revo2_ik.reference_trajectory import DEFAULT_RB3_JOINT_NAMES
        indices = [command.reference.joint_names.index(n) for n in DEFAULT_RB3_JOINT_NAMES]
        command.reference_joint_pos[0, indices] = torch.as_tensor(
            arm_q, device=command.device, dtype=command.reference_joint_pos.dtype)
