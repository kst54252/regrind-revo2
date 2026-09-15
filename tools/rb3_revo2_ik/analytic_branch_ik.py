"""Opt-in closed-form IK for the verified RB3 ZYY-ZYZ model.

Enumerates shoulder (2), elbow (2), wrist (2), then every 2*pi lift inside
the model's *bounded* joint coordinates. Never used by the default controller.
Exact wrist singularities have continuous solution families, not eight isolated
solutions: report them explicitly instead of claiming finite enumeration is all IK.
FK and mounted-wrist/world transforms remain owned by RB3730Kinematics.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BranchSolutions:
    q: np.ndarray
    branch_ids: np.ndarray
    geometric_count: int
    singular_families: list
    exhaustive_isolated: bool


def periodic_lifts(q, lower, upper):
    """All legal bounded-coordinate representatives; no temporal angle wrapping."""
    q, lower, upper = map(lambda x: np.asarray(x, dtype=float), (q, lower, upper))
    if (q.shape != (6,) or lower.shape != (6,) or upper.shape != (6,)
            or not np.isfinite([q, lower, upper]).all() or np.any(lower >= upper)):
        raise ValueError('Expected finite six-joint coordinates and ordered limits')
    choices = []
    for angle, lo, hi in zip(q, lower, upper):
        first = int(np.ceil((lo-angle-1e-10)/(2*np.pi)))
        last = int(np.floor((hi-angle+1e-10)/(2*np.pi)))
        choices.append([angle+2*np.pi*k for k in range(first, last+1)])
    return np.array(list(product(*choices)), dtype=float).reshape(-1, 6)


class AnalyticBranchIK:
    def __init__(self, kin):
        self.kin = kin
        # Reject changed geometry; do not quietly apply a spherical-wrist
        # formula to a different asset. All lengths come from the existing model.
        if kin.config['joint_axes'] != ['Z', 'Y', 'Y', 'Z', 'Y', 'Z']:
            raise ValueError('Analytic IK requires the verified ZYYZYZ axes')
        o = kin.joint_offsets
        expected = np.zeros((6, 3))
        expected[0, 2] = o[0, 2]
        expected[2, 1:] = o[2, 1:]
        expected[4, 2] = o[4, 2]
        if not np.allclose(o, expected, atol=1e-12, rtol=0) or min(o[2, 2], o[4, 2]) <= 0:
            raise ValueError('Unsupported RB3 offsets / non-spherical wrist')
        self.height, self.side, self.l1, self.l2 = o[0, 2], o[2, 1], o[2, 2], o[4, 2]

    def inverse_all(self, position, quaternion_xyzw):
        k = self.kin
        p, quat = np.asarray(position, float), np.asarray(quaternion_xyzw, float)
        if (p.shape != (3,) or quat.shape != (4,) or not np.isfinite(p).all()
                or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12):
            raise ValueError('Expected finite world position and nonzero XYZW quaternion')
        world_r = Rotation.from_quat(quat).as_matrix()
        # T_world_link6 = T_world_wrist @ inv(T_link6_wrist), using the
        # existing exact mount, including arbitrary fixed rotation/translation.
        link_r = world_r @ k.link6_to_wrist_rotation.T
        center = k.base_rotation.T @ (p-link_r@k.link6_to_wrist_position-k.base_position)
        center[2] -= self.height
        r06 = k.base_rotation.T @ link_r
        x, y, z = center
        rho = np.hypot(x, y)
        empty = lambda families=None: BranchSolutions(
            np.empty((0, 6)), np.empty((0, 3), int), 0, families or [], not bool(families))
        if rho < abs(self.side)-1e-12:
            return empty()
        if rho < 1e-12:
            return empty([dict(kind='shoulder_continuum')])
        theta = np.arctan2(y, x)
        alpha = np.arcsin(np.clip(self.side/rho, -1., 1.))
        solutions, ids, geometries, families = [], [], [], []
        for shoulder_id, q1 in enumerate((theta-alpha, theta-np.pi+alpha)):
            radial = np.cos(q1)*x + np.sin(q1)*y
            cosine = (radial**2+z**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2)
            if abs(cosine) > 1+1e-12:
                continue
            elbow = np.arccos(np.clip(cosine, -1., 1.))
            for elbow_id, q3 in enumerate((elbow, -elbow)):
                q2 = np.arctan2(radial, z)-np.arctan2(
                    self.l2*np.sin(q3), self.l1+self.l2*np.cos(q3))
                r03 = (Rotation.from_rotvec([0, 0, q1]).as_matrix()
                       @ Rotation.from_rotvec([0, q2+q3, 0]).as_matrix())
                w = r03.T @ r06
                sine = np.hypot(w[0, 2], w[1, 2])
                if sine < 1e-10:
                    # q4+q6 (q5=0), or q4-q6 (q5=pi), is fixed. Infinite
                    # pairs cannot truthfully be called a finite 'all branches'.
                    families.append(dict(kind='wrist_continuum', shoulder=shoulder_id,
                                         elbow=elbow_id, q123=[q1, q2, q3]))
                    continue
                q5 = np.arctan2(sine, w[2, 2])
                q4 = np.arctan2(w[1, 2], w[0, 2])
                q6 = np.arctan2(w[2, 1], -w[2, 0])
                for wrist_id, angles in enumerate(((q4, q5, q6), (q4+np.pi, -q5, q6+np.pi))):
                    q = np.array([q1, q2, q3, *angles])
                    canonical = np.arctan2(np.sin(q), np.cos(q))
                    lifts = periodic_lifts(canonical, k.joint_lower, k.joint_upper)
                    if len(lifts) and not any(np.max(np.abs(canonical-old)) < 1e-9 for old in geometries):
                        geometries.append(canonical)
                        solutions.extend(lifts)
                        ids.extend([[shoulder_id, elbow_id, wrist_id]]*len(lifts))
        if not solutions:
            return empty(families)
        q = np.asarray(solutions)
        # Verify against the pre-existing FK, not a second forward model.
        fp, fr = k.forward_batch(q)
        pe = np.linalg.norm(fp-p, axis=1)
        oe = (Rotation.from_quat(quat).inv()*Rotation.from_quat(fr)).magnitude()
        if pe.max() > 1e-8 or oe.max() > 1e-8:
            raise RuntimeError(f'Closed-form / verified FK disagreement: {pe.max()} m, {oe.max()} rad')
        return BranchSolutions(q, np.asarray(ids), len(geometries), families, not families)
