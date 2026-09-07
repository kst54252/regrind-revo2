"""Opt-in bounded warm-start shortcut; retain the existing solver and fallback."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


class PoseJacobian:
    """Same model transforms/residual with analytic derivatives, per-solve cache."""
    def __init__(self, kin):
        self.kin=kin
        self.q=None

    def evaluate(self,q,target_position,target_rotation,weight):
        if self.q is not None and np.array_equal(q,self.q):return self.value,self.derivative
        k=self.kin;R=k.base_rotation.copy();p=k.base_position.copy()
        rotations=Rotation.from_rotvec(k.joint_axes*q[:,None]).as_matrix()
        origins=[];axes=[]
        for offset,axis,rot in zip(k.joint_offsets,k.joint_axes,rotations):
            p=p+R@offset;origins.append(p.copy());axes.append(R@axis);R=R@rot
        p=p+R@k.link6_to_wrist_position;R=R@k.link6_to_wrist_rotation
        phi=Rotation.from_matrix(target_rotation.T@R).as_rotvec()
        x,y,z=phi;hat=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        theta=np.linalg.norm(phi)
        a=1/12+theta**2/720 if theta<1e-4 else (1-.5*theta/np.tan(.5*theta))/theta**2
        left_inverse=np.eye(3)-.5*hat+a*(hat@hat)
        axes=np.asarray(axes)
        self.value=np.r_[weight*(p-target_position),phi]
        self.derivative=np.vstack([weight*np.cross(axes,p-np.asarray(origins)).T,
                                   left_inverse@target_rotation.T@axes.T])
        self.q=q.copy()
        return self.value,self.derivative

    def residual(self,*args):return self.evaluate(*args)[0]
    def jacobian(self,*args):return self.evaluate(*args)[1]


class WarmStartIK:
    def __init__(self, reference, max_step_rad=.15):
        if not np.isfinite(max_step_rad) or max_step_rad<=0:
            raise ValueError('max_step_rad must be positive and finite')
        self.reference = reference
        self.local = copy.copy(reference)
        self.local._candidate_seeds = lambda warm, neutral: [
            np.clip(warm, reference.joint_lower+1e-10, reference.joint_upper-1e-10)]
        self.max_step_rad = max_step_rad
        self.calls = self.fallbacks = 0

    def inverse(self, position, quaternion, *, initial_q, neutral_q, **kwargs):
        self.calls += 1
        differential=PoseJacobian(self.reference)
        self.local._residual=differential.residual
        result = self.local.inverse(position, quaternion, initial_q=initial_q,
                                    neutral_q=neutral_q, jacobian=differential.jacobian, **kwargs)
        if (result.success and result.finite and not result.joint_limit_violation
                and np.max(np.abs(result.q-initial_q)) <= self.max_step_rad):
            return result
        self.fallbacks += 1
        return self.reference.inverse(position, quaternion, initial_q=initial_q,
                                      neutral_q=neutral_q, **kwargs)
