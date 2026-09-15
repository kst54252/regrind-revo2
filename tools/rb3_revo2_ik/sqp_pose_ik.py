"""Diagnostic SLSQP pose IK; bounded q/v/a, explicit nonlinear pose constraints.

No controller/default changes. A failed iterate is returned for inspection,
never labelled an accepted command. Uses the existing FK pose Jacobian.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from tools.rb3_revo2_ik.warm_start_ik import PoseJacobian


def solve_sqp(kin, position, quaternion, previous_q, previous_velocity, dt, speed,
              *, acceleration=250., position_tolerance=1e-4, orientation_tolerance=1e-3,
              axis_margin_deg=0., maxiter=150):
    prev, pv, speed = map(lambda x: np.asarray(x, float), (previous_q, previous_velocity, speed))
    p, quat = np.asarray(position, float), np.asarray(quaternion, float)
    scalars = np.array([dt, acceleration, position_tolerance, orientation_tolerance, axis_margin_deg])
    if (any(x.shape != (6,) or not np.isfinite(x).all() for x in (prev, pv, speed))
            or not np.isfinite(scalars).all() or np.any(scalars[:4] <= 0)
            or np.any(speed <= 0) or not 0 <= axis_margin_deg < 90 or maxiter < 1
            or p.shape != (3,) or quat.shape != (4,) or not np.isfinite(p).all()
            or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12
            or np.any(prev < kin.joint_lower-1e-9) or np.any(prev > kin.joint_upper+1e-9)):
        raise ValueError('Invalid SQP input, timestep, limits or tolerance')
    lower = np.maximum.reduce([kin.joint_lower, prev-speed*dt, prev+(pv-acceleration*dt)*dt])
    upper = np.minimum.reduce([kin.joint_upper, prev+speed*dt, prev+(pv+acceleration*dt)*dt])
    if np.any(lower >= upper):
        raise ValueError('Empty SQP q/v/a bounds')
    differential = PoseJacobian(kin)  # per-target cache: never reuse with another target
    rotation = Rotation.from_quat(quat).as_matrix()
    # Slight interior target avoids labelling roundoff above the stated test
    # tolerance a pass. The reported metric always uses the stated tolerance.
    scale = np.r_[np.full(3, .999*position_tolerance), np.full(3, .999*orientation_tolerance)]

    def residual(q):
        r, j = differential.evaluate(q, p, rotation, 1.)
        return r/scale, j/scale[:, None]

    def constraint(q):
        r, _ = residual(q)
        c = [1-r[:3]@r[:3], 1-r[3:]@r[3:]]
        if axis_margin_deg:
            c.append(np.sin(q[4])**2-np.sin(np.deg2rad(axis_margin_deg))**2)
        return np.array(c)

    def constraint_jac(q):
        r, j = residual(q)
        rows = [-2*r[:3]@j[:3], -2*r[3:]@j[3:]]
        if axis_margin_deg:
            row = np.zeros(6)
            row[4] = 2*np.sin(q[4])*np.cos(q[4])
            rows.append(row)
        return np.array(rows)

    # Small increments + small changes in increments; dimensionless by speed*dt.
    step_scale = speed*dt
    def objective(q):
        step = (q-prev)/step_scale
        change = (q-prev-pv*dt)/step_scale
        return .5*(step@step+change@change)

    def gradient(q):
        return (2*(q-prev)-pv*dt)/step_scale**2

    start = time.perf_counter()
    result = minimize(objective, np.clip(prev+pv*dt, lower, upper), jac=gradient,
                      method='SLSQP', bounds=list(zip(lower, upper)),
                      constraints=[dict(type='ineq', fun=constraint, jac=constraint_jac)],
                      options=dict(maxiter=maxiter, ftol=1e-9))
    elapsed = time.perf_counter()-start
    finite = bool(np.isfinite(result.x).all())
    pe, oe, _, _ = kin.pose_error(result.x, p, quat) if finite else (np.inf, np.inf, None, None)
    bounded = finite and bool(np.all(result.x >= lower-1e-8) and np.all(result.x <= upper+1e-8))
    margin_ok = finite and np.abs(np.sin(result.x[4])) >= np.sin(np.deg2rad(axis_margin_deg))-1e-8
    feasible = bool(bounded and pe <= position_tolerance and oe <= orientation_tolerance and margin_ok)
    info = dict(optimizer_success=bool(result.success), feasible=feasible,
                accepted=bool(result.success and feasible), message=str(result.message),
                status=int(result.status), iterations=int(result.nit), solve_s=elapsed,
                position_error_m=float(pe), orientation_error_rad=float(oe),
                box_feasible=bounded, axis_margin_met=bool(margin_ok))
    return result.x, info
