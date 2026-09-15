"""Recorded raw IK knot path, time-stretched only, for KINEMATIC viewing.

Not policy inference, physical grasp evaluation, or a new IK algorithm.
Each original joint knot is preserved. Quintic scalar timing stays on each
joint-linear segment; it does not assert exact Cartesian interpolation.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT))
from tools.arm_diagnostics.compare_singularity_methods import load_episode
from tools.rb3_revo2_ik.reference_trajectory import load_reference_trajectory, analyze_continuity, DEFAULT_REVO2_JOINT_NAMES
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics

OUT=Path(__file__).resolve().parent
FILE=OUT/'raw_ik_slow_reference.npz'
if FILE.exists():raise FileExistsError(FILE)
source=ROOT/'outputs/diagnostics/ik120_improvement_20260907/old20_baseline'
kin,data=load_episode(source,11)
lo,hi=108,132
rows=[]
with (source/'physics.jsonl').open() as stream:
    for line in stream:
        row=json.loads(line)
        if row['episode']==11 and lo<=row['episode_step']<=hi:rows.append(row)
assert [r['episode_step'] for r in rows]==list(range(lo,hi+1))
q=data['q'][lo:hi+1]
hand=np.array([r['hand_target'] for r in rows])
assert tuple(data['meta']['joint_names'][i] for i in data['meta']['hand_ids'])==DEFAULT_REVO2_JOINT_NAMES
object_state=np.array([r['state']['object_state'][:7] for r in rows])
ikpos=data['p'][lo:hi+1]; ikquat=data['quat'][lo:hi+1]
fps=60.;vmax=.65;amax=2.
allj=np.c_[q,hand]
# max s'(u)=1.875, max |s''(u)|=10/sqrt(3) for s=10u³-15u⁴+6u⁵.
displacement=np.max(np.abs(np.diff(allj,axis=0)),axis=1)
duration=np.maximum.reduce([np.ones(len(displacement))/fps,
    1.875*displacement/vmax,np.sqrt((10/np.sqrt(3))*displacement/amax)])
ticks=np.ceil(duration*fps).astype(int)
duration=ticks/fps
knots=[0];dense=[allj[0]];objects=[object_state[0]]
original_targets=[np.r_[ikpos[0],ikquat[0]]];rawindex=[float(lo)]
analytic_v=[np.zeros(12)];analytic_a=[np.zeros(12)]
for k,n in enumerate(ticks):
    dq=allj[k+1]-allj[k];dur=duration[k]
    object_rot=Slerp([0.,1.],Rotation.from_quat(object_state[k:k+2,3:7]))
    target_rot=Slerp([0.,1.],Rotation.from_quat(ikquat[k:k+2]))
    for step in range(1,n+1):
        u=step/n;s=10*u**3-15*u**4+6*u**5
        dense.append(allj[k]+s*dq)
        objects.append(np.r_[object_state[k,:3]+s*(object_state[k+1,:3]-object_state[k,:3]),object_rot(s).as_quat()])
        original_targets.append(np.r_[ikpos[k]+s*(ikpos[k+1]-ikpos[k]),target_rot(s).as_quat()])
        rawindex.append(lo+k+s)
        analytic_v.append(dq*(30*u**2-60*u**3+30*u**4)/dur)
        analytic_a.append(dq*(60*u-180*u**2+120*u**3)/dur**2)
    knots.append(len(dense)-1)
dense=np.array(dense);objects=np.array(objects);original_targets=np.array(original_targets)
np.testing.assert_allclose(dense[np.array(knots)],allj,atol=1e-12,rtol=0)
assert np.abs(analytic_v).max()<=vmax+1e-10
assert np.abs(analytic_a).max()<=amax+1e-10
assert np.all(dense[:,:6]>=kin.joint_lower-1e-9) and np.all(dense[:,:6]<=kin.joint_upper+1e-9)
handkin=Revo2Kinematics();lower,upper=handkin.get_joint_limits()
assert np.all(dense[:,6:]>=lower-1e-7) and np.all(dense[:,6:]<=upper+1e-7)
p,r=kin.forward_batch(dense[:,:6])
pose_error=np.linalg.norm(p-original_targets[:,:3],axis=1)
rot_error=(Rotation.from_quat(original_targets[:,3:]).inv()*Rotation.from_quat(r)).magnitude()
# 1.5 s holds clearly separate loop resets from the selected forward motion.
hold=90
def held(a):return np.concatenate([np.repeat(a[:1],hold,axis=0),a,np.repeat(a[-1:],hold,axis=0)])
np.savez_compressed(FILE,
    rb3_joints=held(dense[:,:6]),revo2_joints=held(dense[:,6:]),
    reference_joints=held(dense),rb3_joint_names=np.array(kin.joint_names),
    revo2_joint_names=np.array(DEFAULT_REVO2_JOINT_NAMES),fps=np.array(fps),
    wrist_pos=held(p),wrist_quat=held(r),quaternion_order=np.array('xyzw'),
    object_pos=held(objects[:,:3]),object_quat=held(objects[:,3:]),
    source_raw_ik_knot_indices=np.array(knots)+hold,
    source_episode_step=held(np.array(rawindex)),
    original_ik_target_pos=held(original_targets[:,:3]),
    original_ik_target_quat=held(original_targets[:,3:]),
    role=np.array('KINEMATIC raw IK knot visualization; wrist_pos is FK of interpolated q, NOT policy target; object is recorded post-step pose'))
traj=load_reference_trajectory(FILE);check=analyze_continuity(traj,.5)
assert not len(check.nonfinite_frames) and not len(check.discontinuity_frames)
assert check.max_abs_velocity_per_joint.max()<=vmax+1e-8
summary={
 'mode':'KINEMATIC_ONLY; no policy, IK solve, actuator tuning or dynamic grasp',
 'source':str(source.relative_to(ROOT)), 'episode':11,
 'source_episode_steps':[lo,hi],
 'source_command_time_s':[(lo-1)*data['dt'],(hi-1)*data['dt']],
 'source_state_time_s':[lo*data['dt'],hi*data['dt']],
 'raw_peak_step_rad':float(np.abs(np.diff(q,axis=0)).max()),
 'raw_peak_path_speed_rad_s':float(np.abs(np.diff(q,axis=0)).max()/data['dt']),
 'motion_seconds':float(duration.sum()),'hold_each_end_seconds':hold/fps,
 'display_samples':traj.frames,'fps':fps,
 'max_analytic_speed_rad_s':float(np.abs(analytic_v).max()),
 'max_analytic_acceleration_rad_s2':float(np.abs(analytic_a).max()),
 'max_sampled_speed_rad_s':float(check.max_abs_velocity_per_joint.max()),
 'max_sampled_acceleration_rad_s2':float(check.max_abs_acceleration_per_joint.max()),
 'all_original_25_joint_knots_preserved':True,
 'joint_interpolation':'piecewise joint-linear path, common quintic scalar timing; no angle wrapping or branch changes',
 'hand':'recorded leader commands, existing replay expands mimic',
 'object':'recorded post-step visual poses, not physics; common interpolation progress',
 'wrist_reference':'FK of dense q for geometric replay validation, not original target tracking',
 'dense_fk_vs_interpolated_original_ik_target_max_m':float(pose_error.max()),
 'dense_fk_vs_interpolated_original_ik_target_max_rad':float(rot_error.max()),
 'input_metadata_sha256':hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),
 'reference_sha256':hashlib.sha256(FILE.read_bytes()).hexdigest(),
}
(OUT/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
