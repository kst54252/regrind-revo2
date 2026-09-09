"""Shared approved video-controller configuration for arm play and transfer.

Reuse the existing single-environment controller; no new IK/control algorithm.
"""
import json
from regrind.assets import REGRIND_PROJECT_ROOT

VIDEO_PRESET = REGRIND_PROJECT_ROOT/'config/experiments/rb3_transfer_recovery_candidate.json'
GAIN_FILE = REGRIND_PROJECT_ROOT/'config/experiments/rb3_precision_candidates.json'
CONTACT_FILE = REGRIND_PROJECT_ROOT/'config/experiments/revo2_rubber_contact.json'


def video_settings():
    return json.loads(VIDEO_PRESET.read_text())


def configure_video_execution(cfg):
    import sys
    # Direct RSL-RL scripts do not start at the repository's package root,
    # unlike the evaluator. Resolve the existing tools.* IK package equally.
    if str(REGRIND_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0,str(REGRIND_PROJECT_ROOT))
    from regrind.data.rb3_revo2_reference import RB3_JOINT_NAMES
    from regrind.tasks.manager_based.dexterous.mdp.simple_mounted_interface import SimpleMountedWristCfg
    if cfg.scene.num_envs != 1:
        raise ValueError('The approved video controller requires num_envs=1; parallel controller changes are not implicit')
    settings=video_settings();old=cfg.actions.root_pose
    gains=json.loads(GAIN_FILE.read_text())[settings['arm_gains_key']]
    cfg.scene.robot.actuators['rb3_arm'].stiffness=dict(zip(RB3_JOINT_NAMES,gains['kp']))
    cfg.scene.robot.actuators['rb3_arm'].damping=dict(zip(RB3_JOINT_NAMES,gains['kd']))
    cfg.actions.root_pose=SimpleMountedWristCfg(asset_name='robot',scale_pos=old.scale_pos,
        scale_rot=old.scale_rot,raw_clip=old.raw_clip,response_tau=settings['response_tau'],
        velocity_path=settings['arm_velocity_path'],response_at_physics=settings['arm_response_physics'],
        fast_ik=True,ik_policy_rate=False,velocity_bounded_ik=False,ik_acceleration_limit=0.)


def select_video_arguments(args):
    """Expose resolved fields to the existing evaluator's telemetry as well."""
    settings=video_settings()
    args.mode='simple';args.transfer_config=None;args.fast_ik=True
    args.ik_policy_rate=False;args.velocity_bounded_ik=False;args.ik_acceleration_limit=0.
    args.arm_gains_file=str(GAIN_FILE)
    for key in ('arm_gains_key','arm_velocity_path','arm_response_physics','response_tau'):
        setattr(args,key,settings[key])
    if not args.fingertip_contact_config: args.fingertip_contact_config=str(CONTACT_FILE)
