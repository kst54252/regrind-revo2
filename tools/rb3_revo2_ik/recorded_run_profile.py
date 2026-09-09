"""Restore a supported mounted video run's settings, not its recorded actions."""
import hashlib
import json
from pathlib import Path


def match_recording(args, location):
    path = Path(location)
    if path.is_dir(): path = path/'metadata.json'
    record = json.loads(path.read_text())
    if (record.get('mode') != 'simple' or record.get('stage') != 'grasp'
            or not record.get('frozen_policy_verified') or record.get('zero_actions')):
        raise ValueError('Recording profile requires a completed live simple-arm policy run')
    if record.get('fingertip_contact') or args.fingertip_contact_config or args.transfer_evaluation:
        raise ValueError('Do not mix a recorded hard-contact controller with contact/transfer experiments')
    if args.zero_actions or args.new_state_seed is not None or args.recovery_no_can:
        raise ValueError('Recording profile cannot change actions, placements or contact')
    expected = dict(record['input_sha256'])
    expected[record['checkpoint']] = record['checkpoint_sha256']
    for filename, digest in expected.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Recorded input changed: {filename}')
    # Explicit recording selection takes precedence over launcher defaults.
    args.mode = 'simple'; args.stage = 'grasp'
    args.checkpoint = record['checkpoint']; args.states = record['state_bank']
    args.episodes = len(record['ends'])
    args.transfer_config = None  # Restore resolved values, not a mutable preset.
    for key in ('fast_ik','ik_policy_rate','velocity_bounded_ik','ik_acceleration_limit',
                'arm_gains_key','arm_gains_file','arm_velocity_path','arm_response_physics'):
        setattr(args,key,record[key])
    args.response_tau = record['response_tau_s']
    args.realtime_view = record['realtime_view']
    args.recovery_capture = record.get('recovery_runtime') is not None
    return dict(metadata=str(path.resolve()),metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                record=record)


def verify_recorded_runtime(record, current):
    """Reject input/config drift before the policy rollout starts."""
    for key in ('physics_dt','control_dt','gains','limits','joint_names','gravity','initial_native_settings'):
        if record[key] != current[key]:
            raise ValueError(f'Recorded runtime mismatch: {key}')
