"""Recorded-profile restoration must be explicit and reject confounded inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tools.rb3_revo2_ik.recorded_run_profile import match_recording, verify_recorded_runtime


class RecordedProfileTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.model=self.root/'model.pt';self.model.write_bytes(b'preserved model')
        self.states=self.root/'states.jsonl';self.states.write_text('saved states')
        self.meta=dict(mode='simple',stage='grasp',frozen_policy_verified=True,zero_actions=False,
            checkpoint=str(self.model),checkpoint_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest(),
            input_sha256={str(self.states):hashlib.sha256(self.states.read_bytes()).hexdigest()},
            state_bank=str(self.states),ends=[{}],realtime_view=False,recovery_runtime={},
            fast_ik=True,ik_policy_rate=False,velocity_bounded_ik=False,ik_acceleration_limit=0.,
            arm_gains_key='c3',arm_gains_file='recorded.json',arm_velocity_path=True,
            arm_response_physics=True,response_tau_s=.1)
        self.path=self.root/'metadata.json';self.path.write_text(json.dumps(self.meta))

    def args(self):
        return SimpleNamespace(fingertip_contact_config=None,transfer_evaluation=False,
            zero_actions=False,new_state_seed=None,recovery_no_can=False,checkpoint='different-latest.pt',
            transfer_config='possibly-changed-preset.json',episodes=20)

    def test_explicit_snapshot_overrides_defaults_without_loading_actions(self):
        args=self.args();info=match_recording(args,self.root)
        self.assertEqual(args.checkpoint,str(self.model));self.assertEqual(args.episodes,1)
        self.assertEqual(args.response_tau,.1);self.assertTrue(args.arm_velocity_path)
        self.assertIsNone(args.transfer_config);self.assertTrue(args.recovery_capture)
        self.assertEqual(info['record'],self.meta)

    def test_changed_checkpoint_or_state_bank_fails_before_simulation(self):
        for path in (self.model,self.states):
            content=path.read_bytes();path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Recorded input changed'):match_recording(self.args(),self.root)
            path.write_bytes(content)

    def test_contact_transfer_and_action_overrides_rejected(self):
        for key,value in [('fingertip_contact_config','rubber.json'),('transfer_evaluation',True),
                          ('zero_actions',True),('new_state_seed',42),('recovery_no_can',True)]:
            args=self.args();setattr(args,key,value)
            with self.assertRaises(ValueError):match_recording(args,self.root)

    def test_runtime_must_match_recording(self):
        keys=('physics_dt','control_dt','gains','limits','joint_names','gravity','initial_native_settings')
        record=dict.fromkeys(keys,'same');verify_recorded_runtime(record,deepcopy(record))
        for key in keys:
            current=deepcopy(record);current[key]='changed'
            with self.assertRaisesRegex(ValueError,key):verify_recorded_runtime(record,current)
