import ast
import json
from pathlib import Path
import tempfile
import unittest
from tools.rb3_revo2_ik.actual_motion_replay import load_actual_motion


class TestActualMotion(unittest.TestCase):
    def fixture(self,path):
        meta=dict(mode='floating',stage='grasp',physics_dt=1/120,
                  ends=[dict(termination=dict(success=True))])
        row=dict(episode=0,time_s=1/120,desired_wrist_pos=[99,99,99],
                 state=dict(wrist_pos=[.4,0,.1],wrist_quat=[0,0,0,1],
                            hand_q=[.1]*6,follower_q=[.1]*5))
        self.save(path,meta,row)
        return meta,row

    def save(self,path,meta,row):
        (path/'metadata.json').write_text(json.dumps(meta))
        (path/'physics.jsonl').write_text(json.dumps(row)+'\n')

    def test_measured_state_not_policy_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            _,seq=load_actual_motion(p,1,1/120)
            self.assertEqual(seq[0][0]['state']['wrist_pos'],[.4,0,.1])

    def test_timestamp_and_success_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m,r=self.fixture(p)
            r['time_s']=2/120;self.save(p,m,r)
            with self.assertRaises(AssertionError):load_actual_motion(p,1,1/120)
            r['time_s']=1/120;m['ends'][0]['termination']['success']=False;self.save(p,m,r)
            with self.assertRaises(ValueError):load_actual_motion(p,1,1/120)

    def test_invalid_quaternion_or_dt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m,r=self.fixture(p)
            with self.assertRaises(ValueError):load_actual_motion(p,1,1/30)
            r['state']['wrist_quat']=[0,0,0,0];self.save(p,m,r)
            with self.assertRaises(ValueError):load_actual_motion(p,1,1/120)

    def test_runtime_has_no_policy_or_state_teleport(self):
        path=Path(__file__).resolve().parents[1]/'tools/rb3_revo2_ik/actual_motion_replay.py'
        tree=ast.parse(path.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_actual_motion')
        attrs=[n.attr for n in ast.walk(run) if isinstance(n,ast.Attribute)]
        self.assertFalse(any('write_joint_state' in a or 'write_root' in a for a in attrs))
        names=[n.id for n in ast.walk(run) if isinstance(n,ast.Name)]
        self.assertNotIn('policy',names)
        self.assertIn('apply_actions',attrs)  # Keep normal six-leader/mimic drive path.


if __name__=='__main__':unittest.main()
