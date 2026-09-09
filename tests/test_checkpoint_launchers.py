"""Exercise shell model selection without importing Isaac or loading a policy."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CheckpointLauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.checkpoint = self.directory / 'default.pt'
        self.checkpoint.touch()
        self.reference = self.directory / 'reference.h5'
        self.reference.touch()
        self.backend = self.directory / 'backend'
        self.backend.write_text(
            '#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n'
            'print(json.dumps(sys.argv[1:]))\n'
            'if "--output" in sys.argv:\n'
            '    p = Path(sys.argv[sys.argv.index("--output") + 1])\n'
            '    p.mkdir(parents=True, exist_ok=True)\n'
            '    (p / "metadata.json").write_text("{}")\n'
        )
        self.backend.chmod(0o755)
        self.env = dict(os.environ, ISAAC_SIM_PYTHON=str(self.backend),
                        REGRIND_FLOATING_CHECKPOINT=str(self.checkpoint),
                        REGRIND_SEQUENCE='20200709_143747_left')

    def run_script(self, script, *args, check=True):
        return subprocess.run(['bash', str(ROOT / 'scripts' / script), *map(str, args)],
                              cwd=ROOT, env=self.env, text=True, capture_output=True,
                              check=check)

    def rl(self, mode, *args, **kwargs):
        return self.run_script('rl.sh', mode, '--reference', self.reference,
                               '--headless', *args, **kwargs)

    def test_play_and_arm_use_shared_default(self):
        for mode in ('play', 'play-arm'):
            result = self.rl(mode)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index('--checkpoint') + 1], str(self.checkpoint))
            if mode == 'play-arm':
                self.assertIn('Approved video controller', result.stderr)
                self.assertEqual(args[args.index('--arm-controller')+1],'video')
                self.assertIn('--transfer-evaluation',args)
                self.assertIn('--expected-reference',args)

    def test_explicit_old_arm_baseline_is_preserved(self):
        args=json.loads(self.rl('play-arm','--arm-controller','baseline').stdout)
        self.assertIn('Regrind-RB3-Revo2-TunaCan-Online-Play-v0',args)
        self.assertNotIn('--transfer-evaluation',args)

    def test_explicit_selection_and_other_tasks_are_not_overridden(self):
        for selection in (('--checkpoint', 'explicit.pt'), ('--checkpoint=explicit.pt',),
                          ('--load_run', 'old_run'), ('agent.load_checkpoint=model_4999.pt',),
                          ('--task', 'OtherTask'), ('--legacy-arm-rl',),
                          ('--sequence', 'other_sequence')):
            with self.subTest(selection=selection):
                args = json.loads(self.rl('play', *selection).stdout)
                self.assertNotIn(str(self.checkpoint), args)

    def test_missing_default_fails_but_explicit_checkpoint_still_works(self):
        self.checkpoint.unlink()
        result = self.rl('play', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('default 10000-update floating checkpoint', result.stderr)
        self.rl('play', '--checkpoint', 'explicit.pt')

    def test_training_is_not_pinned(self):
        for mode in ('train', 'train-arm'):
            args = json.loads(self.rl(mode).stdout)
            self.assertNotIn('--checkpoint', args)

    def test_transfer_and_play_select_same_controller_without_affecting_floating(self):
        for mode in ('train-arm','play-arm'):
            for controller in ('video','baseline'):
                args=json.loads(self.rl(mode,'--arm-controller',controller).stdout)
                if mode=='train-arm' or controller=='video':
                    self.assertEqual(args[args.index('--arm-controller')+1],controller)
        for mode in ('train','play'):
            self.assertNotIn('--arm-controller',json.loads(self.rl(mode).stdout))

    def test_candidate_historical_profile_disables_new_material_default(self):
        for option in ('--match-recording','--transfer-config'):
            args=json.loads(self.run_script('play_arm_candidate.sh',self.directory/'view',option,'profile').stdout)
            self.assertNotIn('--arm-controller',args)

    def test_candidate_override_wins_without_changing_controller(self):
        for extra in ((), ('--checkpoint', 'explicit.pt')):
            args = json.loads(self.run_script('play_arm_candidate.sh', self.directory / 'view', *extra).stdout)
            values = [args[i + 1] for i, value in enumerate(args) if value == '--checkpoint']
            self.assertEqual(values[-1], 'explicit.pt' if extra else str(self.checkpoint))
            self.assertIn('config/experiments/rb3_transfer_recovery_candidate.json', args)

    def test_capture_default_and_explicit_model(self):
        explicit = self.directory / 'explicit.pt'
        explicit.touch()
        for kind in ('arm', 'floating'):
            for override in (False, True):
                out = self.directory / f'{kind}_{override}'
                model = explicit if override else self.checkpoint
                arguments = [out]
                if override:
                    arguments = [out, model] if kind == 'arm' else [model, out]
                self.run_script(f'record_{kind}_comparison.sh', *arguments)
                logs = sorted(out.glob('*.log'))
                self.assertEqual(len(logs), 2)
                for log in logs:
                    args = json.loads(log.read_text())
                    self.assertEqual(args[args.index('--checkpoint') + 1], str(model))

    def test_hq_capture_is_explicit_and_forwards_visual_only_settings(self):
        out = self.directory / 'hq'
        self.run_script('record_floating_comparison.sh', '--presentation-hq', out)
        for log in out.glob('*.log'):
            args = json.loads(log.read_text())
            self.assertIn('--hide-revo2-keypoints', args)
            self.assertIn('env.video_recorder.window_width=3840', args)
            self.assertIn('env.video_recorder.window_height=2160', args)


if __name__ == '__main__':
    unittest.main()
