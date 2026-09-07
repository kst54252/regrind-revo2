"""Lightweight navigation/import contracts after the offline-analysis split."""
import importlib
import json
from pathlib import Path
import pkgutil
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryEntrypointTest(unittest.TestCase):
    def test_offline_analysis_modules_import_without_isaac(self):
        package = importlib.import_module('tools.arm_diagnostics')
        modules = list(pkgutil.iter_modules(package.__path__))
        self.assertGreater(len(modules), 0)
        for module in modules:
            with self.subTest(module=module.name):
                importlib.import_module(f'tools.arm_diagnostics.{module.name}')

    def test_analysis_launchers_resolve_module_targets(self):
        for path in sorted((ROOT / 'scripts').glob('*.sh')):
            for line in path.read_text().splitlines():
                if '-m tools.arm_diagnostics.' not in line:
                    continue
                module = line.split('-m ', 1)[1].split()[0]
                with self.subTest(script=path.name):
                    self.assertIsNotNone(importlib.util.find_spec(module))

    def test_candidate_remains_explicit_with_existing_gains(self):
        config = json.loads((ROOT / 'config/experiments/rb3_smooth_bounded_ik.json').read_text())
        gains = json.loads((ROOT / 'config/experiments/rb3_precision_candidates.json').read_text())
        self.assertIn(config['arm_gains_key'], gains)
        self.assertTrue(config['velocity_bounded_ik'])
        self.assertTrue(config['arm_response_physics'])
        self.assertEqual(config['ik_acceleration_limit'], 250.)
        self.assertEqual(config['response_tau'], .1)
        launcher = (ROOT / 'scripts/play_arm_candidate.sh').read_text()
        self.assertIn('config/experiments/rb3_transfer_recovery_candidate.json', launcher)
        self.assertNotIn('rb3_smooth_bounded_ik.json', launcher)


if __name__ == '__main__':
    unittest.main()
