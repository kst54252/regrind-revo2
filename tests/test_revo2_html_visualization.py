"""Checks for the self-contained browser FK visualizer."""

import tempfile
import unittest
from pathlib import Path

from tools.revo2_kinematics.visualize_revo2_keypoints_html import build_html


class Revo2HtmlVisualizationTest(unittest.TestCase):
    def test_builds_standalone_html_with_exact_serialized_fk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "viewer.html"
            generated, max_error = build_html(output)
            html = generated.read_text(encoding="utf-8")

            self.assertEqual(generated, output.resolve())
            self.assertLessEqual(max_error, 1.0e-12)
            self.assertIn("Revo2 keypoint &amp; mimic FK", html)
            self.assertIn("right_thumb_metacarpal_joint", html)
            self.assertIn("right_index_distal_joint", html)
            self.assertIn("kp_20_little_tip", html)
            self.assertNotIn("__MODEL_JSON__", html)
            self.assertNotIn("src=\"http", html)


if __name__ == "__main__":
    unittest.main()
