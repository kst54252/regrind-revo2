from __future__ import annotations

import unittest

import numpy as np

from regrind.workcell import WORKCELL_LAYOUT


class WorkcellLayoutTest(unittest.TestCase):
    def test_dimensions_and_heights(self):
        layout = WORKCELL_LAYOUT
        np.testing.assert_allclose(layout["robot_base"]["size"], (0.5, 0.5, 0.7))
        np.testing.assert_allclose(layout["table"]["size_xy"], (0.8, 1.6))
        self.assertAlmostEqual(layout["table"]["top_z"] - layout["floor_z"], 0.72)
        self.assertAlmostEqual(
            layout["robot_base"]["top_z"] - layout["floor_z"], 0.70
        )

    def test_robot_center_and_table_edge_touch(self):
        layout = WORKCELL_LAYOUT
        np.testing.assert_allclose(layout["robot_mount"]["position"][:2], (0.0, 0.0))
        base = layout["robot_base"]
        table = layout["table"]
        base_edge = base["center"][0] + base["size"][0] / 2.0
        table_edge = table["center_xy"][0] - table["size_xy"][0] / 2.0
        self.assertAlmostEqual(base_edge, table_edge)


if __name__ == "__main__":
    unittest.main()
