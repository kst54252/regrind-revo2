"""Shared geometry for the physical RB3-730 table workcell.

The task coordinate frame stays attached to the table surface (world Z=0),
which preserves the DexYCB/REGRIND object coordinates.  The common physical
floor is therefore at Z=-0.72 m and the 0.70 m robot pedestal places the RB3
mounting plane 20 mm below the table surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from regrind.assets import REGRIND_PROJECT_ROOT


WORKCELL_CONFIG_PATH = (
    REGRIND_PROJECT_ROOT / "config" / "workcell" / "rb3_revo2_table.json"
)


def load_workcell_layout(path: str | Path = WORKCELL_CONFIG_PATH) -> dict:
    """Load and validate the metric workcell description."""

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as config_file:
        layout = json.load(config_file)

    base_size = np.asarray(layout["robot_base"]["size"], dtype=float)
    base_center = np.asarray(layout["robot_base"]["center"], dtype=float)
    mount_position = np.asarray(layout["robot_mount"]["position"], dtype=float)
    table_size = np.asarray(layout["table"]["size_xy"], dtype=float)
    table_center = np.asarray(layout["table"]["center_xy"], dtype=float)
    floor_z = float(layout["floor_z"])
    table_top_z = float(layout["table"]["top_z"])
    base_top_z = float(layout["robot_base"]["top_z"])

    for name, value, shape in (
        ("robot_base.size", base_size, (3,)),
        ("robot_base.center", base_center, (3,)),
        ("robot_mount.position", mount_position, (3,)),
        ("table.size_xy", table_size, (2,)),
        ("table.center_xy", table_center, (2,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite shape {shape}, got {value}")
    if np.any(base_size <= 0.0) or np.any(table_size <= 0.0):
        raise ValueError("workcell dimensions must be positive")
    if not np.isclose(base_center[2] - base_size[2] / 2.0, floor_z):
        raise ValueError("robot pedestal bottom must coincide with the floor")
    if not np.isclose(base_center[2] + base_size[2] / 2.0, base_top_z):
        raise ValueError("robot pedestal top is inconsistent with its size/center")
    if not np.isclose(mount_position[2], base_top_z):
        raise ValueError("RB3 mount must coincide with the pedestal top")
    if not np.isclose(table_top_z - floor_z, 0.72):
        raise ValueError("table surface must be 0.72 m above the floor")

    # The table's near X edge touches the pedestal's forward (+X) edge.
    pedestal_forward_edge = base_center[0] + base_size[0] / 2.0
    table_near_edge = table_center[0] - table_size[0] / 2.0
    if not np.isclose(pedestal_forward_edge, table_near_edge):
        raise ValueError("table must touch the +X edge of the robot pedestal")
    return layout


WORKCELL_LAYOUT = load_workcell_layout()
ROBOT_MOUNT_POSITION = tuple(WORKCELL_LAYOUT["robot_mount"]["position"])
ROBOT_MOUNT_QUATERNION_XYZW = tuple(
    WORKCELL_LAYOUT["robot_mount"]["quaternion_xyzw"]
)

