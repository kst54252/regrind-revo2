"""Revo2 right-hand configuration for interaction-mesh retargeting.

``revo2_keypoints.json`` is the source of truth for the MANO index order,
semantic keypoint parent links, and local coordinates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from regrind.assets import REGRIND_ASSETS_DIR


ROBOT_NAME = "revo2"
ROBOT_DOF = 6
ROBOT_MODEL_DOF = 11  # six leaders plus five dependent distal joints in Drake
ROBOT_ASSET_DIR = REGRIND_ASSETS_DIR / "revo2"
ROBOT_URDF_FILE = os.environ.get(
    "REVO2_URDF_FILE", str(ROBOT_ASSET_DIR / "urdf" / "revo2_right.urdf")
)
KEYPOINTS_FILE = Path(
    os.environ.get(
        "REVO2_KEYPOINTS_FILE", str(ROBOT_ASSET_DIR / "revo2_keypoints.json")
    )
)

WRIST_LINK_NAME = "right_hand_base_link"
ACTUATED_JOINT_NAMES = (
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)
JOINT_LOWER = np.zeros(ROBOT_DOF, dtype=float)
JOINT_UPPER = np.array((1.57, 1.03, 1.41, 1.41, 1.41, 1.41), dtype=float)
DEFAULT_JOINT_POSITIONS = np.zeros(ROBOT_DOF, dtype=float)

# follower: (leader, multiplier, offset), matching the URDF mimic tags.
MIMIC_JOINTS = {
    "right_thumb_distal_joint": ("right_thumb_proximal_joint", 1.0, 0.0),
    "right_index_distal_joint": ("right_index_proximal_joint", 1.155, 0.0),
    "right_middle_distal_joint": ("right_middle_proximal_joint", 1.155, 0.0),
    "right_ring_distal_joint": ("right_ring_proximal_joint", 1.155, 0.0),
    "right_pinky_distal_joint": ("right_pinky_proximal_joint", 1.155, 0.0),
}


def _load_semantic_keypoints(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    try:
        with path.open("r", encoding="utf-8") as json_file:
            raw = json.load(json_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Revo2 keypoint JSON not found: {path}. Set REVO2_KEYPOINTS_FILE "
            "or install the packaged asset."
        ) from error

    records = []
    iterable = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for outer_index, item in iterable:
        mano_index = int(item.get("mano_index", outer_index))
        records.append((mano_index, item))
    records.sort(key=lambda record: record[0])

    indices = [index for index, _ in records]
    if indices != list(range(21)):
        raise ValueError(
            "Revo2 keypoint JSON must contain MANO indices 0..20 exactly once; "
            f"found {indices}"
        )

    mapping = {}
    semantic_points = {}
    for index, item in records:
        key = str(index)
        try:
            parent_link = str(item["parent_link"])
            xyz = np.asarray(item["xyz"], dtype=float)
            name = str(item["name"])
        except KeyError as error:
            raise ValueError(
                f"Revo2 keypoint {index} is missing field {error.args[0]!r}"
            ) from error
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"Revo2 keypoint {index} xyz must be finite shape (3,)")
        mapping[key] = parent_link
        semantic_points[key] = {
            "mano_index": index,
            "name": name,
            "parent_link": parent_link,
            "xyz": xyz,
        }
    return mapping, semantic_points


MANO_TO_REVO2_MAPPING, SEMANTIC_KEYPOINTS = _load_semantic_keypoints(KEYPOINTS_FILE)
MANO_JOINTS = list(MANO_TO_REVO2_MAPPING)

OBJECT_CONFIGS = {
    "tuna_fish_can": {
        "obj_scale": 1.0,
        "obj_scale_suffix": "",
        "object_urdf_file": str(
            REGRIND_ASSETS_DIR / "tuna_fish_can" / "tuna_fish_can.urdf"
        ),
        "object_keypoints_paths": {
            "bottom": str(
                REGRIND_ASSETS_DIR / "tuna_fish_can" / "object_points_50.npy"
            )
        },
        "object_body_name": "tuna_fish_can",
        "table_height": None,
    }
}


def get_object_config(object_name: str) -> dict:
    """Return packaged object geometry; DexYCB demo paths remain CLI inputs."""
    if object_name not in OBJECT_CONFIGS:
        raise ValueError(
            f"Unknown object {object_name!r} for Revo2; expected one of "
            f"{sorted(OBJECT_CONFIGS)}, or pass explicit object path arguments."
        )
    return dict(OBJECT_CONFIGS[object_name])
