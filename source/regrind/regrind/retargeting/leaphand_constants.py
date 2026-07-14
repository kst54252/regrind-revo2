"""LeapHand retargeting configuration (hand-object subset).

Slimmed from ``omniretarget/src/leaphand_constants.py``: only the LeapHand + object
fields used by the hand-object interaction-mesh retargeter are kept (the SMPL /
full-body humanoid tables are dropped). Asset paths resolve against the packaged
``regrind/assets`` dir; demo / keypoint paths resolve against ``$REGRIND_DATA_DIR``,
matching the env-cfg convention.
"""

import os

from regrind.assets import REGRIND_ASSETS_DIR

ROBOT_NAME = "leaphand"
ROBOT_DOF = 16
ROBOT_URDF_FILE = str(REGRIND_ASSETS_DIR / "leaphand_urdf" / "leap_hand_right.urdf")

MANO_JOINTS = [str(i) for i in range(21)]
MANO_TO_LEAP_MAPPING = {
    # Wrist
    "0": "mano_0",  # palm_lower
    # Thumb
    "13": "mano_13",
    "14": "mano_14",
    "15": "mano_15",
    "16": "thumb_tip_head",
    # Index
    "1": "mano_1",
    "2": "mano_2",
    "3": "mano_3",
    "17": "index_tip_head",
    # Middle
    "4": "mano_4",
    "5": "mano_5",
    "6": "mano_6",
    "18": "middle_tip_head",
    # Ring
    "10": "mano_10",
    "11": "mano_11",
    "12": "mano_12",
    "19": "ring_tip_head",
}

# Static per-object configuration. Paths are relative; resolve via get_object_config().
OBJECT_CONFIGS = {
    "scissors": {
        "obj_scale": 2.0,
        "obj_scale_suffix": "_2x",
        "object_urdf": "scissors/scissors_2x.urdf",
        "demo_path": "arctic_demo/arctic_leap_scissors_2x/demo_30fps.h5",
        "demo_data_type": "arctic",
        "keypoint_paths": {
            "bottom": "keypoints/scissors_bottom_100_filtered_handles.npy",
            "top": "keypoints/scissors_top_100_filtered_handles.npy",
        },
        "table_height": 0.93,
    },
    "screwdriver": {
        "obj_scale": 1.5,
        "obj_scale_suffix": "_1.5x",
        "object_urdf": "screwdriver/screwdriver_1.5x.urdf",
        "demo_path": "zed_mocap_demo/screwdriver_1_5x/demo_30fps.h5",
        "demo_data_type": "zed_mocap",
        "keypoint_paths": {
            "bottom": "keypoints/screwdriver_100.npy",
        },
        "table_height": 0.502,
    },
}


def get_object_config(object_name: str) -> dict:
    """Resolve absolute asset/demo/keypoint paths for ``object_name``.

    Object URDFs resolve under the packaged ``regrind/assets`` dir; demo and keypoint
    files resolve under ``$REGRIND_DATA_DIR``.
    """
    if object_name not in OBJECT_CONFIGS:
        raise ValueError(
            f"Unknown object {object_name!r} for {ROBOT_NAME}; "
            f"expected one of {sorted(OBJECT_CONFIGS)}."
        )
    cfg = dict(OBJECT_CONFIGS[object_name])
    data_dir = os.environ["REGRIND_DATA_DIR"]
    cfg["object_urdf_file"] = str(REGRIND_ASSETS_DIR / cfg["object_urdf"])
    cfg["demo_file"] = os.path.join(data_dir, cfg["demo_path"])
    cfg["object_keypoints_paths"] = {
        key: os.path.join(data_dir, rel) for key, rel in cfg["keypoint_paths"].items()
    }
    return cfg
