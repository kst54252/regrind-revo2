import xml.etree.ElementTree as ET

import numpy as np

from regrind.retargeting import revo2_constants as rc


def test_revo2_semantic_keypoint_config_is_complete_and_ordered():
    assert rc.ROBOT_DOF == 6
    assert rc.ROBOT_MODEL_DOF == 11
    assert list(rc.MANO_TO_REVO2_MAPPING) == [str(index) for index in range(21)]
    assert list(rc.SEMANTIC_KEYPOINTS) == [str(index) for index in range(21)]
    xyz = np.stack([rc.SEMANTIC_KEYPOINTS[str(i)]["xyz"] for i in range(21)])
    assert xyz.shape == (21, 3)
    assert np.isfinite(xyz).all()


def test_revo2_urdf_contains_configured_links_and_joints():
    root = ET.parse(rc.ROBOT_URDF_FILE).getroot()
    link_names = {element.attrib["name"] for element in root.findall("link")}
    joint_names = {element.attrib["name"] for element in root.findall("joint")}

    assert set(rc.ACTUATED_JOINT_NAMES) <= joint_names
    assert set(rc.MIMIC_JOINTS) <= joint_names
    assert {
        point["parent_link"] for point in rc.SEMANTIC_KEYPOINTS.values()
    } <= link_names


def test_revo2_joint_limits_and_defaults_are_valid():
    assert rc.JOINT_LOWER.shape == (6,)
    assert rc.JOINT_UPPER.shape == (6,)
    assert rc.DEFAULT_JOINT_POSITIONS.shape == (6,)
    assert np.all(rc.JOINT_LOWER <= rc.DEFAULT_JOINT_POSITIONS)
    assert np.all(rc.DEFAULT_JOINT_POSITIONS <= rc.JOINT_UPPER)
