"""Isaac Lab articulation configuration for the assembled RB3-730 + Revo2."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from regrind.assets import REGRIND_PROJECT_ROOT
from regrind.data.rb3_revo2_reference import (
    RB3_JOINT_NAMES,
    REFERENCE_JOINT_NAMES,
    REVO2_JOINT_NAMES,
)


RB3_REVO2_USD_PATH = REGRIND_PROJECT_ROOT / "USD" / "rb3_revo2.usd"

REVO2_FOLLOWER_JOINTS = {
    "right_thumb_distal_joint": ("right_thumb_proximal_joint", 1.0, 0.0),
    "right_index_distal_joint": ("right_index_proximal_joint", 1.155, 0.0),
    "right_middle_distal_joint": ("right_middle_proximal_joint", 1.155, 0.0),
    "right_ring_distal_joint": ("right_ring_proximal_joint", 1.155, 0.0),
    "right_pinky_distal_joint": ("right_pinky_proximal_joint", 1.155, 0.0),
}

_ARM_STIFFNESS = dict(zip(RB3_JOINT_NAMES, (300.0, 500.0, 500.0, 300.0, 200.0, 50.0)))
_ARM_DAMPING = dict(zip(RB3_JOINT_NAMES, (20.0, 20.0, 20.0, 20.0, 20.0, 10.0)))
_ARM_EFFORT = dict(zip(RB3_JOINT_NAMES, (10.0, 100.0, 100.0, 100.0, 100.0, 10.0)))

_HAND_ALL_JOINTS = REVO2_JOINT_NAMES + tuple(REVO2_FOLLOWER_JOINTS)
_HAND_STIFFNESS = {name: 3.0 for name in _HAND_ALL_JOINTS}
_HAND_DAMPING = {name: 0.1 for name in _HAND_ALL_JOINTS}
_HAND_EFFORT = {name: 0.5 for name in _HAND_ALL_JOINTS}


RB3_REVO2_CFG = ArticulationCfg(
    # The assembled stage contains the RB3 and Revo2 as sibling branches.
    # Point IsaacLab at the single articulation-root API explicitly so it
    # does not mistake referenced source prims for multiple articulations.
    articulation_root_prim_path="/rb3_730es_u/Geometry",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(RB3_REVO2_USD_PATH),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="force",
            max_velocity=1000.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            # Imported visual/collision meshes contain adjacent overlaps.  A
            # baseline tracking task keeps robot-object contacts enabled but
            # disables robot self-contact to avoid an initial PhysX explosion.
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    actuators={
        "rb3_arm": ImplicitActuatorCfg(
            joint_names_expr=list(RB3_JOINT_NAMES),
            stiffness=_ARM_STIFFNESS,
            damping=_ARM_DAMPING,
            effort_limit_sim=_ARM_EFFORT,
            velocity_limit_sim=10.0,
        ),
        "revo2_hand": ImplicitActuatorCfg(
            joint_names_expr=list(_HAND_ALL_JOINTS),
            stiffness=_HAND_STIFFNESS,
            damping=_HAND_DAMPING,
            effort_limit_sim=_HAND_EFFORT,
            velocity_limit_sim=100.0,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={name: 0.0 for name in REFERENCE_JOINT_NAMES + tuple(REVO2_FOLLOWER_JOINTS)},
    ),
    soft_joint_pos_limit_factor=1.0,
)
