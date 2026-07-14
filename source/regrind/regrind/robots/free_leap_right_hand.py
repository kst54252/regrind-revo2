import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from regrind.assets import REGRIND_ASSETS_DIR

ACTUATED_JOINT_NAMES = [
    # fingers (16) in QPOS_INDEX order: index → middle → ring → thumb
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]

RELATIVE_ACTION_SCALE = {
    "(if|mf|rf|th)_(mcp|rot|pip|dip|cmc|axl|ipl)": 3.2,  # fingers
}


FREE_LEAP_RIGHT_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{REGRIND_ASSETS_DIR}/free_leap_right_hand/floating_leap_right_hand_flattened.usda",
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="force",
            max_effort=1.0e6,
            max_velocity=1000.0,
        ),  # Need to set this for position control
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["(if|mf|rf|th)_(mcp|rot|pip|dip|cmc|axl|ipl)"],
            stiffness=3.0,
            damping=0.1,
            effort_limit_sim=0.5,
            velocity_limit_sim=100.0,
            friction=0.01,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "if_mcp": 0.0,
            "if_rot": 0.0,
            "if_pip": 0.0,
            "if_dip": 0.0,
            "mf_mcp": 0.0,
            "mf_rot": 0.0,
            "mf_pip": 0.0,
            "mf_dip": 0.0,
            "rf_mcp": 0.0,
            "rf_rot": 0.0,
            "rf_pip": 0.0,
            "rf_dip": 0.0,
            "th_cmc": 0.0,
            "th_axl": 0.0,
            "th_mcp": 0.0,
            "th_ipl": 0.0,
        }
    ),
    soft_joint_pos_limit_factor=1.0
)
