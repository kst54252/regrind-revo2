import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from regrind.assets import REGRIND_ASSETS_DIR

SCISSORS_OBJECT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{REGRIND_ASSETS_DIR}/scissors/scissors_flattened.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        ),
    ),
    actuators={},
    init_state=ArticulationCfg.InitialStateCfg(
        # Root pose (position and orientation)
        pos=(0.0, 0.0, 1.0),  # Position in world frame (x, y, z)
        rot=(0.0, 0.0, 0.0, 1.0),  # Quaternion rotation (x, y, z, w)
        # Root velocity
        lin_vel=(0.0, 0.0, 0.0),  # Linear velocity (x, y, z)
        ang_vel=(0.0, 0.0, 0.0),  # Angular velocity (x, y, z)
        # Joint positions
        joint_pos={
            "scissors_joint": 0.0,
        }
    ),
)

SCISSORS_2X_3D_PRINTED_OBJECT_CFG = SCISSORS_OBJECT_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{REGRIND_ASSETS_DIR}/scissors_2x_3d_printed/scissors_2x_3d_printed_flattened.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=5.0,
        ),
    )
)

REAL_SCISSORS_OBJECT_CFG = SCISSORS_OBJECT_CFG.replace(
    actuators={
        "scissors_joint": ImplicitActuatorCfg(
            joint_names_expr=["scissors_joint"],
            stiffness=0.0,
            damping=0.0,
            friction=0.2,
        ),
    },
)
