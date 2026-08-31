import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from regrind.assets import REGRIND_ASSETS_DIR

SCREWDRIVER_OBJECT_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{REGRIND_ASSETS_DIR}/screwdriver/screwdriver_flattened.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
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
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        # Root pose (position and orientation)
        pos=(0.0, 0.0, 1.0),  # Position in world frame (x, y, z)
        rot=(0.0, 0.0, 0.0, 1.0),  # Quaternion rotation (x, y, z, w)
        # Root velocity
        lin_vel=(0.0, 0.0, 0.0),  # Linear velocity (x, y, z)
        ang_vel=(0.0, 0.0, 0.0),  # Angular velocity (x, y, z)
    ),
)

SCREWDRIVER_1_5X_3D_PRINTED_OBJECT_CFG = SCREWDRIVER_OBJECT_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{REGRIND_ASSETS_DIR}/screwdriver_1_5x_3d_printed/screwdriver_1_5x_3d_printed_flattened.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
    )
)
