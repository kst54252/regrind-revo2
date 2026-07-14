from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

TABLE_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.7, 0.7, 0.4),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
            torsional_patch_radius=0.02,
            min_torsional_patch_radius=0.005,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.948 - 0.2),
    ),
)
