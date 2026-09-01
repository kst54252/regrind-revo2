"""Dynamic YCB 007 tuna-fish-can asset used by the RB3+Revo2 task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from regrind.assets import REGRIND_PROJECT_ROOT


TUNA_CAN_VISUAL_USD_PATH = REGRIND_PROJECT_ROOT / "007_tuna_fish_can" / "textured_simple.usd"
TUNA_CAN_USD_PATH = REGRIND_PROJECT_ROOT / "USD" / "tuna_fish_can_rigid.usda"

TUNA_CAN_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TUNA_CAN_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=2.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        # The wrapper authors a bounds-derived cylinder collider.  The file
        # spawner accepts one CollisionPropertiesCfg in IsaacLab 3.x.
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.020875),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
)
