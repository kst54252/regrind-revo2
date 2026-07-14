"""Per-fingertip vs table filtered contact sensors for WujiHand (Isaac Lab ContactSensor)."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg

# Scene sensor keys: ``contact_fingertip{i}_table`` for thumb … pinky (finger 1 … 5).
WUJI_FINGERTIP_TABLE_CONTACT_SENSOR_NAMES: tuple[str, ...] = tuple(
    f"contact_fingertip{i}_table" for i in range(1, 6)
)


def install_wuji_fingertip_table_contact_sensors(scene) -> None:
    """Register five single-body sensors (distal link vs ``Table``) on ``scene``.

    Each sensor exposes ``ContactSensorData.force_matrix_w`` for normal forces between
    ``right_finger{i}_link4`` and the table rigid body only.

    Activates contact reporters on the robot USD spawn if needed.
    """
    robot = getattr(scene, "robot", None)
    if robot is None:
        return
    if not getattr(robot.spawn, "activate_contact_sensors", False):
        scene.robot = robot.replace(spawn=robot.spawn.replace(activate_contact_sensors=True))
    table_filter = rf"{{ENV_REGEX_NS}}/Table"
    for i in range(1, 6):
        attr = f"contact_fingertip{i}_table"
        if getattr(scene, attr, None) is not None:
            continue
        setattr(
            scene,
            attr,
            ContactSensorCfg(
                prim_path=rf"{{ENV_REGEX_NS}}/Robot/root/right_finger{i}_link4",
                filter_prim_paths_expr=[table_filter],
                history_length=3,
                track_air_time=False,
                debug_vis=False,
                force_threshold=0.1,
            ),
        )
