"""Launch the RB3+Revo2 replay directly from a terminal.

This starts a full Isaac Sim GUI, opens the assembled robot Stage, loads the
YCB object visual, and replays the 12-DoF reference.  ``--physics-object``
makes the can dynamic so only gravity and robot contact move it. Close the
Isaac Sim window to terminate the process.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = PROJECT_ROOT / "USD" / "rb3_revo2.usd"
ISAAC_REFERENCE_ROOT = PROJECT_ROOT / "outputs" / "isaac" / "dexycb"
DEFAULT_SEQUENCE = "20200709_143626_right"
DEFAULT_OBJECT_MESH = PROJECT_ROOT / "007_tuna_fish_can" / "textured_simple.obj"
REPLAY_SCRIPT = Path(__file__).with_name("replay_reference_isaac_sim.py")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument(
        "--sequence",
        default=DEFAULT_SEQUENCE,
        help=(
            "Prepared DexYCB sequence name under outputs/isaac/dexycb. "
            "Ignored when --trajectory is supplied."
        ),
    )
    parser.add_argument("--trajectory", type=Path, help="Explicit 12-DoF reference override.")
    parser.add_argument("--list-sequences", action="store_true")
    parser.add_argument("--object-mesh", type=Path, default=DEFAULT_OBJECT_MESH)
    parser.add_argument(
        "--physics-object",
        action="store_true",
        help=(
            "Place the can once as a dynamic rigid body; drive the robot with "
            "joint position targets and let gravity/contact move the can."
        ),
    )
    parser.add_argument("--object-mass", type=float, default=0.15, help="Can mass in kg.")
    parser.add_argument(
        "--object-friction", type=float, default=0.8, help="Static/dynamic friction."
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--dt",
        type=float,
        help="Override frame period in seconds; default uses trajectory FPS.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Loop playback (default: on for kinematic, off for physics-object).",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Open at the start frame without automatically playing.",
    )
    parser.add_argument(
        "--demo-skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the pre-retargeting DexYCB/MANO 21-joint skeleton.",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--camera-eye", nargs=3, type=float, default=(1.25, 1.15, 0.85), metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--camera-target", nargs=3, type=float, default=(0.25, 0.10, 0.28), metavar=("X", "Y", "Z")
    )
    return parser


def _resolved_existing(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _stage_is_loading(context):
    pending, loaded, _ = context.get_stage_loading_status()
    return bool(pending or loaded)


def main():
    args = _parser().parse_args()
    available_sequences = sorted(
        path.parent.name
        for path in ISAAC_REFERENCE_ROOT.glob("*/rb3_revo2_reference.h5")
    )
    if args.list_sequences:
        print("Available Isaac replay sequences:")
        for sequence in available_sequences:
            print(f"  {sequence}")
        return
    # SimulationApp also inspects sys.argv. Do not forward this launcher's
    # --speed/--trajectory options to Kit as unrelated application flags.
    sys.argv = [sys.argv[0]]
    if args.speed <= 0:
        raise ValueError("--speed must be > 0")
    if args.object_mass <= 0 or args.object_friction < 0:
        raise ValueError("--object-mass must be > 0 and --object-friction must be >= 0")
    if args.dt is not None and args.dt <= 0:
        raise ValueError("--dt must be > 0")
    stage_path = _resolved_existing(args.stage, "robot Stage")
    trajectory_candidate = args.trajectory or (
        ISAAC_REFERENCE_ROOT / args.sequence / "rb3_revo2_reference.h5"
    )
    trajectory_path = _resolved_existing(trajectory_candidate, "trajectory")
    object_mesh_path = _resolved_existing(args.object_mesh, "object mesh")

    os.environ["REVO2_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["REVO2_TRAJECTORY_PATH"] = str(trajectory_path)
    os.environ["REVO2_OBJECT_MESH_PATH"] = str(object_mesh_path)
    os.environ["REVO2_REPLAY_SPEED"] = str(args.speed)
    loop = (not args.physics_object) if args.loop is None else args.loop
    os.environ["REVO2_REPLAY_LOOP"] = "1" if loop else "0"
    os.environ["REVO2_REPLAY_AUTO_PLAY"] = "0" if args.paused else "1"
    os.environ["REVO2_REPLAY_START_FRAME"] = str(args.start_frame)
    os.environ["REVO2_SHOW_DEMO_SKELETON"] = "1" if args.demo_skeleton else "0"
    os.environ["REVO2_PHYSICS_OBJECT"] = "1" if args.physics_object else "0"
    os.environ["REVO2_OBJECT_MASS_KG"] = str(args.object_mass)
    os.environ["REVO2_OBJECT_FRICTION"] = str(args.object_friction)
    if args.dt is not None:
        os.environ["REVO2_REPLAY_DT"] = str(args.dt)

    # Isaac/Omniverse imports must happen after SimulationApp construction.
    print("[launcher] starting Isaac Sim GUI (first launch may compile shaders)...", flush=True)
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": False,
            "width": args.width,
            "height": args.height,
            "enable_cameras": True,
        }
    )

    import omni.ui as ui
    import omni.usd
    from isaacsim.core.utils.viewports import set_camera_view

    context = omni.usd.get_context()
    print(f"[launcher] opening Stage: {stage_path}")
    context.open_stage(str(stage_path))
    deadline = time.monotonic() + 90.0
    while simulation_app.is_running() and _stage_is_loading(context):
        if time.monotonic() > deadline:
            raise TimeoutError("USD Stage loading timed out")
        simulation_app.update()
    for _ in range(5):
        simulation_app.update()
    set_camera_view(args.camera_eye, args.camera_target)

    namespace = {"__name__": "__main__", "__file__": str(REPLAY_SCRIPT)}
    source = REPLAY_SCRIPT.read_text(encoding="utf-8")
    exec(compile(source, str(REPLAY_SCRIPT), "exec"), namespace)
    replay = namespace["REPLAY"]

    # The terminal launcher needs controls that do not depend on Script Editor.
    sequence_label = trajectory_path.parent.name
    replay_mode = "PHYSICS CAN" if args.physics_object else "KINEMATIC"
    control_window = ui.Window(
        f"RB3 + Revo2 Replay | {replay_mode} | {sequence_label}", width=540, height=205
    )
    with control_window.frame:
        with ui.VStack(spacing=6):
            status = ui.Label("Initializing Isaac articulation...", height=24)
            ui.Label(f"Sequence: {sequence_label}", height=22)
            ui.Label(f"Mode: {replay_mode}", height=22)
            with ui.HStack(height=34, spacing=5):
                ui.Button("Play", clicked_fn=replay.play)
                ui.Button("Pause", clicked_fn=replay.pause)
                ui.Button("Reset", clicked_fn=replay.reset)
                ui.Button("-1", clicked_fn=lambda: replay.seek(max(0, replay.current_frame - 1)))
                ui.Button(
                    "+1",
                    clicked_fn=lambda: replay.seek(
                        min(replay.trajectory.frames - 1, replay.current_frame + 1)
                    ),
                )
                ui.Button("Summary", clicked_fn=replay.summary)
            with ui.HStack(height=30, spacing=5):
                ui.Label("Go to frame", width=95)
                frame_model = ui.SimpleIntModel(args.start_frame)
                ui.IntField(model=frame_model)
                ui.Button(
                    "Go",
                    width=60,
                    clicked_fn=lambda: replay.seek(frame_model.as_int),
                )
            ui.Label(
                (
                    "Can: gravity/contact only | Magenta: MANO | Cyan: wrist target"
                    if args.physics_object
                    else "Magenta: MANO | Cyan: wrist path | Red: FK | Green: target"
                ),
                height=22,
            )

    print("[launcher] GUI ready. Close the Isaac Sim window to exit.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
            initialize_task = namespace.get("_INITIALIZE_TASK")
            if (
                initialize_task is not None
                and initialize_task.done()
                and initialize_task.exception() is not None
            ):
                error = initialize_task.exception()
                print("[launcher] replay initialization failed:", repr(error), flush=True)
                traceback.print_exception(type(error), error, error.__traceback__)
                raise error
            state = "PLAYING" if replay.playing else "PAUSED"
            status.text = (
                f"{replay_mode} | {state}  |  frame {replay.current_frame:03d}/"
                f"{replay.trajectory.frames - 1:03d}  |  "
                f"{1.0 / replay.trajectory.dt:.3g} Hz  |  speed {args.speed:g}x"
            )
    finally:
        replay.pause()
        simulation_app.close()


if __name__ == "__main__":
    main()
