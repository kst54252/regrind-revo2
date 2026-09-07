# Isaac Sim replay

[Repository map](architecture.md) · [Current status](current-status.md) ·
[RL and online deployment](RL_TASK.md)

For wrist/root/link6/mount definitions, conversion equations, and same-sample
PhysX versus USD logs, see [wrist frame diagnosis](WRIST_FRAME_DIAGNOSIS.md).
For baseline/candidate policy evaluation and deterministic command replay, start
with the [diagnostic command index](../scripts/README.md#experiments-and-failure-reproduction-opt-in).

## Start

```bash
./scripts/run_isaac_replay.sh --list-sequences
./scripts/run_isaac_replay.sh --sequence 20200709_143747_left
```

The viewer shows RB3-730, Revo2, the tuna mesh, source MANO21 skeleton, and
wrist target/FK markers. Its control window supports play, pause, reset,
previous/next frame, and direct frame selection.

## Replay modes

### Kinematic default

The replay writes all 12 reference joints per frame and moves the object along
the recorded trajectory. Use it to verify trajectory/FK correspondence, not
contact success.

```bash
./scripts/run_isaac_replay.sh --sequence 20200709_143747_left
```

### Dynamic object comparison

`--physics-object` spawns the can once and then lets gravity, the table, and
robot contact move it. Robot modes are:

- `kinematic`: write arm and hand states directly.
- `arm-kinematic`: write RB3 state while driving the hand.
- `position`: drive both RB3 and Revo2; tracking error is expected.

```bash
./scripts/run_isaac_replay.sh \
  --sequence 20200709_143747_left --physics-object
```

Open-loop dynamic replay is a contact comparison, not a trained closed-loop
grasp. For policy feedback plus online IK, use `./scripts/rl.sh play-arm` as
documented in [RL_TASK.md](RL_TASK.md).

## Workcell coordinates

Replay and the legacy arm environment share
`config/workcell/rb3_revo2_table.json`:

- table top: world `Z=0`
- floor: `Z=-0.72 m`
- pedestal top and RB3 mount: `Z=-0.02 m`
- pedestal: `0.50 x 0.50 m`
- table: `X 0.80 x Y 1.60 m`, attached in robot `+X`

References generated for this layout record the RB3 base pose. Do not change
only the stage geometry without regenerating or transforming the reference.

## Replay a converted policy rollout

```bash
./scripts/run_isaac_replay.sh \
  --trajectory outputs/floating/20200709_143747_left/reference_12dof.h5 \
  --demo-skeleton --no-loop
```

Without `--physics-object`, both the learned rollout state and recorded can pose
are visualized; that is not a new physical grasp trial. Add
`--physics-object --robot-control arm-kinematic` for a contact comparison.

For automated kinematic validation add `--headless --exit-after-replay`. Check
joint ordering/limits, finite values, continuity, wrist FK error, and expected
frame count. Isaac execution is required for articulation, rendering, and
contact claims.

## Common options

- `--speed 0.5`: replay speed
- `--paused`: start paused
- `--start-frame N`: initial frame
- `--dt SECONDS`: override trajectory time step
- `--no-demo-skeleton`: hide MANO21
- `--object-mass KG`, `--object-friction VALUE`: dynamic-can properties
- `--loop` / `--no-loop`: repeat behavior

The implementation and diagnostic options are documented in
[`tools/rb3_revo2_ik/README.md`](../tools/rb3_revo2_ik/README.md).
