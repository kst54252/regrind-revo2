# Wrist frame contract and diagnostic

[Replay](ISAAC_SIM_REPLAY.md) · [RL task](RL_TASK.md)

Verified 2026-09-07 using the current vertical mount, the stable
`20200709_143747_left` reference, and recorded X=0.50/Y=0 online telemetry.
No controller, policy, actuator, or RL configuration was changed by this audit.

## Frames and actual code paths

`F` is the floating articulation root link, `B` the physical Revo2 palm base,
`L` RB3 link6, `M` its Revo2 mount, and `C` the stock RB3 TCP.

| Frame | Source USD path | Meaning |
|---|---|---|
| F | `/revo2_floating` in `USD/revo2_floating.usda` | Floating rigid root, not its center of mass |
| B, floating | `/revo2_floating/Geometry/world/right_hand_base_link` | Physical palm/wrist |
| Revo2 container | `/World/revo2_right` in `USD/rb3_revo2_vertical.usda` | Assembly namespace/initial placement, not a live wrist reader |
| Geometry container | `/World/revo2_right/Geometry/world` | Not the mounted physical palm frame |
| B, mounted | `/World/revo2_right/Geometry/world/right_hand_base_link` | Actual IK end-effector frame |
| L | `/World/rb3_730es_u/Geometry/link0/link1/link2/link3/link4/link5/link6` | Last arm link, before flange/adapter offset |
| M | `L/revo2_mount` | Arm-side attachment frame |
| Hand attachment | `B/rb3_mount` | Coincides with palm base to numerical precision |
| C | `L/tcp` | Stock TCP, different from B |

The diagnostic remaps `/World` to `/World/envs/env_0/Robot` for the assembled
task and spawns the same floating asset at `/World/FloatingProbe`.
Each log header contains the exact runtime paths.

- `robots/free_revo2_right_hand.py` loads the floating overlay. Its
  `Physics/right_hand_base_joint` attaches F to B with identity local frames.
- `mdp/actions.py::SE3ImpedanceActionTerm` compares the target against root link
  position/orientation. `FloatingActionsCfg` applies its wrench at
  `right_hand_base_link`; the floating command observes that same palm body.
- `mdp/rb3_revo2_actions.py::RB3WristIKAction` builds the same translation and
  left-composed rotation-vector residual target and passes it directly to IK.
- `tools/rb3_revo2_ik/rb3_kinematics.py::forward` includes the link6-to-B offset;
  `inverse` therefore expects B, not L or the stock TCP.
- Standalone `replay_reference_isaac_sim.py` obtains the wrist reader path from
  `rb3_model.json["mounted_wrist_frame"]` and reads it with `SingleRigidPrim`.
  It separately checks viewport/USD synchronization.
- Offline `scripts/floating_to_rb3.sh` defaults `--wrist-rpy` to `(0,0,0)`.
  A manually supplied correction is a separate trajectory operation, not a
  required adapter correction for this model.

## Fixed transforms and conversion

Use column vectors: `T_A_B` maps B coordinates into A. All lengths are meters.

```text
T_F_B = I
T_L_M = Trans(0, 0, 0.141304972)
T_M_B ≈ I
T_L_C = Trans(0, 0, 0.100000000)
T_C_B = Trans(0, 0, 0.041304972)
```

The assembler joint retains local numerical residuals of a few nanometers
and about 6e-8 rad; these are included in measured PhysX transforms and are
well below validation tolerances. The nominal model uses identity rotation.
The mounted Revo2 container-to-base relation is **not** a fixed runtime wrist
transform: the container can remain at its authored placement while B moves.

For an arbitrary floating root target:

```python
T_world_base_expected = T_world_floating_target @ T_floating_base

# Current repository IK already includes the adapter:
T_world_ik_target = T_world_base_expected
q = kin.inverse(
    T_world_ik_target[:3, 3],
    Rotation.from_matrix(T_world_ik_target[:3, :3]).as_quat(),  # XYZW
    initial_q=q_previous,
)

# Only for a different solver whose endpoint is bare link6:
T_world_link6_target = T_world_base_expected @ np.linalg.inv(T_link6_base)

# Only for a solver whose endpoint is the stock TCP:
T_world_tcp_target = T_world_link6_target @ T_link6_tcp
```

Thus the current bridge requires neither another 141 mm shift nor a 180-degree
flip. Offsets are applied along rotated **local** axes, not world Z.
For parallel environments the command coordinates are environment-relative;
add that environment's origin to obtain world positions. The diagnostic uses
environment 0, whose origin is zero. RB3 base placement is `(0,0,-0.02)`.

OpenUSD `Gf.Matrix4d` uses row vectors. Before the equations above:

```python
T_world_prim = np.asarray(cache.GetLocalToWorldTransform(prim)).T
T_A_B = np.linalg.inv(T_world_A) @ T_world_B
```

## Execute and read logs

Reference target check at X=0.50 (original X=0.40 plus 0.10):

```bash
bash scripts/diagnose_wrist_frames.sh --headless --offset-xy 0.1 0 \
  --output outputs/diagnostics/wrist_frames_new_reference.jsonl
```

Reconstruct the measured arm joints and target from each same-time recorded
sample; this does not run the policy again:

```bash
bash scripts/diagnose_wrist_frames.sh --headless \
  --telemetry outputs/diagnostics/rb3_arm_tracking_x0p50_y0.npz \
  --output outputs/diagnostics/wrist_frames_new_recorded.jsonl
```

Outputs refuse to overwrite existing files. Add `--reference PATH` to choose
another reference, or `--max-frames N` for a short check. No physics integration
occurs after initialization: snapshots isolate the frame contract from tracking.
The floating probe is placed at the target, not replaying a measured floating
policy trajectory. Telemetry mode reconstructs six measured arm joints; fingers
are not replayed because they do not change any of the audited wrist frames.

JSONL contains a metadata header, one record per sample, and a summary:

- `floating_wrist_target`, `expected_revo2_base`, `actual_mounted_base`,
  `ik_target`, `link6_target`, and `ik_fk_at_actual_q`: 4x4 matrices.
- `world_runtime`: simultaneous PhysX root/link/palm measurements; non-rigid
  attachment/TCP frames are derived from those poses and USD local transforms.
- `world_usd`: raw XformCache readings, including the two container prims.
- `fixed_transforms`: measured rigid relationships and USD local attachments.
- `errors`: position meters, orientation radians and degrees.
- With telemetry, `recorded_actual_mounted_base` is the original physical pose;
  `actual_mounted_base` is its independent joint-state reconstruction.

`source_frame_index` in existing online telemetry is the command index after
`env.step`; it can be one frame ahead of the action target. This tool pairs
stored target/actual arrays by sample and does not substitute that later
reference pose. `sample_time_s` is relative sample time, not wall-clock time.

## Observed results

| Check | Samples | Max position error | Max orientation error |
|---|---:|---:|---:|
| X=0.50 reference target → mounted base after strict IK | 38 | 4.65e-7 m | 1.31e-6 rad |
| Recorded arm-q FK → reconstructed physical base | 37 | 3.95e-7 m | 1.02e-6 rad |
| Recorded measured base → reconstructed physical base | 37 | 6.15e-8 m | 0 rad |
| Recorded action target → recorded measured base | 37 | 0.05043 m | 0.40107 rad (22.98°) |

No frame mismatch was detected at 1e-5 m / 1e-4 rad tolerances. The measured
execution error is reproduced from actual arm joints while the fixed mount
contract remains correct; an extra fixed frame correction is not supported.

In this no-step diagnostic, raw USD transforms stayed at authored poses despite
the publication request. `usd_readback_mismatch_samples` reports this separately
from `frame_mismatch_samples`. It is not evidence of an IK frame error or proof
that the normal advancing replay viewport is stale. Use `world_runtime` for
same-time physical pose comparisons, and retain raw USD values for readback
diagnosis.

Verified artifacts:

- `outputs/diagnostics/wrist_frames_x050_20260907.jsonl`
- `outputs/diagnostics/wrist_frames_recorded_x050_verified_20260907.jsonl`

Existing lightweight regressions: 37/37 passed. The isolated Isaac snapshots
verify geometry and frame semantics, not closed-loop grasp reliability.

Next-stage measured rollout evidence: [command delivery and dynamic tracking
diagnosis](ARM_EXECUTION_DIAGNOSIS.md). That report separates actual residual
IK inputs, interpolated simulator targets, and post-physics base measurements.
