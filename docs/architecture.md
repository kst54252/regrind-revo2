# Repository architecture

This is the repository index. Read the section matching the task, then follow
one deeper document if needed:

- [Current implementation status](current-status.md)
- [Data pipeline and coordinate frames](DATA_PIPELINE.md)
- [Floating Revo2 RL task](RL_TASK.md)
- [Isaac replay](ISAAC_SIM_REPLAY.md)
- [Execution/diagnostic command index](../scripts/README.md), for arm comparisons
- [Presentation media commands](../scripts/README.md#presentation-media), for captures
- [Cleanup audit record](cleanup-plan.md), for cleanup work only

## System shape

The primary design separates floating-hand policy training from arm deployment:

```text
DexYCB -> Revo2 retargeting -> world alignment -> RB3 strict IK -> replay
                                      |
                                      +-> floating Revo2 residual PPO
                                            -> online RB3 IK deployment
```

`regrind/` is upstream-derived code inside this Git repository, not a submodule
or nested repository.

## Repository map

| Path | Responsibility |
|---|---|
| `scripts/` | Maintained launchers and shared shell setup |
| `tools/dexycb_batch/` | Multi-sequence preprocessing and orchestration |
| `tools/dexycb_world_transform/` | Camera-to-world transforms and viewers |
| `tools/revo2_kinematics/` | Isaac-independent Revo2 FK and 21 keypoints |
| `tools/rb3_revo2_ik/` | RB3 FK/IK, trajectory tools, diagnostics, replay |
| `tools/arm_diagnostics/` | Offline trace analysis and paired-result comparisons; no simulator launch |
| `regrind/scripts/` | Retargeting and Isaac Lab/RSL-RL Python entry points |
| `regrind/source/regrind/regrind/` | Installable package, tasks, assets, and MDP terms |
| `config/workcell/` | Shared table, pedestal, and mount geometry |
| `tests/` | Main simulator-independent tests |
| `regrind/source/regrind/test/` | Extra package tests outside the main test wrapper |
| `dataset/` | Read-only raw DexYCB input |
| `USD/`, `007_tuna_fish_can/` | Robot/workcell and tuna-can assets |
| `outputs/`, `logs/` | Mixed generated artifacts and experiment runs |

Diagnostic JSON/JSONL under `outputs/diagnostics/` stay local (the small
`arm_transfer_recovery/heldout_initial_states_v2.jsonl` launcher input is tracked).
Historical diagnostic commands may require separately copied local records;
a fresh clone does not contain those traces. Config/model/keypoint JSON and
pipeline manifests are not covered by this ignore rule.

## Maintained entry points

| Command | Purpose |
|---|---|
| `./scripts/run_pipeline.sh` | Preprocess, retarget, align, and build IK references |
| `./scripts/run_isaac_replay.sh` | Open the RB3+Revo2 Isaac replay |
| `./scripts/rl.sh train` | Train floating Revo2 PPO by default |
| `./scripts/rl.sh play` | Evaluate/export a floating policy rollout |
| `./scripts/rl.sh play-arm` | Deploy the floating policy through online RB3 IK |
| `./scripts/rl.sh zero` / `debug` | Reference-only or observation/reward validation |
| `./scripts/floating_to_rb3.sh` | Convert a floating rollout through offline strict IK |
| `./scripts/random_can_full_replay.sh` | Random placement, policy, IK, and workcell replay |
| `./scripts/run_tests.sh` | Shell checks and root `tests/` discovery |

Old duplicate train/play/zero/debug aliases were removed. Use `scripts/rl.sh`;
the [migration table](cleanup-plan.md#readable-layout-cleanup) records replacements.

Opt-in experiments are **not** replacements for `rl.sh play-arm`:
`evaluate_mounted_interface.sh` dispatches floating/legacy/simple modes;
`arm_transfer_recovery.sh` adds recovery capture to that evaluator;
`play_arm_candidate.sh` explicitly selects the candidate in
`config/experiments/rb3_transfer_recovery_candidate.json` and fast IK. Start from
the [diagnostic index](../scripts/README.md#experiments-and-failure-reproduction-opt-in),
not every historical report. The minimal adapter is
`mdp/simple_mounted_interface.py` with `tools/rb3_revo2_ik/frozen_policy_adapter.py`;
it reuses the policy contract and FK/IK, but preserves a separate execution path.

## Data flow

```text
dataset/<sequence>/
  -> outputs/preprocessed/dexycb/<sequence>/*.npz
  -> outputs/retargeted/dexycb/<sequence>/*.h5
  -> outputs/isaac/dexycb/<sequence>/world_trajectory.h5
  -> outputs/isaac/dexycb/<sequence>/rb3_revo2_reference.h5
```

The main implementations are:

1. `tools/dexycb_batch/preprocess_dataset.py`
2. `tools/dexycb_batch/retarget_all.py` ->
   `regrind/scripts/retarget_hand_object.py`
3. `tools/dexycb_world_transform/transform_trajectory.py`
4. `tools/rb3_revo2_ik/build_reference_trajectory.py`

See [DATA_PIPELINE.md](DATA_PIPELINE.md) for arrays and coordinate conventions.

## Subsystems and dependencies

| Subsystem | Important implementation | Depends on |
|---|---|---|
| Retargeting | `regrind/source/regrind/regrind/retargeting/` | NumPy/SciPy/HDF5, pydrake, robot constants |
| Revo2 FK | `tools/revo2_kinematics/revo2_kinematics.py` | model JSON and semantic keypoint JSON |
| RB3 FK/IK | `tools/rb3_revo2_ik/rb3_kinematics.py` | SciPy and `rb3_model.json` |
| Reference loading | `tools/rb3_revo2_ik/reference_trajectory.py` | HDF5/NPZ conventions |
| Isaac replay | `tools/rb3_revo2_ik/replay_reference_isaac_sim.py` | Isaac Sim, USD assets, workcell config |
| Floating RL | `.../config/revo2_floating/` and `.../mdp/` | Isaac Lab, RSL-RL, reference loader |
| Online arm deployment | `.../config/rb3_revo2/` and `mdp/rb3_revo2_actions.py` | floating policy contract and bounded RB3 IK |

In the last two rows, `...` means
`regrind/source/regrind/regrind/tasks/manager_based/dexterous/`.

The primary floating task controls a six-dimensional wrist residual and six
Revo2 leaders. Online deployment preserves that policy contract and converts
the wrist command to RB3 joints. LeapHand/WujiHand and combined-arm task
registrations remain secondary compatibility paths.

## Boundary contracts

- Revo2 has six independent joints; five distal joints are deterministic mimic
  followers. The 21 semantic points are FK outputs, never optimization DoFs.
- Final joint arrays are RB3 six joints followed by Revo2 six leaders.
- Quaternion order is metadata-driven: preprocessed files use `wxyz`, retargeted
  files use `xyzw`, world files use `wxyz`, and final RB3+Revo2 references use `xyzw`.
- One rigid camera-to-world transform is applied to object, wrist, and MANO.
- MANO21 sequential topology differs from Revo2 semantic correspondence order;
  use the [stage-specific field/order contract](DATA_PIPELINE.md#mano와-revo2-topology).
- The workcell uses tabletop `Z=0`; replay and RL share the same workcell file.
- Pure-Python tests cannot establish Isaac/PhysX contact behavior.

## Important configuration

| Concern | Location |
|---|---|
| Workcell geometry and mount | `config/workcell/rb3_revo2_table.json` |
| RB3 chain and mounted wrist | `tools/rb3_revo2_ik/rb3_model.json` |
| Per-sequence alignment | `tools/dexycb_batch/prepare_isaac_references.py::ISAAC_ALIGNMENT` |
| Default sequence/reference | `scripts/_common.sh` |
| Revo2 constants/keypoints | `regrind/source/regrind/regrind/retargeting/revo2_constants.py` |
| Robot and object assets | `regrind/source/regrind/regrind/assets/` |
| RL observations/rewards/RSI | `.../dexterous/mdp/` and task environment configs |
| Floating PPO | `.../config/revo2_floating/agents/rsl_rl_ppo_cfg.py` |

Consult [current-status.md](current-status.md) before assuming that a preserved
compatibility path has been recently exercised.
