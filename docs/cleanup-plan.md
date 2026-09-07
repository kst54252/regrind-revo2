# Cleanup audit record

> On-demand document: read this only for cleanup, retention, or repository
> archaeology. Start normal work from [architecture.md](architecture.md).

Audited 2026-09-05 and reviewed 2026-09-06. References were checked through
imports, calls, registrations, configs, shell entry points, docs, tests, Git
state/history, string paths, and targeted USD/URDF attributes. Dataset,
checkpoint, mesh, and trajectory payloads were not broadly loaded.

## Completed SAFE cleanup

| Item | Result |
|---|---|
| `.vscode/browse.vc.db*` | Removed reproducible IDE database files. |
| Python/pytest caches | Removed audited caches outside excluded artifact trees. |
| `regrind/source/regrind/regrind.egg-info/` | Removed generated install metadata. |
| Four unused tuna derivatives | Removed ignored USD/SDF/PTH derivatives; active simple assets remain. |
| `007_tuna_fish_can/object_points_50.png` | Removed generated preview and ignored its exact path; runtime NPY remains. |
| Visualization HTML | Restored after finding tracked manifests with direct page paths. |
| Empty `.codex/` | Skipped because the environment exposes it as a busy managed mount. |

At execution time, 37 lightweight tests passed, maintained entry/config/asset
paths resolved, manifests resolved to restored HTML, and regenerated caches were
removed. This is historical evidence, not a claim about the current tree.

## Reviewed candidates

| Candidate | Decision | Evidence or prerequisite |
|---|---|---|
| Visualization HTML and manifests | **KEEP** | Manifests directly record all ten pages; change retention semantics as one unit. |
| Revo2 mail patch | **STILL UNCERTAIN** | Reverse-applies and has no caller, but its source commit is absent and it is the only local mail-patch provenance. |
| `007_tuna_fish_can/points.xyz` | **KEEP** | No consumer found; original asset provenance is unclear and saving 74 KB is negligible. |
| Full tuna XML/OBJ/MTL chain | **STILL UNCERTAIN** | No active internal consumer beyond the XML, but external/source-geometry use is unknown. |
| Upstream convenience scripts | **KEEP** | `regrind/README.md` and VS Code launch templates expose them. |
| Standalone world trim/view tools | **KEEP** | Distinct recent operator CLIs; lack of an internal caller is insufficient. |
| `regrind/source/.vscode/` | **KEEP** | Intentional ROS 2 contributor settings with negligible storage cost. |
| Historical LeapHand/combined-arm logs | **STILL UNCERTAIN** | RSL-RL can select runs by directory; no retention or best-model decision exists. |
| Floating intermediate checkpoints | **KEEP** | Support resume/comparison; maintained scripts also select explicit checkpoints. |
| Root/tool README corrections | **COMPLETED** | Structure/output policy and missing example paths were corrected. |
| Preprocess camera help | **COMPLETED** | Help now matches implemented second-camera selection. |
| Adapter converter YAML and hash | **KEEP** | Preserve regeneration recipe/cache identity until portability requirements are known. |

## RISKY: keep unless separately scoped

| Area | Why it is protected |
|---|---|
| Tracked `outputs/` and selected checkpoints/logs | Active inputs, launcher defaults, validation evidence, and resume history are mixed with generated data. |
| Duplicate Revo2 keypoint and tuna keypoint files | Standalone and installed-package paths intentionally consume different copies. |
| Duplicate textures and Wuji visual/collision meshes | OBJ, USD, MJCF, and URDF references require separate deployment paths. |
| LeapHand/WujiHand and `--legacy-arm-rl` | Still registered and part of upstream compatibility. |
| Root `USD/` graph and vertical adapter sources | Payload/asset references and FK mount assumptions are runtime-critical. |
| Compatibility wrappers and manual diagnostics | Existing user commands and operator workflows may call them externally. |
| Dataset tail-trimming tool | Destructive standalone maintenance workflow; absence of callers does not imply dead code. |
| Mirroring/MANO topology duplicates | Consolidation can change handedness or visualization semantics and needs tests. |
| Root versus package test directories | Test authority and dependency boundaries are not yet defined. |
| Root/package/tool/log directory layout | Moving it requires widespread path and external-command updates with little benefit. |
| Dated checkpoint and actuator defaults | These change evaluation behavior, not repository tidiness. |
| Machine-specific Isaac/keypoint paths | Overrides and external source-of-truth assumptions need a portability decision. |

## Prerequisites for later cleanup

1. Define tracked-output, HTML-manifest, log, and checkpoint retention policies.
2. Choose archive locations for the mail patch and original tuna source assets.
3. Decide whether root or package tests are authoritative before changing layout.
4. Define supported machines and asset-regeneration workflow before normalizing
   absolute paths.
5. Keep source/tool removal, asset migration, and experiment pruning as separate
   reviewable changes.

## Supported-path cleanup 2026-09-07

Branch: `codex/cleanup-supported-paths-20260907`, forked from `84d0423`.
The tree already contained modified source/docs, untracked experiments, ignored
data/logs, and the deleted preview listed above. These are not cleanup changes.
Only the three removal targets and documents being edited were snapshotted at
their current contents in commit `34c896d`; no checkpoint, log, data, asset or
unrelated source was staged. That snapshot includes pre-existing documentation
edits, deliberately separated from the cleanup commits.

### Decisions and reference evidence

| Current path / family | Decision / action | Reason, checks and expected impact |
|---|---|---|
| `scripts/screen_arm_transfer_recovery.sh` | **SAFE TO CHANGE — removed** | Fixed completed eight-condition launcher; no unique implementation. All eight metadata files exist. Only report references found; recipe retained in ARM_TRANSFER_RECOVERY. |
| `scripts/screen_arm_response_recovery.sh` | **SAFE TO CHANGE — removed** | Fixed completed two-candidate launcher; both results preserved. Same evaluator and no unique tests or settings implementation. |
| `scripts/finish_arm_transfer_recovery.sh` | **SAFE TO CHANGE — removed** | Fixed completed seven-run orchestration. Every result exists; old “resume” text was stale. General evaluator and explicit reproduction commands remain. |
| Six `train_rb3_*` / `play_rb3_*` / `run_rl_*` compatibility aliases | **KEEP** | Deliberately documented external CLI compatibility, not dead code. Delegate to `rl.sh`; negligible duplication. |
| `tools/dexycb_batch/run_all.sh`, `run_preprocess_all.sh`; `tools/rb3_revo2_ik/run_replay_gui.sh` | **KEEP** | Pipeline substeps, preprocessing-only CLI and GUI bootstrap have distinct behavior/direct callers. Root scripts remain the recommended entry points. |
| Minimal bridge, actual/recovery replay, precision and warm-start experiments; their tests/configs | **KEEP** | Actual imports/CLI flags, rejection criteria and comparison reports depend on them. Separate baseline/candidate navigation instead of deleting failed experiments. |
| Manual upstream scripts, standalone trim/asset generators, package tests | **DEFER** | External/manual use and unique dependency boundaries remain uncertain; no proof of redundancy. |
| Duplicate keypoints, model/asset copies, local settings, datasets, checkpoints, experiment logs/videos | **PROTECTED / DEFER** | Runtime or provenance/retention dependencies; no deletion or movement this round. |

For all three removed launchers: full shell contents inspected; imports/calls,
string paths, task/config registration, tests, shell entry points, documentation,
`.agents` and VS Code references searched. No runtime caller or asset reference
was found. They were independently executable, but specifically encoded completed
bounded sweeps with occupied output directories. This completion evidence, the
retained shared evaluator and Git snapshot—not absence of search hits—justify
removal. No USD/URDF/mesh/texture or model path changes are involved.

Recover one exact retired file without altering the worktree:

```bash
git show 34c896d:scripts/screen_arm_transfer_recovery.sh
```

No tests consolidated or removed: mount/FK/IK/order, mimic, quaternions/units,
observation/action parity, command-versus-state, timing/reset and paired-state
comparisons remain under `tests/`. No `.gitignore` changes in this round.

### Validation record

Artifacts: `outputs/diagnostics/cleanup_20260907/`. Baseline status and tracked
diff are saved there, separately from prior experiment evidence. Pre-cleanup:
98 tests passed; full floating and legacy-arm 20-placement evaluations completed
using the same saved original state bank, checkpoint and reference. Post-cleanup
checks and exact trace comparisons are recorded below after execution.
