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
