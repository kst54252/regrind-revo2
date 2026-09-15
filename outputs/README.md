# Generated results and reproduction records

This is not the maintained application source tree. Start runs from
[`scripts/README.md`](../scripts/README.md); implementations live in `tools/`
and `regrind/`. Dated capture scripts here reproduce particular figures/videos;
they are retained evidence, not alternate training or deployment entry points.

| Directory | Role | Retention |
|---|---|---|
| `preprocessed/`, `retargeted/`, `isaac/`, `floating/` | Pipeline/reference/rollout data | Some are tracked; preserve inputs selected by launchers/checkpoints. |
| `diagnostics/` | Paired experiment traces, solver arrays, summaries and plots | Compact CSV/PNG may be tracked. Raw JSON/JSONL/NPZ and `preserved_before/` snapshots stay local. |
| `visualizations/` | Presentation figures, videos, captures and study exports | Small figures, capture metadata and reproduction instructions may be tracked. HTML/video and raw policy/physics/timing traces stay local. |

The small `diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl`
bank is explicitly tracked because a maintained launcher uses it. Do not replace
that exception with a blanket JSON ignore. Configuration, assets, keypoint JSON
and dataset manifests are outside the diagnostic ignore rules.

**Ignored does not mean disposable.** Historical analyzers and `--match-recording`
can require ignored traces/arrays. Copy the selected run's metadata, initial-state
bank, traces, checkpoint and reference separately when reproducing on another PC;
a Git clone alone is insufficient. Check `git ls-files -- PATH` and
`git check-ignore -v PATH` before assuming an artifact is backed up.

GitHub marks `outputs/` as generated evidence, keeping it out of source-language
statistics. Diagnostic CSV files retain their writer's CRLF endings and exact
values/hashes; the scoped whitespace rule accepts CRLF without suppressing other
whitespace checks or changing source-file rules.

Keep each new comparison in a new run directory. Preserve unique failure records;
do not delete a run based only on its age or an empty `policy.json`. This cleanup
does not rewrite existing Git history or move any runtime asset/reference path.
