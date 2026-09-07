---
name: validate-regrind-change
description: Validate a completed repository implementation or refactor with existing tests, maintained launchers, boundary checks, and a focused diff review. Use after changes or when asked whether work is ready; it is not a debugging workflow for an unexplained failure.
---

# Validate a REGRIND change

1. Inspect status and the focused diff, separating pre-existing work. Use the
   architecture map only if the affected callers, configs, or tests are unclear.
2. Search affected symbols and call sites. Select the smallest existing test or
   maintained script that exercises the changed contract; verify all flags in
   repository files.
3. Check the applicable boundary contracts in `docs/architecture.md`, then run
   targeted checks. Run `./scripts/run_tests.sh` when dependencies allow.
4. Require the documented Isaac validation for Isaac/PhysX behavior; pure-Python
   tests are insufficient. Do not regenerate unrelated artifacts for validation.
5. Finish with `git diff --check`, focused diff review, and status inspection.
   Report executed and skipped checks, remaining risk, and a pass/fail judgment.
