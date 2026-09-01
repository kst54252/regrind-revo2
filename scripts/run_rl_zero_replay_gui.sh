#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${PROJECT_ROOT}/scripts/run_rl_zero_replay.sh" \
    --viz kit \
    --max_visible_envs 1 \
    "$@"
