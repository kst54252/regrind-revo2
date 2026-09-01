#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}"

export REGRIND_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/regrind/source/regrind${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/regrind/scripts/validate_rb3_revo2_zero_residual_skeleton.py" \
    --viz kit \
    --max_visible_envs 1 \
    "$@"
