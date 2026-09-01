#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Isaac Sim Python not found: ${PYTHON_BIN}" >&2
    exit 2
fi

export REGRIND_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/regrind/source/regrind${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/regrind/scripts/rsl_rl/play.py" \
    --task Regrind-RB3-Revo2-TunaCan-Play-v0 \
    "$@"

