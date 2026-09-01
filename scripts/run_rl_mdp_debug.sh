#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}"
DEFAULT_REFERENCE="${PROJECT_ROOT}/outputs/isaac/dexycb/20200709_143626_right/rb3_revo2_reference.h5"
DEFAULT_KEYPOINTS="${PROJECT_ROOT}/007_tuna_fish_can/object_points_50.npy"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Isaac Sim Python not found: ${PYTHON_BIN}" >&2
    exit 2
fi

export REGRIND_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/regrind/source/regrind${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/regrind/scripts/debug_rb3_revo2_mdp.py" \
    --reference "${DEFAULT_REFERENCE}" \
    --object-keypoints "${DEFAULT_KEYPOINTS}" \
    "$@"
