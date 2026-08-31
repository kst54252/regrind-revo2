#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_PYTHON:-${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Project Python not found: ${PYTHON_BIN}" >&2
    echo "Set PROJECT_PYTHON or ISAAC_SIM_PYTHON to a compatible Python executable." >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m unittest discover -s tests -p 'test_*.py' -v
