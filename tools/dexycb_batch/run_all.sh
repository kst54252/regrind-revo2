#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 2
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/preprocess_dataset.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/build_html_gallery.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/retarget_all.py" --python "${PYTHON_BIN}" "$@"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/prepare_isaac_references.py" --python "${PYTHON_BIN}" "$@"
