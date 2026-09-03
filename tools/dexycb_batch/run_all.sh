#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}"
TRIM_SEQUENCE="${REGRIND_TRIM_SEQUENCE:-20200709_143747_left}"
TRIM_LEADING_FRAMES="${REGRIND_TRIM_LEADING_FRAMES:-12}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 2
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/preprocess_dataset.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/trim_preprocessed_sequence.py" \
    "${PROJECT_ROOT}/outputs/preprocessed/dexycb/${TRIM_SEQUENCE}/dexycb_right_hand_preprocessed.npz" \
    --drop-first "${TRIM_LEADING_FRAMES}" \
    --summary "${PROJECT_ROOT}/outputs/preprocessed/dexycb/${TRIM_SEQUENCE}/preprocess_summary.json"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/build_html_gallery.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/retarget_all.py" --python "${PYTHON_BIN}" "$@"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dexycb_batch/prepare_isaac_references.py" --python "${PYTHON_BIN}" "$@"
