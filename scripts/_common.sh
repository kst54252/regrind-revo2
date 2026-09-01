#!/usr/bin/env bash

# Shared launcher helpers. This file is sourced by scripts and is not intended
# to be executed directly.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "scripts/_common.sh must be sourced, not executed." >&2
    exit 2
fi

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DEFAULT_SEQUENCE="${REGRIND_SEQUENCE:-20200709_143747_left}"
readonly ISAAC_PYTHON="${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

setup_regrind_python() {
    [[ -x "${ISAAC_PYTHON}" ]] || die \
        "Isaac Sim Python not found: ${ISAAC_PYTHON} (set ISAAC_SIM_PYTHON)"
    export REGRIND_PROJECT_ROOT="${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}/regrind/source/regrind${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${PROJECT_ROOT}"
}

reference_for_sequence() {
    local sequence="$1"
    echo "${PROJECT_ROOT}/outputs/isaac/dexycb/${sequence}/rb3_revo2_reference.h5"
}

require_file() {
    local path="$1"
    local label="$2"
    [[ -f "${path}" ]] || die "${label} not found: ${path}"
}
