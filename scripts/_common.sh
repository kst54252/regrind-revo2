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
# Pin user-facing floating-policy evaluation; never infer a model from capture runs.
readonly DEFAULT_FLOATING_CHECKPOINT="${REGRIND_FLOATING_CHECKPOINT:-${PROJECT_ROOT}/logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt}"

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

training_reference_for_sequence() {
    local sequence="$1"
    local stable_reference="${PROJECT_ROOT}/outputs/isaac/dexycb/${sequence}/rb3_revo2_reference_stable.h5"
    if [[ -f "${stable_reference}" ]]; then
        echo "${stable_reference}"
    else
        reference_for_sequence "${sequence}"
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    [[ -f "${path}" ]] || die "${label} not found: ${path}"
}

has_policy_selection() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --checkpoint|--checkpoint=*|--load_run|--load_run=*|--task|--task=*|--agent|--agent=*|agent.load_run=*|agent.load_checkpoint=*|agent.experiment_name=*)
                return 0 ;;
        esac
    done
    return 1
}
