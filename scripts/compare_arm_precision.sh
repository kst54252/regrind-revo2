#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
exec "${ISAAC_PYTHON}" "${PROJECT_ROOT}/tools/rb3_revo2_ik/compare_arm_precision.py" "$@"
