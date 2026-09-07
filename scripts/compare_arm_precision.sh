#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
cd "${PROJECT_ROOT}"
exec "${ISAAC_PYTHON}" -m tools.arm_diagnostics.compare_arm_precision "$@"
