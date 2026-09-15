#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
cd "${PROJECT_ROOT}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "${ISAAC_PYTHON}" -m tools.arm_diagnostics.compare_singularity_methods "$@"
