#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
setup_regrind_python
cd "${PROJECT_ROOT}"
exec "${ISAAC_PYTHON}" -m tools.rb3_revo2_ik.analyze_transfer_recovery "$@"
