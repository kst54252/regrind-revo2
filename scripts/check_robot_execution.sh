#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
setup_regrind_python
exec "${ISAAC_PYTHON}" -m tools.robot_execution.dry_run "$@"
