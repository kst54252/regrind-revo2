#!/usr/bin/env bash
# Recorded measured motion only: policy is neither loaded nor called.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/evaluate_mounted_interface.sh" --mode simple --stage actual "$@"
