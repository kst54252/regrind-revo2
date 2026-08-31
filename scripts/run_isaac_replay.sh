#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${PROJECT_ROOT}/tools/rb3_revo2_ik/run_replay_gui.sh" "$@"
