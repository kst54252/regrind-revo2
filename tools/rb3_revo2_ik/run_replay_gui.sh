#!/usr/bin/env bash
set -euo pipefail

ISAAC_REPLAY_PYTHON="${ISAAC_SIM_PYTHON:-/home/wanjunkim/IsaacLab/.venv/bin/python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "${ISAAC_REPLAY_PYTHON}" ]]; then
    echo "Isaac Sim Python not found: ${ISAAC_REPLAY_PYTHON}" >&2
    echo "Set ISAAC_SIM_PYTHON to the Isaac Sim/Isaac Lab Python executable." >&2
    exit 2
fi

exec "${ISAAC_REPLAY_PYTHON}" "${SCRIPT_DIR}/launch_replay_gui.py" "$@"
