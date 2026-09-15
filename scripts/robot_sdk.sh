#!/usr/bin/env bash
# Device SDKs stay isolated from floating/Isaac training dependencies.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ENV="${PROJECT_ROOT}/tools/robot_execution/.venv"
cd "${PROJECT_ROOT}"
case "${1:-doctor}" in
    install)
        if [[ ! -x "${SDK_ENV}/bin/python" ]]; then
            "${ROBOT_BOOTSTRAP_PYTHON:-python3}" -m venv "${SDK_ENV}"
        fi
        exec "${SDK_ENV}/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: \
            -r tools/robot_execution/requirements-sdk.txt
        ;;
    doctor|probe)
        [[ -x "${SDK_ENV}/bin/python" ]] || { echo 'Run bash scripts/robot_sdk.sh install first.' >&2; exit 2; }
        exec "${SDK_ENV}/bin/python" -m tools.robot_execution.sdk_probe "$@"
        ;;
    *)
        echo 'Usage: bash scripts/robot_sdk.sh install|doctor|probe [--config PATH --device rb3|revo2|all --output NEW.json]' >&2
        exit 2
        ;;
esac
