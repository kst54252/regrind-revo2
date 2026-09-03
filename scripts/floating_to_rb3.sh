#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: ./scripts/floating_to_rb3.sh --rollout FILE --object-start X Y Z [options]

Required:
  --rollout FILE             Floating Revo2 policy rollout from scripts/rl.sh play
  --object-start X Y Z       Desired tuna-can origin in RB3 world coordinates

Options:
  --out FILE                 Output final 12-DoF HDF5
  --object-quat X Y Z W      Desired initial can orientation (XYZW)
  --wrist-rpy R P Y          Mounted-wrist local correction in degrees

Additional arguments are forwarded to build_reference_trajectory.py.
EOF
}

rollout=""
output=""
object_start=()
object_quat=()
wrist_rpy=(0 0 0)
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rollout)
            [[ $# -ge 2 ]] || die "--rollout requires a file"
            rollout="$2"
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || die "--out requires a file"
            output="$2"
            shift 2
            ;;
        --object-start)
            [[ $# -ge 4 ]] || die "--object-start requires X Y Z"
            object_start=("$2" "$3" "$4")
            shift 4
            ;;
        --object-quat)
            [[ $# -ge 5 ]] || die "--object-quat requires X Y Z W"
            object_quat=("$2" "$3" "$4" "$5")
            shift 5
            ;;
        --wrist-rpy)
            [[ $# -ge 4 ]] || die "--wrist-rpy requires R P Y"
            wrist_rpy=("$2" "$3" "$4")
            shift 4
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            passthrough+=("$1")
            shift
            ;;
    esac
done

[[ -n "${rollout}" ]] || die "--rollout is required"
[[ ${#object_start[@]} -eq 3 ]] || die "--object-start X Y Z is required"
require_file "${rollout}" "floating rollout"
if [[ -z "${output}" ]]; then
    output="${rollout%.*}_rb3_revo2_reference.h5"
fi
setup_regrind_python

arguments=(
    "${PROJECT_ROOT}/tools/rb3_revo2_ik/build_reference_trajectory.py"
    "${rollout}"
    --out "${output}"
    --object-start-position "${object_start[@]}"
    --target-wrist-local-rpy-deg "${wrist_rpy[@]}"
)
if [[ ${#object_quat[@]} -eq 4 ]]; then
    arguments+=(--object-start-quat-xyzw "${object_quat[@]}")
fi
arguments+=("${passthrough[@]}")
exec "${ISAAC_PYTHON}" "${arguments[@]}"
