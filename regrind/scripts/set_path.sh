#!/bin/bash

REGRIND_ROOT=$PWD

# Set default paths
DEFAULT_DATA_DIR="${PWD}/data"

# Prompt the user for input, allowing overrides
read -p "Enter the desired data directory [default: ${DEFAULT_DATA_DIR}], leave empty to use default: " DATA_DIR
REGRIND_DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

# Export to current session
export REGRIND_ROOT="$REGRIND_ROOT"
export REGRIND_DATA_DIR="$REGRIND_DATA_DIR"

# Confirm the paths with the user
echo "REGRIND root directory set to: REGRIND_ROOT=${REGRIND_ROOT}"
echo "Data directory set to: REGRIND_DATA_DIR=${REGRIND_DATA_DIR}"
