#!/bin/bash
set -e

# Runs build_rho_by_dataset_table.py with its defaults (pools src-domain regressions).
#
# Usage:
#   ./plot/scripts/run_build_rho_by_dataset_table.sh [other build_rho_by_dataset_table.py args...]

python plot/build_rho_by_dataset_table.py "$@"
