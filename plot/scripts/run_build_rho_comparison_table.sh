#!/bin/bash
set -e

# Runs build_rho_comparison_table.py with its defaults (GeoYFCC-Text, all embeddings/distances).
#
# Usage:
#   ./plot/scripts/run_build_rho_comparison_table.sh [other build_rho_comparison_table.py args...]

python plot/build_rho_comparison_table.py "$@"
