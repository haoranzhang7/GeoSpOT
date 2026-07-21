#!/bin/bash
set -e

# Runs build_overall_rho_csv.py with its defaults (GeoYFCC-Text, all embeddings/distances with
# a precomputed distance matrix), saving one overall Spearman's rho per (embedding, distance)
# to plot/plots/rho_overall.csv.
#
# Usage:
#   ./plot/scripts/run_build_overall_rho_csv.sh [other build_overall_rho_csv.py args...]

python plot/build_overall_rho_csv.py "$@"
