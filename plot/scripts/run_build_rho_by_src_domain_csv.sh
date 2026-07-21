#!/bin/bash
set -e

# Runs build_rho_by_src_domain_csv.py with its defaults (GeoYFCC-Text, all embeddings/distances
# with a precomputed distance matrix), saving the mean/std Spearman's rho across fixed-source-
# domain regressions per (embedding, distance) to plot/plots/rho_avg_by_src_domain.csv.
#
# Usage:
#   ./plot/scripts/run_build_rho_by_src_domain_csv.sh [other build_rho_by_src_domain_csv.py args...]

python plot/build_rho_by_src_domain_csv.py "$@"
