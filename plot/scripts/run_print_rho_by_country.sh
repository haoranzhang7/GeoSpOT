#!/bin/bash
set -e

# Runs print_rho_by_country.py, printing a given country's fixed-source-domain rho/R2 per
# embedding/distance combo plus the across-source-domain average for every combo, reading the
# CSVs built by run_build_rho_by_src_domain_csv.sh (build those first if they don't exist yet).
#
# Usage:
#   ./plot/scripts/run_print_rho_by_country.sh --country "United States"
#   ./plot/scripts/run_print_rho_by_country.sh --src_domain 57

python plot/print_rho_by_country.py "$@"
