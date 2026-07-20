#!/bin/bash
set -e

# Runs plot_method_vs_baselines.py once over all domains, and once restricted to China/US.
#
# Usage:
#   ./plot/scripts/run_plot_method_vs_baselines.sh [other plot_method_vs_baselines.py args...]

python plot/plot_method_vs_baselines.py --out plot/plots/subset_selection_method_vs_baselines.png "$@"
python plot/plot_method_vs_baselines.py --domains China US --out plot/plots/subset_selection_method_vs_baselines_china_us.png "$@"
