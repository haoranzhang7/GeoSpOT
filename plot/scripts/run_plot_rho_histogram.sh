#!/bin/bash
set -e

# Runs plot_rho_histogram.py: once comparing embeddings under OT distance, once comparing
# distance types under GeoCLIP, and once with every (embedding, distance) method overlaid
# on a single density plot -- as raw rho and again as |rho|.
#
# Usage:
#   ./plot/scripts/run_plot_rho_histogram.sh [other plot_rho_histogram.py args...]

python plot/plot_rho_histogram.py --embedding_types bert geoclip satclip --distance_types ot "$@"
python plot/plot_rho_histogram.py --embedding_types geoclip --distance_types ot mmd fid "$@"
python plot/plot_rho_histogram.py --embedding_types bert geoclip satclip --distance_types cosine fid geodesic mmd ot \
    --plot_type density --output_name rho_hist_all_methods "$@"
python plot/plot_rho_histogram.py --embedding_types bert geoclip satclip --distance_types cosine fid geodesic mmd ot \
    --plot_type density --output_name rho_hist_all_methods --abs_rho "$@"
