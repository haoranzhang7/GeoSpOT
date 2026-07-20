#!/bin/bash
set -e

# Runs plot_trends_side_by_side.py to compare GeoCLIP vs. SatCLIP OT-distance trends
# as two panels in one figure.
#
# Usage:
#   ./plot/scripts/run_plot_trends_side_by_side.sh [other plot_trends_side_by_side.py args...]

DATASET="geoyfcc_text"

python plot/plot_trends_side_by_side.py \
    --distance_files \
        "data/${DATASET}/distances/ot_distance/ot_distance_geoclip_max_per_domain.csv" \
        "data/${DATASET}/distances/ot_distance/ot_distance_satclip_L40_max_per_domain.csv" \
    --embedding_types geoclip satclip \
    --distance_type ot \
    --results_file "results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv" \
    --output_dir "plot/plots/combined" \
    --rescale_acc "$@"
