#!/bin/bash
set -e

# Runs plot_trends_side_by_side.py twice: once comparing GeoCLIP vs. SatCLIP OT-distance
# trends as two panels, and once comparing BERT, GeoCLIP, SatCLIP, and Geodesic distance
# trends as four panels, all sharing the y-axis.
#
# Usage:
#   ./plot/scripts/run_plot_trends_side_by_side.sh [other plot_trends_side_by_side.py args...]

DATASET="geoyfcc_text"
DIST_DIR="data/${DATASET}/distances/ot_distance"
SUFFIX="method_sinkhorn_log_reg_0.01_iter_1000"

BERT_FILE="${DIST_DIR}/ot_distance_matrix_bert_all_combinations_${SUFFIX}_metric_cosine_norm_max_per_domain.csv"
GEOCLIP_FILE="${DIST_DIR}/ot_distance_matrix_geoclip_all_combinations_${SUFFIX}_metric_cosine_norm_max_per_domain.csv"
SATCLIP_FILE="${DIST_DIR}/ot_distance_matrix_satclip_all_combinations_${SUFFIX}_metric_cosine_norm_max_per_domain.csv"
GEODESIC_FILE="${DIST_DIR}/ot_distance_matrix_geodesic_all_combinations_${SUFFIX}_metric_geodesic_norm_max_per_domain.csv"

# GeoCLIP vs. SatCLIP
python plot/plot_trends_side_by_side.py \
    --distance_files "$GEOCLIP_FILE" "$SATCLIP_FILE" \
    --embedding_types geoclip satclip \
    --distance_type ot \
    --results_file "results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv" \
    --output_dir "plot/plots/combined" \
    --rescale_acc "$@"

# BERT, GeoCLIP, SatCLIP, Geodesic
python plot/plot_trends_side_by_side.py \
    --distance_files "$BERT_FILE" "$GEOCLIP_FILE" "$SATCLIP_FILE" "$GEODESIC_FILE" \
    --embedding_types bert geoclip satclip geodesic \
    --distance_type ot \
    --results_file "results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv" \
    --output_dir "plot/plots/combined" \
    --rescale_acc --figsize 44 12 "$@"
