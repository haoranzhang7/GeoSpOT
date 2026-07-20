#!/bin/bash
set -e

# Runs generate_domain_distance_pairs_csv.py against the current (euclidean) MMD-BERT
# distance matrix for GeoYFCC-Text.
#
# Usage:
#   ./plot/scripts/run_generate_domain_distance_pairs_csv.sh [other generate_domain_distance_pairs_csv.py args...]

python plot/generate_domain_distance_pairs_csv.py \
    --distance_file data/geoyfcc_text/distances/mmd_distance/mmd_bert_kmultiscale_meuclidean_nmax_per_domain_and_normalized_after.csv \
    "$@"
