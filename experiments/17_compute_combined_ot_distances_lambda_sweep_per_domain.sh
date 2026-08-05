#!/bin/bash
set -e

# Per-domain-pair combined OT distances lambda * {geoclip,satclip,geodesic} + (1 - lambda) * bert,
# swept over lambda = 0.1 .. 0.9 in steps of 0.1. Same per-source-domain loop as
# 06_compute_ot_distances.sh (K=1, source-domain-idx looped 0..TOTAL_DOMAINS-1), unlike
# 16_compute_combined_ot_distances_lambda_sweep.sh's pooled source-domain-idx=all setup -- that
# one only fills the "all" row (see save_k1_matrix in ot_distance.py), which
# plot/trend_common.py's load_distance_matrix drops since it has no matching per-domain-pair
# target column, so it can't be used for the rho-vs-accuracy analysis in
# plot/build_overall_rho_csv.py / plot/build_rho_by_src_domain_csv.py. This script fills the full
# src x tgt matrix those need.
#
# Takes the location embedding and CUDA device as arguments so the three location types can be
# run in parallel on separate GPUs, e.g.:
#   bash experiments/17_compute_combined_ot_distances_lambda_sweep_per_domain.sh geoclip 0
#   bash experiments/17_compute_combined_ot_distances_lambda_sweep_per_domain.sh satclip 2
#   bash experiments/17_compute_combined_ot_distances_lambda_sweep_per_domain.sh geodesic 3

MODEL_TYPE="${1:?Usage: $0 <geoclip|satclip|geodesic> <cuda_device>}"
CUDA_DEVICE="${2:?Usage: $0 <geoclip|satclip|geodesic> <cuda_device>}"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

DATASET="geoyfcc_text"
TOTAL_DOMAINS=62
K=1
NORMALIZE_COST="max_per_domain"
LAMBDA_VALUES="0.5,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9"
SRC_DOMAIN_INDICES=$(seq -s, 0 $((TOTAL_DOMAINS - 1)))

# Lambda values and source domains are looped over inside a single process (via comma-separated
# lists), which loads the embeddings once and reuses them instead of reloading per combination.
python src/distances/ot_distance.py \
    --dataset-name $DATASET \
    --embedding-type "${MODEL_TYPE}+bert" \
    --lambda-values $LAMBDA_VALUES \
    --source-domain-idx $SRC_DOMAIN_INDICES \
    --total-domains $TOTAL_DOMAINS \
    --k $K \
    --metric cosine \
    --normalize-cost $NORMALIZE_COST
