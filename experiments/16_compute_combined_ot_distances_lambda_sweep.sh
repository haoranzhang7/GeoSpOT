#!/bin/bash
set -e

# Combined OT distances lambda * {geoclip,satclip,geodesic} + (1 - lambda) * bert,
# swept over lambda = 0.0 .. 1.0 in steps of 0.1. Same "all target" setup as
# 11_compute_ot_distances_all_target.sh (source-domain-idx=all), so this
# requires the embeddings/coordinates used there to already be in place.

export CUDA_VISIBLE_DEVICES=3

DATASET="geoyfcc_text"
TOTAL_DOMAINS=62
NORMALIZE_COST="max_per_domain"
K_VALUES=(1 2 5)
LAMBDA_VALUES="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"

# Lambda values are looped over inside a single process (via --lambda-values), which loads the
# embeddings once per (model_type, k) and reuses them instead of reloading per lambda.
for model_type in geoclip satclip geodesic; do
    for k in "${K_VALUES[@]}"; do
        extra_args=()
        if [ "$k" -gt 1 ]; then
            extra_args+=(--greedy-sequential)
        fi
        python src/distances/ot_distance.py \
            --dataset-name $DATASET \
            --embedding-type "${model_type}+bert" \
            --lambda-values $LAMBDA_VALUES \
            --source-domain-idx all \
            --total-domains $TOTAL_DOMAINS \
            --k $k \
            --metric cosine \
            --normalize-cost $NORMALIZE_COST \
            "${extra_args[@]}"
    done
done
