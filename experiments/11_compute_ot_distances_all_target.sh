#!/bin/bash
set -e

# Compute OT distances with the "target" being the pooled distribution across
# every domain (source-domain-idx=all), instead of one specific held-out domain.
# Used by 12_subset_selection_all_target.sh to pick training domains whose
# combined embeddings best match the overall/global data distribution.

export CUDA_VISIBLE_DEVICES=2

DATASET="geoyfcc_text"
TOTAL_DOMAINS=62
NORMALIZE_COST="max_per_domain"
K_VALUES=(1 2 5)

for model_type in bert geoclip satclip geodesic; do
    for k in "${K_VALUES[@]}"; do
        extra_args=()
        if [ "$k" -gt 1 ]; then
            extra_args+=(--greedy-sequential)
        fi
        python src/distances/ot_distance.py \
            --dataset-name $DATASET \
            --embedding-type $model_type \
            --source-domain-idx all \
            --total-domains $TOTAL_DOMAINS \
            --k $k \
            --metric cosine \
            --normalize-cost $NORMALIZE_COST \
            "${extra_args[@]}"
    done
done
