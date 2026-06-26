#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

DATASET="geoyfcc_text"
NORMALIZE_COST="max_per_domain_and_normalized_after"

for model_type in geoclip satclip; do #bert geoclip satclip; do
    for metric in cosine; do
        python src/distances/mmd_distance.py \
            --dataset $DATASET \
            --embedding-type $model_type \
            --metric $metric \
            --normalize-cost $NORMALIZE_COST
    done
done
