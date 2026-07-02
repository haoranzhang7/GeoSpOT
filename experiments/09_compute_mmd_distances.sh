#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

DATASET="geoyfcc_text"
NORMALIZE_COST="max_per_domain_and_normalized_after"

for model_type in bert geoclip satclip; do
    for metric in euclidean; do #in geodesic; do #cosine; do
        python src/distances/mmd_distance.py \
            --dataset $DATASET \
            --embedding-type $model_type \
            --metric $metric \
            --normalize-cost $NORMALIZE_COST
    done
done
