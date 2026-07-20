#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

DATASET="geoyfcc_text"
NORMALIZE_COST="none"

for model_type in bert geoclip satclip; do
    for metric in euclidean; do
        python src/distances/fid_distance.py \
            --dataset $DATASET \
            --embedding-type $model_type \
            --metric $metric \
            --normalize-cost $NORMALIZE_COST
    done
done
