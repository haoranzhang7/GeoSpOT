#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

DATASET="geoyfcc_text"
NORMALIZE_COST="max_per_domain"

for model_type in bert geoclip satclip; do
    for metric in cosine euclidean; do
        python src/distances/mmd_distance.py \
            --dataset $DATASET \
            --embedding-type $model_type \
            --metric $metric \
            --normalize-cost $NORMALIZE_COST
    done
done

# geodesic needs its own embedding-type (2D lat/lon coords), can't pair with bert/geoclip/satclip embeddings
python src/distances/mmd_distance.py \
    --dataset $DATASET \
    --embedding-type geodesic \
    --metric geodesic \
    --normalize-cost $NORMALIZE_COST
