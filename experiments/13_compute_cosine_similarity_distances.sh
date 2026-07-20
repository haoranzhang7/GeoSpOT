#!/bin/bash
set -e

DATASET="geoyfcc_text"

for model_type in bert geoclip satclip; do
    python src/distances/cosine_similarity_distance.py \
        --dataset $DATASET \
        --embedding-type $model_type
done
