#!/bin/bash
set -e

EMBEDDING_DIR="./data/embeddings"
RESULT_DIR="./data/geoyfcc/distances/ot_distance"

for model_type in bert geoclip satclip_L10 satclip_L40; do
    python src/data/distances/pairwise_distance.py \
        --embedding_dir $EMBEDDING_DIR \
        --result_dir $RESULT_DIR \
        --model_type $model_type
done
