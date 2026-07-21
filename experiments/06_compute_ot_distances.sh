#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

DATASET="geoyfcc_text"
TOTAL_DOMAINS=62
K=1
NORMALIZE_COST="max_per_domain"

for model_type in satclip geodesic bert geoclip; do
    for src_domain_idx in $(seq 0 $((TOTAL_DOMAINS - 1))); do
        for metric in cosine; do
            python src/distances/ot_distance.py \
                --dataset-name $DATASET \
                --embedding-type $model_type \
                --source-domain-idx $src_domain_idx \
                --total-domains $TOTAL_DOMAINS \
                --k $K \
                --metric $metric \
                --normalize-cost $NORMALIZE_COST
        done
    done
done