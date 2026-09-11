#!/bin/bash
set -e

DATA_DIR="./data"
OT_DISTANCE_DIR="./data/geoyfcc_text/distances/ot_distance/"
CHECKPOINT_ROOT="./results/subset/checkpoints"
LOG_ROOT="./results/subset/logs"

TARGET_DOMAINS=(57 12)
K_VALUES=(1 2 5)
BUDGET_VALUES=(2000 5000 10000)
SEEDS=(6651033 9272605 1206448 2180968 114325)
OT_EMBEDDING_TYPES=(bert geoclip satclip_L40 geodesic)
LOCATION_EMBEDDINGS=(geoclip satclip_L40 geodesic)  # '+bert' pair order must match 16_...sh (it's in the CSV filename)
LAMBDA=0.5

COMMON=(--dataset geoyfcc_text --domain_type countries --data_dir "$DATA_DIR"
        --checkpoint_root "$CHECKPOINT_ROOT" --log_root "$LOG_ROOT"
        --model bert_singlelabel --lr 2e-5 --num_epochs 50
        --optimizer AdamW --weight_decay 0.01 --scheduler cosine
        --train_batch_size 64 --eval_batch_size 512 --patience 10 --start_from_epoch 20)
OT_COMMON=(--ot_distance_dir "$OT_DISTANCE_DIR" --domain_selection_method ot
           --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000 --ot_metric cosine --ot_norm max_per_domain)

for tgt in "${TARGET_DOMAINS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for budget in "${BUDGET_VALUES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                RUN=(-s "$seed" --subset_size "$budget" --val_subset_size $((budget / 2)) --num_domains "$k" --tgt_domain "$tgt")

                python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" "${RUN[@]}" --domain_selection_method random

                for emb in "${OT_EMBEDDING_TYPES[@]}"; do
                    python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" "${RUN[@]}" "${OT_COMMON[@]}" --ot_embedding_type "$emb"
                done

                for emb in "${LOCATION_EMBEDDINGS[@]}"; do
                    python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" "${RUN[@]}" "${OT_COMMON[@]}" --ot_embedding_type "${emb}+bert" --ot_lambda "$LAMBDA"
                done
            done
        done
    done
done
