#!/bin/bash
set -e

# Paths
DATA_DIR="./data"
OT_DISTANCE_DIR="./data/geoyfcc/distances/ot_distance/"
CHECKPOINT_ROOT="./results/subset/checkpoints"
LOG_ROOT="./results/subset/logs"
RESULTS_ROOT="./results/subset/test_results"

# Dataset/model
DATASET="geoyfcc_text"
DOMAIN_TYPE="countries"
MODEL="bert_singlelabel"
LR=2e-5
NUM_EPOCHS=50
OPTIMIZER="AdamW"
WEIGHT_DECAY=0.01
SCHEDULER="cosine"
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=512
PATIENCE=10
START_FROM_EPOCH=20

# Experiment grid
TARGET_DOMAINS=(57 12 34)
K_VALUES=(1 2 5)
BUDGET_VALUES=(2000 5000 10000)
SEEDS=(6651033 9272605 1206448 2180968 114325)
OT_EMBEDDING_TYPES=("bert" "geoclip" "satclip_L40" "geodesic")

# Val budget = budget / 2
declare -A VAL_BUDGET
VAL_BUDGET[2000]=1000
VAL_BUDGET[5000]=2500
VAL_BUDGET[10000]=5000

for tgt in "${TARGET_DOMAINS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for budget in "${BUDGET_VALUES[@]}"; do
            val_budget=${VAL_BUDGET[$budget]}
            for seed in "${SEEDS[@]}"; do

                # OT-based selection
                for emb in "${OT_EMBEDDING_TYPES[@]}"; do
                    python src/training/pretrain_by_domain_subset.py \
                        --dataset $DATASET --domain_type $DOMAIN_TYPE \
                        --data_dir $DATA_DIR --ot_distance_dir $OT_DISTANCE_DIR \
                        --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT \
                        --model $MODEL --lr $LR --num_epochs $NUM_EPOCHS \
                        --optimizer $OPTIMIZER --weight_decay $WEIGHT_DECAY --scheduler $SCHEDULER \
                        --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE \
                        --patience $PATIENCE --start_from_epoch $START_FROM_EPOCH \
                        -s $seed --subset_size $budget --val_subset_size $val_budget \
                        --num_domains $k --domain_selection_method ot \
                        --tgt_domain $tgt --ot_embedding_type $emb \
                        --ot_method sinkhorn --ot_reg 0.01 --ot_iter 1000 \
                        --ot_metric cosine --ot_norm max_per_domain
                done

                # Global and random selection
                for method in global random; do
                    python src/training/pretrain_by_domain_subset.py \
                        --dataset $DATASET --domain_type $DOMAIN_TYPE \
                        --data_dir $DATA_DIR \
                        --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT \
                        --model $MODEL --lr $LR --num_epochs $NUM_EPOCHS \
                        --optimizer $OPTIMIZER --weight_decay $WEIGHT_DECAY --scheduler $SCHEDULER \
                        --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE \
                        --patience $PATIENCE --start_from_epoch $START_FROM_EPOCH \
                        -s $seed --subset_size $budget --val_subset_size $val_budget \
                        --num_domains $k --domain_selection_method $method \
                        --tgt_domain $tgt
                done

            done
        done
    done
done
