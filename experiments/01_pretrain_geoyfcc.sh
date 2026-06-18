#!/bin/bash
set -e

# Paths
DATA_DIR="./data"
CHECKPOINT_ROOT="./results/pretrain/checkpoints"
LOG_ROOT="./results/pretrain/logs"

# Dataset
DATASET="geoyfcc_text"
DOMAIN_TYPE="countries"
NUM_DOMAINS=62

# Model
MODEL="bert_singlelabel"
LR=2e-5
NUM_EPOCHS=50
OPTIMIZER="AdamW"
WEIGHT_DECAY=0.01
SCHEDULER="cosine"

# Training
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=512
PATIENCE=10
START_FROM_EPOCH=0

SEEDS=(48329 17046 62984 31507 90861)

for domain in $(seq 0 $((NUM_DOMAINS - 1))); do
    for seed in "${SEEDS[@]}"; do
        python src/training/pretrain_by_domain.py \
            -d $domain -s $seed \
            --dataset $DATASET --domain_type $DOMAIN_TYPE \
            --data_dir $DATA_DIR \
            --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT \
            --model $MODEL --lr $LR --num_epochs $NUM_EPOCHS \
            --optimizer $OPTIMIZER --weight_decay $WEIGHT_DECAY --scheduler $SCHEDULER \
            --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE \
            --patience $PATIENCE --start_from_epoch $START_FROM_EPOCH
    done
done
