#!/bin/bash
set -e

DATA_DIR="./data"
CHECKPOINT_ROOT="./results/fewshot/checkpoints"
LOG_ROOT="./results/fewshot/logs"
DATASET="geoyfcc_text"
DOMAIN_TYPE="countries"
MODEL="bert_singlelabel"
LR=1e-5
NUM_EPOCHS=10
OPTIMIZER="AdamW"
WEIGHT_DECAY=0.01
SCHEDULER="cosine"
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=512
PATIENCE=3
START_FROM_EPOCH=0
NUM_DOMAINS=62
SEEDS="48329 17046 62984 31507 90861"
FINETUNE_SIZES=(1 2 5)

for pretrain_domain in $(seq 0 $((NUM_DOMAINS - 1))); do
    for finetune_domain in $(seq 0 $((NUM_DOMAINS - 1))); do
        for n in "${FINETUNE_SIZES[@]}"; do
            python src/evaluation/fewshot/fewshot_transfer.py \
                -d $pretrain_domain -f $finetune_domain -n $n \
                --dataset $DATASET --domain_type $DOMAIN_TYPE \
                --data_dir $DATA_DIR \
                --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT \
                --model $MODEL --lr $LR --num_epochs $NUM_EPOCHS \
                --optimizer $OPTIMIZER --weight_decay $WEIGHT_DECAY --scheduler $SCHEDULER \
                --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE \
                --patience $PATIENCE --start_from_epoch $START_FROM_EPOCH \
                --seeds $SEEDS
        done
    done
done
