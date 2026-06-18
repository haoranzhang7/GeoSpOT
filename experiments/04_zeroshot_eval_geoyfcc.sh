#!/bin/bash
set -e

DATA_DIR="./data"
CHECKPOINT_ROOT="./results/pretrain/checkpoints"
LOG_ROOT="./results/zeroshot/logs"
RESULTS_ROOT="./results/zeroshot/test_results"
DATASET="geoyfcc_text"
DOMAIN_TYPE="countries"
MODEL="bert_singlelabel"
EVAL_BATCH_SIZE=512
NUM_DOMAINS=62
SEEDS="48329 17046 62984 31507 90861"
TARGET_DOMAINS=$(seq 0 $((NUM_DOMAINS - 1)) | tr '\n' ' ')

for domain in $(seq 0 $((NUM_DOMAINS - 1))); do
    python src/evaluation/zeroshot_test_eval.py \
        -d $domain -t $TARGET_DOMAINS \
        --dataset $DATASET --domain_type $DOMAIN_TYPE \
        --data_dir $DATA_DIR \
        --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT --results_root $RESULTS_ROOT \
        --model $MODEL --eval_batch_size $EVAL_BATCH_SIZE \
        --seeds $SEEDS
done
