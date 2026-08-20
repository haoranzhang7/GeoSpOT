#!/bin/bash
set -e

# Same subset-selection setup as 07_subset_selection.sh, but the OT target is
# the pooled distribution across every domain ("all") instead of one specific
# held-out domain. Requires 11_compute_ot_distances_all_target.sh to have been
# run first so the ot_distance_matrix_*.csv / distances_k*_*.csv files include
# an "all" source row/entries.

# Paths
DATA_DIR="./data"
OT_DISTANCE_DIR="./data/geoyfcc_text/distances/ot_distance/"
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
K_VALUES=(1 2 5)
BUDGET_VALUES=(2000 5000 10000)
SEEDS=(6651033 9272605 1206448 2180968 114325)
OT_EMBEDDING_TYPES=("bert" "geoclip" "satclip" "geodesic")

# Val budget = budget / 2
declare -A VAL_BUDGET
VAL_BUDGET[2000]=1000
VAL_BUDGET[5000]=2500
VAL_BUDGET[10000]=5000

# Skip runs that already have a "Training completed in" log, so re-running
# this script only launches the jobs that still need to run.
is_done() {
    local k=$1 budget=$2 seed=$3 method=$4 emb=$5 val_budget=${VAL_BUDGET[$budget]}
    local dir="${LOG_ROOT}/1_pretrain_subset${budget}_K${k}_${method}/${MODEL}"
    local suf="_subset${budget}_K${k}"
    if [ "$method" = "ot" ]; then
        suf+="_OT_${emb}_sinkhorn_log_0.01_1000_cosine_max_per_domain"
    else
        suf+="_${method}"
    fi
    suf+="_V${val_budget}_tgtall"
    grep -ql "Training completed in" "$dir"/pretrain_${DOMAIN_TYPE}_${MODEL}${suf}_seed${seed}_*.log 2>/dev/null
}

for k in "${K_VALUES[@]}"; do
    for budget in "${BUDGET_VALUES[@]}"; do
        val_budget=${VAL_BUDGET[$budget]}
        # Random selection (seeded), K random domains
        for seed in "${SEEDS[@]}"; do
            if is_done "$k" "$budget" "$seed" random ""; then
                echo "Skipping (already completed): K=$k budget=$budget seed=$seed method=random"
                continue
            fi
            python src/training/pretrain_by_domain_subset.py \
                --dataset $DATASET --domain_type $DOMAIN_TYPE \
                --data_dir $DATA_DIR \
                --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT \
                --model $MODEL --lr $LR --num_epochs $NUM_EPOCHS \
                --optimizer $OPTIMIZER --weight_decay $WEIGHT_DECAY --scheduler $SCHEDULER \
                --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE \
                --patience $PATIENCE --start_from_epoch $START_FROM_EPOCH \
                -s $seed --subset_size $budget --val_subset_size $val_budget \
                --num_domains $k --domain_selection_method random \
                --tgt_domain all
        done

        # OT-based selection
        for seed in "${SEEDS[@]}"; do
            for emb in "${OT_EMBEDDING_TYPES[@]}"; do
                if is_done "$k" "$budget" "$seed" ot "$emb"; then
                    echo "Skipping (already completed): K=$k budget=$budget seed=$seed method=ot emb=$emb"
                    continue
                fi
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
                    --tgt_domain all --ot_embedding_type $emb \
                    --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000 \
                    --ot_metric cosine --ot_norm max_per_domain
            done
        done
    done
done
