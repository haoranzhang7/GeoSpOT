#!/bin/bash
set -e

# Evaluates the checkpoints produced by 12_subset_selection_all_target.sh on
# the global test set (tgt_domain=all pools the test split across every
# domain, see get_domain_split_mask in src/data/load_datasets.py). Must be
# run after 12_subset_selection_all_target.sh has produced checkpoints.

# Paths
DATA_DIR="./data"
CHECKPOINT_ROOT="./results/subset/checkpoints"
LOG_ROOT="./results/subset/logs"
RESULTS_ROOT="./results/subset/test_results"

# Dataset/model
DATASET="geoyfcc_text"
DOMAIN_TYPE="countries"
MODEL="bert_singlelabel"
EVAL_BATCH_SIZE=512

# Experiment grid (must match 12_subset_selection_all_target.sh)
K_VALUES=(1 2 5)
BUDGET_VALUES=(2000 5000 10000)
SEEDS=(6651033 9272605 1206448 2180968 114325)
OT_EMBEDDING_TYPES=("bert" "geoclip" "satclip" "geodesic")

# Val budget = budget / 2
declare -A VAL_BUDGET
VAL_BUDGET[2000]=1000
VAL_BUDGET[5000]=2500
VAL_BUDGET[10000]=5000

for k in "${K_VALUES[@]}"; do
    for budget in "${BUDGET_VALUES[@]}"; do
        val_budget=${VAL_BUDGET[$budget]}

        # Random selection (all seeds evaluated in one call)
        python src/evaluation/zeroshot_test_eval_subset.py \
            -t 0 \
            --dataset $DATASET --domain_type $DOMAIN_TYPE \
            --data_dir $DATA_DIR \
            --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT --results_root $RESULTS_ROOT \
            --model $MODEL --eval_batch_size $EVAL_BATCH_SIZE \
            --seeds "${SEEDS[@]}" \
            --subset_size $budget --val_subset_size $val_budget \
            --num_domains $k --domain_selection_method random \
            --tgt_domain all
    done
done

for k in "${K_VALUES[@]}"; do
    for budget in "${BUDGET_VALUES[@]}"; do
        val_budget=${VAL_BUDGET[$budget]}

        # OT-based selection
        for emb in "${OT_EMBEDDING_TYPES[@]}"; do
            python src/evaluation/zeroshot_test_eval_subset.py \
                -t 0 \
                --dataset $DATASET --domain_type $DOMAIN_TYPE \
                --data_dir $DATA_DIR \
                --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT --results_root $RESULTS_ROOT \
                --model $MODEL --eval_batch_size $EVAL_BATCH_SIZE \
                --seeds "${SEEDS[@]}" \
                --subset_size $budget --val_subset_size $val_budget \
                --num_domains $k --domain_selection_method ot \
                --tgt_domain all --ot_embedding_type $emb \
                --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000 \
                --ot_metric cosine --ot_norm max_per_domain
        done
    done
done
