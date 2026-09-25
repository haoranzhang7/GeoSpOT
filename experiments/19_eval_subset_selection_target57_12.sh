#!/bin/bash
set -e

# Evaluates every completed checkpoint from experiments/07_subset_selection.sh (tgt_domain 57, 12)
SUMMARY_CSV="./results/subset/test_results/2_zeroshot_eval_subset/bert_singlelabel/summary/zeroshot_eval_subset_target57_12_full_test_summary.csv"

for TGT in 57 12; do
    python src/evaluation/zeroshot_test_eval_subset_grid.py \
        --dataset geoyfcc_text --domain_type countries \
        --data_dir ./data \
        --checkpoint_root ./results/subset/checkpoints \
        --log_root ./results/subset/logs \
        --results_root ./results/subset/test_results \
        --model bert_singlelabel --eval_batch_size 1024 \
        --seeds 6651033 9272605 1206448 2180968 114325 \
        --tgt_domain "$TGT" \
        --discover_checkpoints \
        --summary_csv "$SUMMARY_CSV"
done
