#!/bin/bash
set -e

# Evaluates checkpoints from 12_subset_selection_all_target.sh and
# 18_subset_selection_combined_lambda05_all_target.sh on the global test set,
# loading the dataset/test dataloader once and reusing it across every config.

python src/evaluation/zeroshot_test_eval_subset_grid.py \
    --dataset geoyfcc_text --domain_type countries \
    --data_dir ./data \
    --checkpoint_root ./results/subset/checkpoints \
    --log_root ./results/subset/logs \
    --results_root ./results/subset/test_results \
    --model bert_singlelabel --eval_batch_size 1024 \
    --seeds 6651033 9272605 1206448 2180968 114325 \
    --tgt_domain all \
    --k_values 1 2 5 --budget_values 2000 5000 10000 \
    --ot_embedding_types bert geoclip satclip geodesic \
    --combined_embeddings geoclip satclip geodesic --combined_lambda 0.5
