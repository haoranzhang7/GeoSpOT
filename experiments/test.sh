#!/bin/bash
# Quick ~2min smoke test of pretrain_by_domain_subset.py (random + ot paths) before
# submitting the full grid via 12_subset_selection_all_target_slurm.sh. Writes to a
# separate results/subset_test dir so it can't be mistaken for a real completed run.
set -e
source "/curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh"
conda activate geospot
cd "$(dirname "$0")/.."

COMMON=(-s 6651033 --subset_size 100 --val_subset_size 50 --num_epochs 1 --patience 1
        --start_from_epoch 0 --num_domains 1 --tgt_domain all
        --checkpoint_root ./results/subset_test/checkpoints --log_root ./results/subset_test/logs)

echo "== random =="
python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" --domain_selection_method random

echo "== ot =="
python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" --domain_selection_method ot \
    --ot_distance_dir ./data/geoyfcc_text/distances/ot_distance/ --ot_embedding_type bert \
    --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000 --ot_metric cosine --ot_norm max_per_domain
