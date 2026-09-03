#!/bin/bash
set -e

# 12_subset_selection_all_target.sh, but selecting with the combined distance
# 0.5*{geoclip,satclip,geodesic} + 0.5*bert. Needs the lambda_0.5 distance CSVs
# from 16_compute_combined_ot_distances_lambda_sweep.sh (else they're recomputed
# on the fly, slowly). Only the combined arms: the baselines in 12 are
# lambda-independent.

DATA_DIR="./data"
OT_DISTANCE_DIR="./data/geoyfcc_text/distances/ot_distance/"
CHECKPOINT_ROOT="./results/subset/checkpoints"
LOG_ROOT="./results/subset/logs"
DATASET="geoyfcc_text"
DOMAIN_TYPE="countries"
MODEL="bert_singlelabel"
LAMBDA=0.5

K_VALUES=(1 2 5)
BUDGET_VALUES=(2000 5000 10000)
SEEDS=(6651033 9272605 1206448 2180968 114325)
# '+' order must match 16_...sh ("${emb}+bert"): it's part of the CSV filename.
LOCATION_EMBEDDINGS=("geoclip" "satclip" "geodesic")

# Skip runs that already have a "Training completed in" log, so re-running
# this script only launches the jobs that still need to run.
is_done() {
  local k=$1 budget=$2 seed=$3 emb=$4 val_budget=$((budget / 2))
  local dir="${LOG_ROOT}/1_pretrain_subset${budget}_K${k}_ot/${MODEL}"
  local suf="_subset${budget}_K${k}_OT_${emb}+bert_sinkhorn_log_0.01_1000_cosine_max_per_domain"
  suf+="_lambda${LAMBDA}_V${val_budget}_tgtall"
  grep -ql "Training completed in" "$dir"/pretrain_${DOMAIN_TYPE}_${MODEL}${suf}_seed${seed}_*.log 2>/dev/null
}

for budget in "${BUDGET_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    for emb in "${LOCATION_EMBEDDINGS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        if is_done "$k" "$budget" "$seed" "$emb"; then
          echo "Skipping (already completed): K=$k budget=$budget seed=$seed emb=${emb}+bert lambda=$LAMBDA"
          continue
        fi
        python src/training/pretrain_by_domain_subset.py \
          --dataset $DATASET --domain_type $DOMAIN_TYPE \
          --data_dir $DATA_DIR --ot_distance_dir $OT_DISTANCE_DIR \
          --checkpoint_root $CHECKPOINT_ROOT --log_root $LOG_ROOT \
          --model $MODEL --lr 2e-5 --num_epochs 50 \
          --optimizer AdamW --weight_decay 0.01 --scheduler cosine \
          --train_batch_size 64 --eval_batch_size 512 \
          --patience 10 --start_from_epoch 20 \
          -s $seed --subset_size $budget --val_subset_size $((budget / 2)) \
          --num_domains $k --domain_selection_method ot \
          --tgt_domain all --ot_embedding_type "${emb}+bert" \
          --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000 \
          --ot_metric cosine --ot_norm max_per_domain --ot_lambda $LAMBDA
      done
    done
  done
done
