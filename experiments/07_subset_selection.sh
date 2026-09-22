#!/bin/bash
set -e

DATA_DIR="./data"
OT_DISTANCE_DIR="./data/geoyfcc_text/distances/ot_distance/"
CHECKPOINT_ROOT="./results/subset/checkpoints"
LOG_ROOT="./results/subset/logs"

TARGETS=(57 12); K_VALUES=(1 2)  # TODO: add 5 back once slurm job 32687664 (tgt57,K5,budget2000) finishes -- it's still training satclip/geodesic/geoclip+bert/satclip+bert/geodesic+bert, and resubmitting those now would duplicate that work
BUDGET_VALUES=(2000)
SEEDS=(6651033 9272605 1206448 2180968 114325)
REST_EMBS=(geoclip satclip); REST_LOC_EMBS=(geoclip satclip)  # run before the priority group below
PRIORITY_EMBS=(bert geodesic); PRIORITY_LOC_EMBS=(geodesic); LAMBDA=0.5  # '+bert' pair order must match 16_...sh (it's in the CSV filename)

# This machine runs shard 0 of SHARD_COUNT; the cluster (07_subset_selection_slurm.sh) excludes
# shard 0 and covers the rest. Both scripts build the exact same JOBS grid/order below, so the
# shard split lines up and nothing gets run twice. Lower SHARD_COUNT to give this machine more.
SHARD_COUNT=2
SHARD_ID=0

# Seeds are the outermost loop so each seed finishes across every config before the next seed
# starts. Within a seed: random first, then geoclip/satclip (+bert), then bert/geodesic/geodesic+bert last.
JOBS=()
for s in "${SEEDS[@]}"; do for tgt in "${TARGETS[@]}"; do for k in "${K_VALUES[@]}"; do for b in "${BUDGET_VALUES[@]}"; do
    JOBS+=("$tgt $k $b $s random none none")
    for e in "${REST_EMBS[@]}"; do JOBS+=("$tgt $k $b $s ot $e none"); done
    for e in "${REST_LOC_EMBS[@]}"; do JOBS+=("$tgt $k $b $s ot ${e}+bert $LAMBDA"); done
    for e in "${PRIORITY_EMBS[@]}"; do JOBS+=("$tgt $k $b $s ot $e none"); done
    for e in "${PRIORITY_LOC_EMBS[@]}"; do JOBS+=("$tgt $k $b $s ot ${e}+bert $LAMBDA"); done
done; done; done; done

is_done() {
    local tgt=$1 k=$2 b=$3 s=$4 m=$5 e=$6 lam=$7 v=$((b / 2))
    local dir="results/subset/logs/1_pretrain_subset${b}_K${k}_${m}/bert_singlelabel"
    local suf="_subset${b}_K${k}"
    if [ "$m" = ot ]; then
        suf+="_OT_${e}_sinkhorn_log_0.01_1000_cosine_max_per_domain"
        [ "$lam" != none ] && suf+="_lambda${lam}"
    else
        suf+="_${m}"
    fi
    suf+="_V${v}_tgt${tgt}"
    grep -ql "Training completed in" "$dir"/pretrain_countries_bert_singlelabel${suf}_seed${s}_*.log 2>/dev/null
}

COMMON=(--dataset geoyfcc_text --domain_type countries --data_dir "$DATA_DIR"
        --checkpoint_root "$CHECKPOINT_ROOT" --log_root "$LOG_ROOT"
        --model bert_singlelabel --lr 2e-5 --num_epochs 50
        --optimizer AdamW --weight_decay 0.01 --scheduler cosine
        --train_batch_size 64 --eval_batch_size 512 --patience 10 --start_from_epoch 20)

echo "Running shard $SHARD_ID/$SHARD_COUNT: $(( (${#JOBS[@]} + SHARD_COUNT - 1 - SHARD_ID) / SHARD_COUNT )) jobs assigned to this machine"

for i in "${!JOBS[@]}"; do
    [ $((i % SHARD_COUNT)) -eq "$SHARD_ID" ] || continue
    read -r TGT K BUDGET SEED METHOD EMB LAMBDA_VAL <<< "${JOBS[$i]}"
    is_done "$TGT" "$K" "$BUDGET" "$SEED" "$METHOD" "$EMB" "$LAMBDA_VAL" && continue

    RUN=(-s "$SEED" --subset_size "$BUDGET" --val_subset_size $((BUDGET / 2)) --num_domains "$K" --tgt_domain "$TGT")
    if [ "$METHOD" = ot ]; then
        ARGS=("${COMMON[@]}" "${RUN[@]}" --ot_distance_dir "$OT_DISTANCE_DIR" --domain_selection_method ot
              --ot_embedding_type "$EMB" --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000
              --ot_metric cosine --ot_norm max_per_domain)
        [ "$LAMBDA_VAL" != none ] && ARGS+=(--ot_lambda "$LAMBDA_VAL")
    else
        ARGS=("${COMMON[@]}" "${RUN[@]}" --domain_selection_method random)
    fi
    python src/training/pretrain_by_domain_subset.py "${ARGS[@]}"
done
