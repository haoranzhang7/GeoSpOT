#!/bin/bash
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=pretrain_subset
#SBATCH --output=results/subset/logs/slurm_%A_%a.out
set -e
source "/curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh"
conda activate geospot
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

TARGETS=(57 12); K_VALUES=(1 2 5); BUDGET_VALUES=(2000 5000 10000); SEEDS=(6651033 9272605 1206448 2180968 114325)
EMBS=(bert geoclip satclip_L40 geodesic)
LOC_EMBS=(geoclip satclip_L40 geodesic); LAMBDA=0.5  # '+bert' pair order must match 16_...sh (it's in the CSV filename)

JOBS=()
for tgt in "${TARGETS[@]}"; do for k in "${K_VALUES[@]}"; do for b in "${BUDGET_VALUES[@]}"; do
    for s in "${SEEDS[@]}"; do JOBS+=("$tgt $k $b $s random none none"); done
    for s in "${SEEDS[@]}"; do for e in "${EMBS[@]}"; do JOBS+=("$tgt $k $b $s ot $e none"); done; done
    for s in "${SEEDS[@]}"; do for e in "${LOC_EMBS[@]}"; do JOBS+=("$tgt $k $b $s ot ${e}+bert $LAMBDA"); done; done
done; done; done

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

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    IDX=()
    for i in "${!JOBS[@]}"; do read -r tgt k b s m e lam <<< "${JOBS[$i]}"; is_done "$tgt" "$k" "$b" "$s" "$m" "$e" "$lam" || IDX+=("$i"); done
    echo "${#IDX[@]}/${#JOBS[@]} jobs remaining"
    [ ${#IDX[@]} -eq 0 ] && exit 0
    IFS=,; sbatch --array="${IDX[*]}" "$0"; exit 0
fi

read -r TGT K BUDGET SEED METHOD EMB LAMBDA <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
ARGS=(-s $SEED --subset_size $BUDGET --val_subset_size $((BUDGET / 2))
      --num_domains $K --domain_selection_method $METHOD --tgt_domain $TGT --start_from_epoch 20)
if [ "$METHOD" = ot ]; then
    ARGS+=(--ot_distance_dir ./data/geoyfcc_text/distances/ot_distance/
           --ot_embedding_type $EMB --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000
           --ot_metric cosine --ot_norm max_per_domain)
    [ "$LAMBDA" != none ] && ARGS+=(--ot_lambda $LAMBDA)
fi

python src/training/pretrain_by_domain_subset.py "${ARGS[@]}"
