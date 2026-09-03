#!/bin/bash
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=pretrain_subset_comb_all
#SBATCH --output=results/subset/logs/slurm_%A_%a.out
set -e
source "/curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh"
conda activate geospot
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

K_VALUES=(1 2 5); BUDGET_VALUES=(2000 5000 10000); SEEDS=(6651033 9272605 1206448 2180968 114325)
# '+' order must match 16_...sh ("${emb}+bert"): it's part of the CSV filename.
EMBS=(geoclip satclip geodesic); LAMBDA=0.5
declare -A VAL_BUDGET=( [2000]=1000 [5000]=2500 [10000]=5000 )

JOBS=()
for k in "${K_VALUES[@]}"; do for b in "${BUDGET_VALUES[@]}"; do
    for e in "${EMBS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$k $b $s $e"); done; done
done; done

is_done() {
    local k=$1 b=$2 s=$3 e=$4 v=${VAL_BUDGET[$b]}
    local dir="results/subset/logs/1_pretrain_subset${b}_K${k}_ot/bert_singlelabel"
    local suf="_subset${b}_K${k}_OT_${e}+bert_sinkhorn_log_0.01_1000_cosine_max_per_domain"
    suf+="_lambda${LAMBDA}_V${v}_tgtall"
    grep -ql "Training completed in" "$dir"/pretrain_countries_bert_singlelabel${suf}_seed${s}_*.log 2>/dev/null
}

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    IDX=()
    for i in "${!JOBS[@]}"; do read -r k b s e <<< "${JOBS[$i]}"; is_done "$k" "$b" "$s" "$e" || IDX+=("$i"); done
    echo "${#IDX[@]}/${#JOBS[@]} jobs remaining"
    [ ${#IDX[@]} -eq 0 ] && exit 0
    IFS=,; sbatch --array="${IDX[*]}" "$0"; exit 0
fi

read -r K BUDGET SEED EMB <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
python src/training/pretrain_by_domain_subset.py \
    -s $SEED --subset_size $BUDGET --val_subset_size ${VAL_BUDGET[$BUDGET]} \
    --num_domains $K --domain_selection_method ot --tgt_domain all --start_from_epoch 20 \
    --ot_distance_dir ./data/geoyfcc_text/distances/ot_distance/ \
    --ot_embedding_type "${EMB}+bert" --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000 \
    --ot_metric cosine --ot_norm max_per_domain --ot_lambda $LAMBDA
