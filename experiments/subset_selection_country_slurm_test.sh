#!/bin/bash
# Smoke test of pretrain_by_domain_subset.py across every (target, k, method) combo
# 07_subset_selection_slurm.sh submits, before submitting the full grid. Tiny subset/1 epoch
# per run keeps it fast despite the sweep. Writes to a separate results/subset_test dir so
# it can't be mistaken for a real completed run.
set -e
source "/curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh"
conda activate geospot
cd "$(dirname "$0")/.."

# Must match 07_subset_selection_slurm.sh's TARGETS/K_VALUES/EMBS/LOC_EMBS/LAMBDA
TARGETS=(57 12); K_VALUES=(1 2 5)
EMBS=(bert geoclip satclip geodesic); LOC_EMBS=(geoclip satclip geodesic); LAMBDA=0.5

COMMON=(-s 6651033 --subset_size 100 --val_subset_size 50 --num_epochs 1 --patience 1 --start_from_epoch 0
        --checkpoint_root ./results/subset_test/checkpoints --log_root ./results/subset_test/logs)
OT_COMMON=(--ot_distance_dir ./data/geoyfcc_text/distances/ot_distance/ --domain_selection_method ot
           --ot_method sinkhorn_log --ot_reg 0.01 --ot_iter 1000 --ot_metric cosine --ot_norm max_per_domain)

for tgt in "${TARGETS[@]}"; do for k in "${K_VALUES[@]}"; do
    RUN=(--num_domains "$k" --tgt_domain "$tgt")

    echo "== random tgt=$tgt k=$k =="
    python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" "${RUN[@]}" --domain_selection_method random

    for e in "${EMBS[@]}"; do
        echo "== ot emb=$e tgt=$tgt k=$k =="
        python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" "${RUN[@]}" "${OT_COMMON[@]}" --ot_embedding_type "$e"
    done
    for e in "${LOC_EMBS[@]}"; do
        echo "== ot emb=${e}+bert tgt=$tgt k=$k =="
        python src/training/pretrain_by_domain_subset.py "${COMMON[@]}" "${RUN[@]}" "${OT_COMMON[@]}" --ot_embedding_type "${e}+bert" --ot_lambda "$LAMBDA"
    done
done; done
