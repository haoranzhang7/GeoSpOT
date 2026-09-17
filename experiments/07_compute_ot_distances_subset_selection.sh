#!/bin/bash
set -e

# Precomputes every OT distance file needed by the subset-selection experiments,
export CUDA_VISIBLE_DEVICES=0

DATASET="geoyfcc_text"
TOTAL_DOMAINS=62
NORMALIZE_COST="max_per_domain"

TARGETS=(57 12)
K_VALUES=(1 2 5)
EMBS=(bert geoclip satclip geodesic)
LOC_EMBS=(geoclip satclip geodesic)  # '+bert' pair order must match 16_...sh (it's in the CSV filename)
LAMBDA=0.5

compute_distances() {
    local tgt=$1 k=$2 emb=$3 lam=$4
    local args=(--dataset-name $DATASET --embedding-type "$emb" --source-domain-idx "$tgt"
                --total-domains $TOTAL_DOMAINS --k "$k" --metric cosine --normalize-cost $NORMALIZE_COST)
    [ "$k" -gt 1 ] && args+=(--greedy-sequential)
    [ -n "$lam" ] && args+=(--lambda "$lam")
    python src/distances/ot_distance.py "${args[@]}"
}

# Target-specific (07_subset_selection.sh / 07_subset_selection_slurm.sh)
for tgt in "${TARGETS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for e in "${EMBS[@]}"; do compute_distances "$tgt" "$k" "$e" ""; done
        for e in "${LOC_EMBS[@]}"; do compute_distances "$tgt" "$k" "${e}+bert" "$LAMBDA"; done
    done
done

# All-target (12_subset_selection_all_target.sh / 18_subset_selection_combined_lambda05_all_target.sh)
for k in "${K_VALUES[@]}"; do
    for e in "${EMBS[@]}"; do compute_distances all "$k" "$e" ""; done
    for e in "${LOC_EMBS[@]}"; do compute_distances all "$k" "${e}+bert" "$LAMBDA"; done
done
