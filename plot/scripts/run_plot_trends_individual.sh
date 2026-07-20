#!/bin/bash
set -e

# Runs plot_trends_individual.py (whole-dataset trend, not fixed to one domain) for every
# (embedding_type, distance_type) combination with a precomputed distance matrix, plus the
# embedding-agnostic geodesic distance.
#
# Usage:
#   ./plot/scripts/run_plot_trends_individual.sh [other plot_trends_individual.py args...]

DATASET="geoyfcc_text"
RESULTS_FILE="results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv"
OUTLIER_DOMAINS=17

distance_file() {
    local dist_type=$1 emb=$2
    case "$dist_type" in
        cosine)   echo "data/${DATASET}/distances/cosine_distance/cosine_${emb}_avg_similarity.csv" ;;
        mmd)      echo "data/${DATASET}/distances/mmd_distance/mmd_${emb}_kmultiscale_meuclidean_nmax_per_domain_and_normalized_after.csv" ;;
        fid)      echo "data/${DATASET}/distances/fid_distance/fid_${emb}_meuclidean_nnone.csv" ;;
        ot)       [ "$emb" = "satclip" ] && emb="satclip_L40"
                  echo "data/${DATASET}/distances/ot_distance/ot_distance_${emb}_max_per_domain.csv" ;;
        geodesic) echo "data/${DATASET}/distances/geodesic_distance/geodesic_avg_distance.csv" ;;
    esac
}

for emb in bert geoclip satclip; do
    for dist_type in cosine mmd fid ot; do
        f=$(distance_file "$dist_type" "$emb")
        if [ ! -f "$f" ]; then
            echo "[skip] missing $f"
            continue
        fi
        python plot/plot_trends_individual.py \
            --distance_file "$f" --distance_type "$dist_type" --embedding_type "$emb" \
            --results_file "$RESULTS_FILE" --output_dir "plot/plots/${emb}" \
            --rescale_acc --outlier_domains $OUTLIER_DOMAINS "$@"
    done
done

python plot/plot_trends_individual.py \
    --distance_file "$(distance_file geodesic _)" --distance_type geodesic --embedding_type geodesic \
    --results_file "$RESULTS_FILE" --output_dir "plot/plots/geodesic" \
    --rescale_acc --outlier_domains $OUTLIER_DOMAINS "$@"
