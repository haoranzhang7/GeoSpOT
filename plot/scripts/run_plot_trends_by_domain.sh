#!/bin/bash
set -e

# Runs plot_trends_by_domain.py once per domain index, fixing it as the source
# domain, then once more fixing it as the target domain. Domain 17 (outlier,
# e.g. Panama for geoyfcc_text) is skipped in both loops.
#
# Usage:
#   ./plot/scripts/run_plot_trends_by_domain.sh --distance_file <path> --distance_type <type> --embedding_type <type> --results_file <path> [other plot_trends_by_domain.py args...]
#
# All arguments are forwarded as-is to plot_trends_by_domain.py, e.g.:
#   ./plot/scripts/run_plot_trends_by_domain.sh \
#       --distance_file data/geoyfcc_text/distances/mmd_distance/mmd_bert_kmultiscale_meuclidean_nmax_per_domain_and_normalized_after.csv \
#       --distance_type mmd \
#       --embedding_type bert \
#       --results_file results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv \
#       --output_dir plot/plots/bert \
#       --rescale_acc \
#       --outlier_domains 17

NUM_DOMAINS=62
EXCLUDE_DOMAIN=17

for ((domain=0; domain<NUM_DOMAINS; domain++)); do
    if [ "$domain" -eq "$EXCLUDE_DOMAIN" ]; then
        continue
    fi
    python plot/plot_trends_by_domain.py --src_domain "$domain" "$@"
done

for ((domain=0; domain<NUM_DOMAINS; domain++)); do
    if [ "$domain" -eq "$EXCLUDE_DOMAIN" ]; then
        continue
    fi
    python plot/plot_trends_by_domain.py --tgt_domain "$domain" "$@"
done
