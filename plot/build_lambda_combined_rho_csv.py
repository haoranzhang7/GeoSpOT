#!/usr/bin/env python
"""
Build a CSV of Spearman's rho between the lambda-combined OT distance (lambda * location +
(1 - lambda) * visual embedding, e.g. lambda * geoclip + (1 - lambda) * bert) and downstream
transfer accuracy -- one row per lambda value swept for each location embedding (geoclip,
satclip, geodesic), plus the two pure endpoints (lambda=0: visual-only, lambda=1: location-only).

Reuses rho_csv_common.py's find_ot_sinkhorn_log_file/find_ot_lambda_combo_files to locate each
distance matrix, and trend_common.py's load_results/build_combined_df to compute rho against
downstream accuracy (as in build_overall_rho_csv.py).

Example:
  python plot/build_lambda_combined_rho_csv.py
  python plot/build_lambda_combined_rho_csv.py --location_embeddings geoclip satclip
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from trend_common import load_results, build_combined_df
from rho_csv_common import LOCATION_EMBEDDING_TYPES, find_ot_sinkhorn_log_file, find_ot_lambda_combo_files


def compute_rho(distance_file, results_df, outlier_domains):
    combined_df = build_combined_df(distance_file, "ot", results_df, [], outlier_domains)
    rho, p_value = spearmanr(combined_df['dist_value'], combined_df['acc_value'])
    linear_reg = LinearRegression().fit(combined_df[['dist_value']], combined_df['acc_value'])
    r2 = r2_score(combined_df['acc_value'], linear_reg.predict(combined_df[['dist_value']]))
    domains_used = sorted(set(combined_df['src_domain_idx']) | set(combined_df['tgt_domain_idx']))
    return rho, p_value, r2, len(combined_df), domains_used


def format_domain_subset(domain_indices):
    """Collapse a sorted list of domain indices into a compact range string, e.g. [0,1,2,4,5] -> '0-2,4-5'."""
    if not domain_indices:
        return ""
    ranges = []
    start = prev = domain_indices[0]
    for idx in domain_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="geoyfcc_text")
    parser.add_argument("--dataset_name", default=None, help="Value for the dataset_name column (default: --dataset).")
    parser.add_argument("--results_file", default="results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv")
    parser.add_argument("--metric", default="avg_test_acc")
    parser.add_argument("--rescale_acc", action="store_true", default=True, help="Rescale the metric as the relative (%%) change from each target domain's self-pair value (on by default, matching the rest of the plot/ pipeline).")
    parser.add_argument("--no_rescale_acc", dest="rescale_acc", action="store_false")
    parser.add_argument("--outlier_domains", type=int, nargs="*", default=[17], help="Domain indices excluded from both src and tgt (default: 17, e.g. Panama for geoyfcc_text).")
    parser.add_argument("--visual_embedding", default="bert",
                         help="Name for the visual_embedding column -- the fixed non-location embedding combined "
                              "with each location embedding (bert/text, in GeoSpOT's case).")
    parser.add_argument("--location_embeddings", nargs="+", default=LOCATION_EMBEDDING_TYPES)
    parser.add_argument("--out", default="plot/plots/rho_lambda_combined.csv")
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.dataset
    target_metric = f"relative_{args.metric}_change_percent" if args.rescale_acc else args.metric
    results_df = load_results(args.results_file, args.metric, args.rescale_acc)

    visual_file, _ = find_ot_sinkhorn_log_file(args.dataset, args.visual_embedding)
    if visual_file is None:
        print(f"[WARNING] No sinkhorn_log ot distance file found for visual embedding={args.visual_embedding}; "
              f"lambda=0.0 endpoint will be skipped for every location embedding")

    rows = []
    for location_embedding in args.location_embeddings:
        location_file, _ = find_ot_sinkhorn_log_file(args.dataset, location_embedding)
        combo_files = find_ot_lambda_combo_files(args.dataset, location_embedding)

        sweep = []
        if visual_file is not None:
            sweep.append((0.0, visual_file))
        sweep.extend(combo_files)
        if location_file is not None:
            sweep.append((1.0, location_file))
        else:
            print(f"[WARNING] No sinkhorn_log ot distance file found for location embedding={location_embedding}, "
                  f"skipping lambda=1.0 endpoint")

        if not sweep:
            print(f"[WARNING] No OT distance files found for {location_embedding}, skipping")
            continue

        embedding_pair = f"{args.visual_embedding}+{location_embedding}"
        for lambda_index, (lambda_location, distance_file) in enumerate(sweep):
            try:
                rho, p_value, r2, n_pairs, domains_used = compute_rho(distance_file, results_df, args.outlier_domains)
            except ValueError as e:
                print(f"[WARNING] {e}, skipping {embedding_pair} lambda={lambda_location}")
                continue

            rows.append({
                "dataset_name": dataset_name, "distance_type": "combined_ot",
                "visual_embedding": args.visual_embedding, "location_embedding": location_embedding,
                "embedding_pair": embedding_pair, "correlation_type": "spearman",
                "target_metric": target_metric, "regression_type": "linear",
                "num_domains_total": "", "excluded_domain_idx": "",
                "lambda_index": lambda_index, "lambda_visual": round(1 - lambda_location, 10),
                "lambda_location": lambda_location, "spearman_rho": rho, "abs_spearman_rho": abs(rho),
                "spearman_p_value": p_value, "r2": r2, "n_pairs": n_pairs,
                "num_domains_used": len(domains_used), "domain_subset": format_domain_subset(domains_used),
            })
            print(f"{embedding_pair} lambda_location={lambda_location}: rho={rho:.4f}, p={p_value:.4f}, "
                  f"r2={r2:.4f}, n_pairs={n_pairs} (file={Path(distance_file).name})")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
