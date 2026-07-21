#!/usr/bin/env python
"""
Build a CSV of the overall Spearman's rho between each domain-pair distance metric (cosine /
MMD / OT / FID / geodesic) and downstream transfer accuracy, computed across all cross-domain
pairs at once (one rho per (embedding_type, distance_type), not fixed to any one domain).

Reuses trend_common.py's load_results/build_combined_df (the same loading/filtering logic as
plot_trends_individual.py) and rho_csv_common.py to locate each distance matrix; combos whose
distance matrix hasn't been computed yet (e.g. FID, or OT on the geodesic embedding, as of this
writing) are skipped with a warning rather than failing the whole run.

Example:
  python plot/build_overall_rho_csv.py
  python plot/build_overall_rho_csv.py --distance_types ot mmd
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from trend_common import load_results, build_combined_df
from rho_csv_common import EMBEDDING_TYPES, DISTANCE_TYPES, iter_available_combos


def compute_overall_rho(distance_file, distance_type, results_df, outlier_domains):
    combined_df = build_combined_df(distance_file, distance_type, results_df, [], outlier_domains)
    rho, p_value = spearmanr(combined_df['dist_value'], combined_df['acc_value'])
    return rho, p_value, len(combined_df)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="geoyfcc_text")
    parser.add_argument("--results_file", default="results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv")
    parser.add_argument("--metric", default="avg_test_acc")
    parser.add_argument("--rescale_acc", action="store_true", default=True,
                         help="Rescale the metric as the relative (%%) change from each target domain's "
                              "self-pair value (on by default, matching the rest of the plot/ pipeline).")
    parser.add_argument("--no_rescale_acc", dest="rescale_acc", action="store_false")
    parser.add_argument("--outlier_domains", type=int, nargs="*", default=[17],
                         help="Domain indices excluded from both src and tgt (default: 17, e.g. Panama "
                              "for geoyfcc_text).")
    parser.add_argument("--embedding_types", nargs="+", default=EMBEDDING_TYPES)
    parser.add_argument("--distance_types", nargs="+", default=DISTANCE_TYPES)
    parser.add_argument("--out", default="plot/plots/rho_overall.csv")
    args = parser.parse_args()

    results_df = load_results(args.results_file, args.metric, args.rescale_acc)

    rows = []
    for embedding_type, distance_type, distance_file, method, note in iter_available_combos(
            args.dataset, args.embedding_types, args.distance_types):
        try:
            rho, p_value, n_pairs = compute_overall_rho(distance_file, distance_type, results_df,
                                                          args.outlier_domains)
        except ValueError as e:
            print(f"[WARNING] {e}, skipping {distance_type}/{embedding_type}")
            continue

        rows.append({
            "dataset": args.dataset, "embedding_type": embedding_type, "distance_type": distance_type,
            "method": method or "", "rho": rho, "abs_rho": abs(rho), "p_value": p_value,
            "n_pairs": n_pairs, "distance_file": str(distance_file), "note": note,
        })
        label = f"{embedding_type}/{distance_type}" + (f"/{method}" if method else "")
        print(f"{label}: rho={rho:.4f}, p={p_value:.4f}, n_pairs={n_pairs} (file={distance_file.name})")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
