#!/usr/bin/env python
"""
Build a CSV of the average Spearman's rho between each domain-pair distance metric (cosine /
MMD / OT / FID / geodesic) and downstream transfer accuracy, averaged over fixed-source-domain
regressions: for each source domain, rho is computed only over that domain's pairs with every
target domain, then those per-domain rho values are averaged (mean/std) into one row per
(embedding_type, distance_type).

This mirrors what plot_trends_by_domain.py + summarize_by_domain_trends.py compute together
(one plot and one CSV row per domain, then aggregated), but without saving 62 plots per combo --
it reuses trend_common.py's load_results/build_combined_df directly and calls spearmanr per
source-domain group. Combos whose distance matrix hasn't been computed yet (e.g. FID, or OT on
the geodesic embedding, as of this writing) are skipped with a warning.

Example:
  python plot/build_rho_by_src_domain_csv.py
  python plot/build_rho_by_src_domain_csv.py --distance_types ot mmd
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from trend_common import load_results, build_combined_df
from rho_csv_common import EMBEDDING_TYPES, DISTANCE_TYPES, iter_available_combos


def per_src_domain_rhos(combined_df, min_pairs):
    rows = []
    for src_domain, group in combined_df.groupby('src_domain_idx'):
        if len(group) < min_pairs:
            continue
        rho, p_value = spearmanr(group['dist_value'], group['acc_value'])
        rows.append({'src_domain_idx': src_domain, 'rho': rho, 'p_value': p_value, 'n_pairs': len(group)})
    return pd.DataFrame(rows)


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
    parser.add_argument("--min_pairs", type=int, default=3,
                         help="Minimum number of target domains needed to compute a given source "
                              "domain's rho; source domains with fewer pairs are skipped.")
    parser.add_argument("--embedding_types", nargs="+", default=EMBEDDING_TYPES)
    parser.add_argument("--distance_types", nargs="+", default=DISTANCE_TYPES)
    parser.add_argument("--out", default="plot/plots/rho_avg_by_src_domain.csv")
    args = parser.parse_args()

    results_df = load_results(args.results_file, args.metric, args.rescale_acc)

    rows = []
    for embedding_type, distance_type, distance_file, method, note in iter_available_combos(
            args.dataset, args.embedding_types, args.distance_types):
        try:
            combined_df = build_combined_df(distance_file, distance_type, results_df, [], args.outlier_domains)
        except ValueError as e:
            print(f"[WARNING] {e}, skipping {distance_type}/{embedding_type}")
            continue

        per_domain = per_src_domain_rhos(combined_df, args.min_pairs)
        if per_domain.empty:
            print(f"[WARNING] No source domains with >= {args.min_pairs} pairs for "
                  f"{distance_type}/{embedding_type}, skipping")
            continue

        rows.append({
            "dataset": args.dataset, "embedding_type": embedding_type, "distance_type": distance_type,
            "method": method or "", "rho_mean": per_domain['rho'].mean(), "rho_std": per_domain['rho'].std(),
            "abs_rho_mean": per_domain['rho'].abs().mean(), "abs_rho_std": per_domain['rho'].abs().std(),
            "n_src_domains": len(per_domain), "distance_file": str(distance_file), "note": note,
        })
        label = f"{embedding_type}/{distance_type}" + (f"/{method}" if method else "")
        print(f"{label}: rho_mean={per_domain['rho'].mean():.4f} "
              f"(n_src_domains={len(per_domain)}, file={distance_file.name})")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
