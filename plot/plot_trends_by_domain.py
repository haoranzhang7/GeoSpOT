"""Like plot_trends_individual.py, but fixed to a single source or target domain, and it
upserts the resulting (rho, p_value, r2, n_pairs) into a running per-domain summary CSV.
Intended to be called once per domain by plot/scripts/run_plot_trends_by_domain.sh."""

import os
import argparse

from trend_common import (
    EMBEDDING_LABELS, EMBEDDING_COLORS, load_results, build_combined_df,
    scatter_regplot, filename_tags, y_title_for,
)

import pandas as pd
import matplotlib.pyplot as plt


def update_summary_csv(csv_path, domain_col, domain_val, rho, p_value, r2, n_pairs):
    """Upsert a domain's test values into the running summary CSV (one row per domain)."""
    row = pd.DataFrame([{
        domain_col: domain_val, 'rho': rho, 'p_value': p_value, 'r2': r2, 'n_pairs': n_pairs,
    }])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[existing[domain_col] != domain_val]
        row = pd.concat([existing, row], ignore_index=True)
    row.sort_values(domain_col).reset_index(drop=True).to_csv(csv_path, index=False)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    results_df = load_results(args.results_file, args.metric, args.rescale_acc)
    combined_df = build_combined_df(args.distance_file, args.distance_type, results_df,
                                     args.mask_domains, args.outlier_domains, args.include_self_pair)

    if args.src_domain is not None:
        combined_df = combined_df[combined_df['src_domain_idx'] == args.src_domain]
        domain_col, domain_val, fixed_domain_str = 'src_domain_idx', args.src_domain, f"src{args.src_domain}"
    else:
        combined_df = combined_df[combined_df['tgt_domain_idx'] == args.tgt_domain]
        domain_col, domain_val, fixed_domain_str = 'tgt_domain_idx', args.tgt_domain, f"tgt{args.tgt_domain}"

    if combined_df.empty:
        raise ValueError("No domain pairs left to plot after filtering - check distance_file/results_file overlap and filters.")

    color = EMBEDDING_COLORS.get(args.embedding_type, 'steelblue')
    x_title = args.x_title or EMBEDDING_LABELS.get(args.embedding_type, f"Domain Distance ({args.distance_type})")
    y_title = y_title_for(args.metric, args.rescale_acc)

    tags = filename_tags(args.rescale_acc, args.mask_domains, args.outlier_domains, args.include_self_pair)
    summary_base = f"trend_{args.distance_type}_{args.metric}_{tags}"
    plot_filepath = os.path.join(args.output_dir, f"{summary_base}_{fixed_domain_str}.png")

    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))
    rho, p_value, r2, n_pairs = scatter_regplot(ax, combined_df, x_title, y_title, color)
    plt.tight_layout()
    fig.savefig(plot_filepath, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved plot to {plot_filepath}")

    summary_csv_path = os.path.join(args.output_dir, f"{summary_base}_by_{'src' if args.src_domain is not None else 'tgt'}_domain.csv")
    update_summary_csv(summary_csv_path, domain_col, domain_val, rho, p_value, r2, n_pairs)
    print(f"Updated summary CSV at {summary_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance_file', type=str, required=True,
                        help="Path to a square N x N distance matrix CSV (rows/cols = domain indices).")
    parser.add_argument('--distance_type', type=str, required=True,
                        help="Label for the distance metric, e.g. 'ot', 'mmd', 'fid', 'cosine'.")
    parser.add_argument('--results_file', type=str, required=True,
                        help="Path to the combined transfer-performance results CSV (long format with src_domain_idx/tgt_domain_idx columns).")
    parser.add_argument('--metric', type=str, default='avg_test_acc',
                        help="Column in results_file to use as the y-axis (e.g. avg_test_acc, avg_test_top3_acc, avg_test_top5_acc).")
    parser.add_argument('--output_dir', type=str, default='./trend_plots',
                        help="Directory to save the plot in.")
    parser.add_argument('--include_self_pair', action='store_true',
                        help="Include src==tgt domain pairs in the plot.")
    parser.add_argument('--rescale_acc', action='store_true',
                        help="Rescale the metric as the relative (%%) change from each source domain's self-pair value.")
    parser.add_argument('--mask_domains', type=int, nargs='*', default=[],
                        help="Domain indices to exclude from both src and tgt.")
    parser.add_argument('--outlier_domains', type=int, nargs='*', default=[],
                        help="Domain indices considered outliers (e.g. for geoyfcc_text, Panama=17). "
                             "Excluded from both src and tgt, so they're left out of the plot and of "
                             "stats like Spearman's rho, the regression fit, and R^2.")
    parser.add_argument('--embedding_type', type=str, required=True, choices=list(EMBEDDING_LABELS),
                        help="Embedding type used to color the plot and label the x-axis.")
    parser.add_argument('--x_title', type=str, default=None,
                        help="Override the auto-generated x-axis label, e.g. an embedding type name like 'BERT Embeddings'.")
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 10),
                        help="Figure size as 'width height'.")

    fixed_domain_group = parser.add_mutually_exclusive_group(required=True)
    fixed_domain_group.add_argument('--src_domain', type=int, default=None,
                        help="Fix the source domain index; plot/Spearman's rho computed only over pairs with this src_domain_idx. "
                             "Saved filename includes 'src{num}'.")
    fixed_domain_group.add_argument('--tgt_domain', type=int, default=None,
                        help="Fix the target domain index; plot/Spearman's rho computed only over pairs with this tgt_domain_idx. "
                             "Saved filename includes 'tgt{num}'.")

    main(parser.parse_args())
