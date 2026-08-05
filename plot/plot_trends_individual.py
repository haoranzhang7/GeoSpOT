"""Scatter a domain-pair distance matrix against downstream transfer accuracy and fit a linear trend."""

import os, argparse

from trend_common import (
    EMBEDDING_LABELS_LONG, EMBEDDING_COLORS, load_results, build_combined_df,
    scatter_regplot, filename_tags, y_title_for,
)

import matplotlib.pyplot as plt


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    results_df = load_results(args.results_file, args.metric, args.rescale_acc)
    combined_df = build_combined_df(args.distance_file, args.distance_type, results_df,
                                     args.mask_domains, args.outlier_domains, args.include_self_pair)

    color = EMBEDDING_COLORS.get(args.embedding_type, 'steelblue')
    x_title = args.x_title or EMBEDDING_LABELS_LONG.get(args.embedding_type, f"Domain Distance ({args.distance_type})")
    y_title = y_title_for(args.metric, args.rescale_acc)

    tags = filename_tags(args.rescale_acc, args.mask_domains, args.outlier_domains, args.include_self_pair)
    plot_filepath = os.path.join(args.output_dir, f"trend_{args.distance_type}_{args.metric}_{tags}.png")

    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))
    rho, p_value, r2, n_pairs = scatter_regplot(ax, combined_df, x_title, y_title, color)
    plt.tight_layout()
    fig.savefig(plot_filepath, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved plot to {plot_filepath} (rho={rho:.4f}, p={p_value:.4f}, r2={r2:.4f}, n_pairs={n_pairs})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance_file', type=str, required=True, help="Path to a square N x N distance matrix CSV (rows/cols = domain indices).")
    parser.add_argument('--distance_type', type=str, required=True, help="Label for the distance metric, e.g. 'ot', 'mmd', 'fid', 'cosine'.")
    parser.add_argument('--results_file', type=str, required=True, help="Path to the combined transfer-performance results CSV (long format with src_domain_idx/tgt_domain_idx columns).")
    parser.add_argument('--metric', type=str, default='avg_test_acc', help="Column in results_file to use as the y-axis (e.g. avg_test_acc, avg_test_top3_acc, avg_test_top5_acc).")
    parser.add_argument('--output_dir', type=str, default='./trend_plots', help="Directory to save the plot in.")
    parser.add_argument('--include_self_pair', action='store_true', help="Include src==tgt domain pairs in the plot.")
    parser.add_argument('--rescale_acc', action='store_true', help="Rescale the metric as the relative (%%) change from each source domain's self-pair value.")
    parser.add_argument('--mask_domains', type=int, nargs='*', default=[], help="Domain indices to exclude from both src and tgt.")
    parser.add_argument('--outlier_domains', type=int, nargs='*', default=[], help="Domain indices considered outliers (e.g. for geoyfcc_text, Panama=17). Excluded from both src and tgt, so they're left out of the plot and of stats like Spearman's rho, the regression fit, and R^2.")
    parser.add_argument('--embedding_type', type=str, required=True, choices=list(EMBEDDING_LABELS_LONG), help="Embedding type used to color the plot and label the x-axis.")
    parser.add_argument('--x_title', type=str, default=None, help="Override the auto-generated x-axis label, e.g. an embedding type name like 'BERT Embeddings'.")
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 10), help="Figure size as 'width height'.")

    main(parser.parse_args())
