"""Like plot_trends_individual.py, but draws one distance matrix per embedding type as a
side-by-side panel in a single figure (e.g. GeoCLIP vs. SatCLIP), sharing the y-axis."""

import os
import argparse

from trend_common import (
    EMBEDDING_LABELS, EMBEDDING_COLORS, load_results, build_combined_df,
    scatter_regplot, filename_tags, y_title_for,
)

import matplotlib.pyplot as plt

ALWAYS_EXCLUDED_DOMAINS = [17]  # e.g. Panama for geoyfcc_text, treated as a permanent outlier


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    results_df = load_results(args.results_file, args.metric, args.rescale_acc)
    dfs, x_titles, colors = [], [], []
    for distance_file, embedding_type in zip(args.distance_files, args.embedding_types):
        dfs.append(build_combined_df(distance_file, args.distance_type, results_df,
                                      args.mask_domains, args.outlier_domains))
        x_titles.append(f"{args.x_label_prefix} ({EMBEDDING_LABELS.get(embedding_type, embedding_type)})")
        colors.append(EMBEDDING_COLORS.get(embedding_type, 'steelblue'))

    y_title = y_title_for(args.metric, args.rescale_acc)
    tags = filename_tags(args.rescale_acc, args.mask_domains, args.outlier_domains, include_self_pair=False)
    embeddings_str = "_".join(args.embedding_types)
    plot_filename = f"trend_{args.distance_type}_{args.metric}_{embeddings_str}_{tags}.png"
    plot_filepath = os.path.join(args.output_dir, plot_filename)

    fig, axes = plt.subplots(1, len(dfs), figsize=tuple(args.figsize), sharey=True)
    for i, (ax, df, x_title, color) in enumerate(zip(axes, dfs, x_titles, colors)):
        rho, p_value, r2, n_pairs = scatter_regplot(
            ax, df, x_title, y_title, color, show_ylabel=(i == 0), show_pvalue=False,
            label_fontsize=28, tick_fontsize=22, legend_fontsize=24)
        print(f"{x_title}: rho={rho:.4f}, p={p_value:.4f}, r2={r2:.4f}, n_pairs={n_pairs}")
    plt.tight_layout()
    fig.savefig(plot_filepath, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved plot to {plot_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance_files', type=str, nargs='+', required=True,
                        help="Paths to square N x N distance matrix CSVs, one per subplot, e.g. an OT distance "
                             "matrix for GeoCLIP followed by one for SatCLIP.")
    parser.add_argument('--embedding_types', type=str, nargs='+', required=True, choices=list(EMBEDDING_LABELS),
                        help="Embedding type per distance file, used to color each subplot and label its x-axis.")
    parser.add_argument('--distance_type', type=str, required=True,
                        help="Label for the distance metric, e.g. 'ot', 'mmd', 'fid'. Assumed to be the same across all subplots.")
    parser.add_argument('--x_label_prefix', type=str, default="GeoSpOT Distance",
                        help="Prefix for each subplot's x-axis label; the embedding name is appended in parentheses, "
                             "e.g. 'GeoSpOT Distance (GeoCLIP)'.")
    parser.add_argument('--results_file', type=str, required=True,
                        help="Path to the combined transfer-performance results CSV (long format with src_domain_idx/tgt_domain_idx columns).")
    parser.add_argument('--metric', type=str, default='avg_test_acc',
                        help="Column in results_file to use as the y-axis (e.g. avg_test_acc, avg_test_top3_acc, avg_test_top5_acc).")
    parser.add_argument('--output_dir', type=str, default='./trend_plots',
                        help="Directory to save the plot in.")
    parser.add_argument('--rescale_acc', action='store_true',
                        help="Rescale the metric as the relative (%%) change from each source domain's self-pair value.")
    parser.add_argument('--mask_domains', type=int, nargs='*', default=[],
                        help="Domain indices to exclude from both src and tgt.")
    parser.add_argument('--outlier_domains', type=int, nargs='*', default=[],
                        help="Additional domain indices considered outliers, excluded from both src and tgt "
                             f"(domain {ALWAYS_EXCLUDED_DOMAINS[0]}, e.g. Panama for geoyfcc_text, is always "
                             "excluded regardless of this flag). Self pairs (src==tgt) are also always excluded.")
    parser.add_argument('--figsize', type=float, nargs=2, default=(16, 8),
                        help="Figure size as 'width height' for the whole (multi-panel) figure.")

    args = parser.parse_args()
    if len(args.embedding_types) != len(args.distance_files):
        raise ValueError("--embedding_types must have the same number of entries as --distance_files")
    args.outlier_domains = sorted(set(args.outlier_domains) | set(ALWAYS_EXCLUDED_DOMAINS))

    main(args)
