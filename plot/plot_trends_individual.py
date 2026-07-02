
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb

from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

plt.rcParams['font.family'] = 'Times New Roman'

KNOWN_DISTANCE_TYPES = ['ot', 'mmd', 'fid', 'cosine', 'arc']

EMBEDDING_LABELS = {
    'bert': 'BERT Embeddings',
    'geoclip': 'GeoCLIP Embeddings',
    'satclip': 'SatCLIP Embeddings',
    'geodesic': 'Geodesic',
}

EMBEDDING_COLORS = {
    'bert': 'mediumorchid',
    'geoclip': 'cornflowerblue',
    'satclip': 'green',
    'geodesic': 'darkorange',
}


def infer_distance_type(distance_file):
    """Guess a distance type label (ot/mmd/fid/cosine/arc) from the filename."""
    stem = os.path.basename(distance_file).lower()
    for known_type in KNOWN_DISTANCE_TYPES:
        if known_type in stem:
            return known_type
    return os.path.splitext(os.path.basename(distance_file))[0]


def infer_embedding_type(distance_file):
    """Guess the embedding type (bert/geoclip/satclip/geodesic) from the filename."""
    stem = os.path.basename(distance_file).lower()
    for known_type in EMBEDDING_LABELS:
        if known_type in stem:
            return known_type
    return None


def load_distance_matrix(distance_file, distance_type):
    """Load a square N x N distance matrix CSV (rows/cols = domain indices) into long format."""
    matrix = pd.read_csv(distance_file, index_col=0)
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)

    distance_df = matrix.stack(future_stack=True).rename('dist_value').rename_axis(['src_domain_idx', 'tgt_domain_idx']).reset_index()
    distance_df['is_self_pair'] = distance_df['src_domain_idx'] == distance_df['tgt_domain_idx']
    distance_df['distance_type'] = distance_type
    return distance_df


def rescale_metric(results_df, metric):
    """Rescale metric as the relative (%) change from each source domain's self-pair value."""
    self_pair_values = results_df.loc[
        results_df['src_domain_idx'] == results_df['tgt_domain_idx']
    ].set_index('src_domain_idx')[metric]

    baseline = results_df['src_domain_idx'].map(self_pair_values)
    rescaled = results_df.copy()
    rescaled[metric] = (results_df[metric] - baseline) * 100 / baseline
    return rescaled


def load_results(results_file, metric, rescale_acc_flag):
    results_df = pd.read_csv(results_file)
    for required_col in ('src_domain_idx', 'tgt_domain_idx'):
        if required_col not in results_df.columns:
            raise ValueError(f"Results file is missing required column '{required_col}'")
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results file. Available columns: {list(results_df.columns)}")

    if rescale_acc_flag:
        results_df = rescale_metric(results_df, metric)

    return results_df[['src_domain_idx', 'tgt_domain_idx', metric]].rename(columns={metric: 'acc_value'})


def exclude_domains(combined_df, domain_indices):
    """Drop pairs where src or tgt domain idx is in domain_indices."""
    if not domain_indices:
        return combined_df
    return combined_df[
        ~combined_df['src_domain_idx'].isin(domain_indices) &
        ~combined_df['tgt_domain_idx'].isin(domain_indices)
    ]


def plot_trends(df, filename, x_title, y_title, color, figsize=(10, 10)):
    scatter_x = 'dist_value'
    scatter_y = 'acc_value'
    scatter_color = tuple(0.7 * c for c in to_rgb(color))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    rho, p_value = spearmanr(df[scatter_x], df[scatter_y])
    linear_reg = LinearRegression().fit(df[[scatter_x]], df[scatter_y])
    y_pred = linear_reg.predict(df[[scatter_x]])
    r2 = r2_score(df[scatter_y], y_pred)

    label_text = r"$\rho$" + f": {rho:.4f},\n" + r"$p$" + f"-value: {p_value:.4f},\n" + \
                    r"$\mathcal{R}^2$" + f": {r2:.4f}"

    alpha_value = 0.3 if len(df) > 50 else 0.6
    ax.scatter(df[scatter_x], df[scatter_y], alpha=alpha_value, color=scatter_color)
    sns.regplot(data=df, x=scatter_x, y=scatter_y, ci=None,
                scatter=False, fit_reg=True, ax=ax, label=label_text, color=color)

    ax.tick_params('both', labelsize=14)
    ax.set_xlabel(f"{x_title}", fontsize=16)
    ax.set_ylabel(f"{y_title}", fontsize=16)
    ax.legend(fontsize=16)

    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    distance_type = args.distance_type or infer_distance_type(args.distance_file)
    distance_df = load_distance_matrix(args.distance_file, distance_type)
    results_df = load_results(args.results_file, args.metric, args.rescale_acc)

    combined_df = pd.merge(distance_df, results_df, how='inner', on=['src_domain_idx', 'tgt_domain_idx'])
    combined_df = combined_df.dropna(subset=['dist_value', 'acc_value'])

    if not args.include_self_pair:
        combined_df = combined_df[~combined_df['is_self_pair']]

    combined_df = exclude_domains(combined_df, args.mask_domains)
    combined_df = exclude_domains(combined_df, args.outlier_domains)

    if combined_df.empty:
        raise ValueError("No domain pairs left to plot after filtering - check distance_file/results_file overlap and filters.")

    embedding_type = args.embedding_type or infer_embedding_type(args.distance_file)
    color = EMBEDDING_COLORS.get(embedding_type, 'steelblue')

    x_title = args.x_title or EMBEDDING_LABELS.get(embedding_type, f"Domain Distance ({distance_type})")
    y_title = "Relative Change in Test Accuracy (%)" if args.rescale_acc else args.metric.replace('_', ' ').title()

    include_self_pair_str = "inc-self" if args.include_self_pair else "exc-self"
    rescale_acc_str = "rescale_acc" if args.rescale_acc else "raw_acc"
    mask_domain_str = f"_mask{','.join(str(i) for i in args.mask_domains)}" if args.mask_domains else ""
    outlier_str = f"_outlier{','.join(str(i) for i in args.outlier_domains)}" if args.outlier_domains else ""
    distance_type = distance_type.lower().replace(' ', '_')
    plot_filename = f"trend_{distance_type}_{args.metric}_{include_self_pair_str}_{rescale_acc_str}{mask_domain_str}{outlier_str}.png"
    plot_filepath = os.path.join(args.output_dir, plot_filename)

    plot_trends(combined_df, plot_filepath, x_title, y_title, color, figsize=tuple(args.figsize))
    print(f"Saved plot to {plot_filepath}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--distance_file', type=str, required=True,
                        help="Path to a square N x N distance matrix CSV (rows/cols = domain indices), e.g. an OT, MMD, FID, or cosine distance matrix.")
    parser.add_argument('--distance_type', type=str, default=None,
                        help="Label for the distance metric, e.g. 'ot', 'mmd', 'fid', 'cosine'. Inferred from the filename if omitted.")
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
    parser.add_argument('--embedding_type', type=str, default=None, choices=list(EMBEDDING_LABELS),
                        help="Embedding type used to color the plot and label the x-axis "
                             "('bert', 'geoclip', 'satclip', or 'geodesic'). Inferred from the filename if omitted.")
    parser.add_argument('--x_title', type=str, default=None,
                        help="Override the auto-generated x-axis label, e.g. an embedding type name like 'BERT Embeddings'.")
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 10),
                        help="Figure size as 'width height'.")

    args = parser.parse_args()

    main(args)
