"""Shared helpers for the plot/plot_trends_{individual,by_domain,side_by_side}.py scripts.

Each of those scripts scatters a domain-pair distance matrix (OT/MMD/FID/cosine/geodesic)
against downstream transfer accuracy for the same domain pairs and fits a linear trend.
"""

import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

EMBEDDING_LABELS = {
    'bert': 'BERT',
    'geoclip': 'GeoCLIP',
    'satclip': 'SatCLIP',
    'geodesic': 'Geodesic',
}
EMBEDDING_LABELS_LONG = {
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


def load_distance_matrix(distance_file, distance_type):
    """Load a square N x N distance matrix CSV (rows/cols = domain indices) into long format.

    OT matrices (see save_k1_matrix in src/distances/ot_distance.py) may carry an extra pooled
    "all" source row used for global subset selection; it has no matching target column, so it's
    dropped here before casting the index/columns to int.
    """
    matrix = pd.read_csv(distance_file, index_col=0)
    matrix = matrix.drop(index="all", errors="ignore").drop(columns="all", errors="ignore")
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)

    distance_df = matrix.stack(future_stack=True).rename('dist_value') \
        .rename_axis(['src_domain_idx', 'tgt_domain_idx']).reset_index()
    distance_df['is_self_pair'] = distance_df['src_domain_idx'] == distance_df['tgt_domain_idx']
    distance_df['distance_type'] = distance_type
    return distance_df


def rescale_metric(results_df, metric):
    """Rescale metric as the relative (%) change from each target domain's own self-pair value:
    (Acc_tgt(M_src) - Acc_tgt(M_tgt)) / Acc_tgt(M_tgt) * 100."""
    self_pair_values = results_df.loc[results_df['src_domain_idx'] == results_df['tgt_domain_idx']].set_index('tgt_domain_idx')[metric]
    baseline = results_df['tgt_domain_idx'].map(self_pair_values)
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
    return combined_df[~combined_df['src_domain_idx'].isin(domain_indices) & ~combined_df['tgt_domain_idx'].isin(domain_indices)]


def build_combined_df(distance_file, distance_type, results_df, mask_domains, outlier_domains,
                       include_self_pair=False):
    distance_df = load_distance_matrix(distance_file, distance_type)
    combined_df = pd.merge(distance_df, results_df, how='inner', on=['src_domain_idx', 'tgt_domain_idx'])
    combined_df = combined_df.dropna(subset=['dist_value', 'acc_value'])

    if not include_self_pair:
        combined_df = combined_df[~combined_df['is_self_pair']]

    combined_df = exclude_domains(combined_df, mask_domains)
    combined_df = exclude_domains(combined_df, outlier_domains)

    if combined_df.empty:
        raise ValueError(f"No domain pairs left to plot after filtering for {distance_file} "
                          "- check distance_file/results_file overlap and filters.")
    return combined_df


def scatter_regplot(ax, df, x_title, y_title, color, *, show_ylabel=True, show_pvalue=True,
                     label_fontsize=16, tick_fontsize=14, legend_fontsize=16):
    """Scatter dist_value vs. acc_value with a fitted regression line; returns (rho, p_value, r2, n)."""
    scatter_x, scatter_y = 'dist_value', 'acc_value'
    scatter_color = tuple(0.7 * c for c in to_rgb(color))

    rho, p_value = spearmanr(df[scatter_x], df[scatter_y])
    linear_reg = LinearRegression().fit(df[[scatter_x]], df[scatter_y])
    r2 = r2_score(df[scatter_y], linear_reg.predict(df[[scatter_x]]))

    label_text = r"$\rho$" + f": {rho:.4f},\n"
    if show_pvalue:
        label_text += r"$p$" + f"-value: {p_value:.4f},\n"
    label_text += r"$\mathcal{R}^2$" + f": {r2:.4f}"

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set(color='#333333', linewidth=1.1)

    alpha_value = 0.15 if len(df) > 50 else 0.4
    ax.scatter(df[scatter_x], df[scatter_y], s=38, alpha=alpha_value, color=scatter_color,
               linewidths=0, zorder=2)
    sns.regplot(data=df, x=scatter_x, y=scatter_y, ci=None,
                scatter=False, fit_reg=True, ax=ax, label=label_text, color=color,
                line_kws={'linewidth': 5, 'zorder': 3, 'solid_capstyle': 'round'})

    ax.tick_params('both', labelsize=tick_fontsize, colors='#333333')
    ax.set_xlabel(x_title, fontsize=label_fontsize, color='#222222')
    if show_ylabel:
        ax.set_ylabel(y_title, fontsize=label_fontsize, color='#222222')
    else:
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelleft=False)

    legend = ax.legend(fontsize=legend_fontsize, handlelength=0, handletextpad=0,
                        frameon=True, fancybox=False, edgecolor='#CCCCCC',
                        facecolor='white', framealpha=0.9, borderpad=0.7)
    for handle in legend.legend_handles:
        handle.set_visible(False)
    for text in legend.get_texts():
        text.set_color('#222222')

    return rho, p_value, r2, len(df)


def filename_tags(rescale_acc, mask_domains, outlier_domains, include_self_pair=None):
    """Build the common `..._exc-self_rescale_acc_mask1,2_outlier3` style filename suffix."""
    tags = []
    if include_self_pair is not None:
        tags.append("inc-self" if include_self_pair else "exc-self")
    tags.append("rescale_acc" if rescale_acc else "raw_acc")
    suffix = "_".join(tags)
    if mask_domains:
        suffix += f"_mask{','.join(str(i) for i in mask_domains)}"
    if outlier_domains:
        suffix += f"_outlier{','.join(str(i) for i in outlier_domains)}"
    return suffix


def y_title_for(metric, rescale_acc):
    return "Relative Change in Test Accuracy (%)" if rescale_acc else metric.replace('_', ' ').title()
