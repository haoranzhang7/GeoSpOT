"""Camera-ready (one-column) side-by-side comparison of subset-selection methods for
target domains US, China, and 'all'. Non-'+' (single-embedding) methods only, for now.

US/China come from results/subset_selection_chart_estimates_from_image.csv (values
traced off an existing results chart, since the local checkpoints/eval outputs for
these two targets no longer exist and wandb only has validation, not test, accuracy).
'all' comes from the zeroshot-eval summary CSV, via build_subset_selection_csv.load_summary.

Usage:
  python plot/plot_subset_selection_target_comparison.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from build_subset_selection_csv import load_summary
from plot_method_vs_baselines import METHOD_LABELS, METHOD_COLORS, K_ORDER, K_MARKERS, K_DODGE

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.8,
})

DOMAINS = ['US', 'China', 'all']
METHODS = ['random', 'geodesic', 'geoclip', 'satclip', 'bert']
GLOBAL_COLOR = 'grey'


def chart_estimate_df(csv_path='results/subset_selection_chart_estimates_from_image.csv'):
    df = pd.read_csv(csv_path)
    df['method'] = df['method'].str.lower()
    return df[df['domain'].isin(['US', 'China']) & ~df['method'].str.contains(r'\+')]


def all_domain_df(results_root='results/subset/test_results', model='bert_singlelabel'):
    df = load_summary(results_root, model)
    df['method'] = df['select_by'].where(df['select_by'] != 'ot', df['ot_embedding_type'])
    df = df[~df['method'].str.contains(r'\+')]
    rows = []
    for (budget, k, method), g in df.groupby(['budget', 'num_select', 'method']):
        rows.append({'domain': 'all', 'budget': int(budget), 'K': str(int(k)), 'method': method,
                      'value_est': g['acc'].mean(), 'std_est': g['acc'].std() if len(g) > 1 else 0.0})
    return pd.DataFrame(rows)


def plot(df, domains, methods, out_path):
    fig, axes = plt.subplots(1, len(domains), figsize=(1.73 * len(domains), 2.1), sharey=True)
    if len(domains) == 1:
        axes = [axes]
    x = np.arange(len(methods))
    for ax, domain in zip(axes, domains):
        ddf = df[df['domain'] == domain]
        for mi, method in enumerate(methods):
            for k in K_ORDER:
                matches = ddf[(ddf['method'] == method) & (ddf['K'] == k)]
                if matches.empty:
                    continue
                r = matches.iloc[0]
                ax.errorbar(x[mi] + K_DODGE[k], r['value_est'], yerr=r['std_est'], fmt=K_MARKERS[k],
                             color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method], elinewidth=0.8,
                             capsize=2, capthick=0.8, markersize=3.5, markeredgecolor='white', markeredgewidth=0.3, zorder=3)

        global_rows = ddf[ddf['method'] == 'global']
        if len(global_rows):
            g = global_rows.iloc[0]
            ax.axhspan(g['value_est'] - g['std_est'], g['value_est'] + g['std_est'], color=GLOBAL_COLOR, alpha=0.15, zorder=1)
            ax.axhline(g['value_est'], color=GLOBAL_COLOR, linestyle='--', linewidth=1, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=45, ha='right')
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_title(f'Target: {"Global" if domain == "all" else domain}', pad=3)
        ax.grid(axis='y', alpha=0.3, linewidth=0.4, zorder=0)
        ax.tick_params(width=0.6, length=2, pad=2)

    axes[0].set_ylabel('Test Accuracy')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    k_handles = [plt.Line2D([0], [0], marker=K_MARKERS[k], linestyle='None', markersize=4, label=f'$K={k}$',
                             markerfacecolor='none', markeredgecolor='dimgray', markeredgewidth=0.8) for k in K_ORDER]
    global_handle = plt.Line2D([0], [0], color=GLOBAL_COLOR, linestyle='--', linewidth=1, label='Global')
    fig.legend(handles=k_handles + [global_handle], loc='upper center', ncol=len(K_ORDER) + 1,
               bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3, columnspacing=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_path}')


def main():
    df = pd.concat([chart_estimate_df(), all_domain_df()], ignore_index=True)
    common_budgets = set.intersection(*(set(df.loc[df['domain'] == d, 'budget']) for d in DOMAINS))
    domain_variants = {'': DOMAINS, '_no_global': ['US', 'China']}
    method_variants = {'': METHODS, '_no_bert': [m for m in METHODS if m != 'bert']}
    for budget in sorted(common_budgets):
        bdf = df[df['budget'] == budget]
        for domain_suffix, domains in domain_variants.items():
            for method_suffix, methods in method_variants.items():
                plot(bdf, domains, methods, f'plot/plots/subset_selection_target_comparison{domain_suffix}{method_suffix}_N{budget}.png')


if __name__ == '__main__':
    main()
