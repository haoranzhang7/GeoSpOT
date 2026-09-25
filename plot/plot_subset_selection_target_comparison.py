"""Camera-ready (one-column) side-by-side comparison of subset-selection methods
across target domains.

US/China come from results/subset_selection_chart_estimates_from_image.csv (values
traced off an existing results chart, since the local checkpoints/eval outputs for
these two targets no longer exist and wandb only has validation, not test, accuracy).
Other domains come from a zeroshot-eval summary CSV: either every summary/*.csv under
--results_root/2_zeroshot_eval_subset/--model (default), or a single CSV passed via
--summary_csv (e.g. a target-specific summary that shouldn't be mixed with other
summary files in the same directory).

Usage:
  python plot/plot_subset_selection_target_comparison.py
  python plot/plot_subset_selection_target_comparison.py --summary_csv path/to/summary.csv \
      --no_chart_estimates --include_combos --out_dir plot/plots/subset_selection/geoyfcc_text
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from build_subset_selection_csv import load_summary
from plot_method_vs_baselines import METHOD_ORDER, METHOD_LABELS, K_ORDER, K_MARKERS
from trend_common import EMBEDDING_COLORS

# Wider than plot_method_vs_baselines.K_DODGE: this plot's markers are bigger and its subplots
# narrower, so the default dodge left the 3 K points per method touching/overlapping.
K_DODGE = {'1': -0.28, '2': 0.0, '5': 0.28}

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10, 'axes.linewidth': 0.6,
    'axes.edgecolor': '#555555',
})

DOMAINS = ['US', 'China', 'all']
# Numeric country_id codes (as they appear in tgt_country/domain) for the two targets in
# experiments/19_eval_subset_selection_target57_12.sh, used to render country names in titles.
DOMAIN_CODE_NAMES = {'57': 'US', '12': 'China'}
METHODS = ['random', 'geodesic', 'geoclip', 'satclip', 'bert']
METRICS = {'acc': 'Test Accuracy', 'top3_acc': 'Test Top-3 Accuracy', 'top5_acc': 'Top-5 Test Accuracy'}
GLOBAL_COLOR = 'grey'
# random has no embedding of its own, so it keeps the neutral black used elsewhere in these plots.
METHOD_COLORS = {'random': 'black', **EMBEDDING_COLORS}


def _method_label(method):
    return METHOD_LABELS[method].replace('+', ' + ')


def chart_estimate_df(csv_path):
    df = pd.read_csv(csv_path)
    df['method'] = df['method'].str.lower()
    return df[df['domain'].isin(['US', 'China']) & ~df['method'].str.contains(r'\+')]


def summary_domain_df(summary_csv=None, results_root='results/subset/test_results',
                       model='bert_singlelabel', include_combos=False, metric='acc'):
    df = pd.read_csv(summary_csv) if summary_csv else load_summary(results_root, model)
    df['domain'] = df['tgt_country'].str.replace('^Domain ', '', regex=True)
    df['method'] = df['select_by'].where(df['select_by'] != 'ot', df['ot_embedding_type'])
    if not include_combos:
        df = df[~df['method'].str.contains(r'\+')]
    rows = []
    for (domain, budget, k, method), g in df.groupby(['domain', 'budget', 'num_select', 'method']):
        rows.append({'domain': domain, 'budget': int(budget), 'K': str(int(k)), 'method': method,
                      'value_est': g[metric].mean(), 'std_est': g[metric].std() if len(g) > 1 else 0.0,
                      'n_seeds': g['model_seed'].nunique()})
    return pd.DataFrame(rows)


def plot(df, domains, methods, out_path, ylabel='Test Accuracy'):
    fig, axes = plt.subplots(1, len(domains), figsize=(2.0 * len(domains), 2.6), sharey=True)
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
                             color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method], elinewidth=1.0,
                             capsize=2.5, capthick=1.0, markersize=4.5, markeredgecolor='white', markeredgewidth=0.4, zorder=3)

        global_rows = ddf[ddf['method'] == 'global']
        if len(global_rows):
            g = global_rows.iloc[0]
            ax.axhspan(g['value_est'] - g['std_est'], g['value_est'] + g['std_est'], color=GLOBAL_COLOR, alpha=0.15, zorder=1)
            ax.axhline(g['value_est'], color=GLOBAL_COLOR, linestyle='--', linewidth=1, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels([_method_label(m) for m in methods], rotation=40, ha='right')
        ax.set_xlim(-0.6, len(methods) - 0.4)
        domain_label = 'Global' if domain == 'all' else DOMAIN_CODE_NAMES.get(domain, domain)
        ax.set_title(f'Target: {domain_label}', pad=3, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linewidth=0.4, zorder=0)
        ax.tick_params(width=0.6, length=2, pad=2)

    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    k_handles = [plt.Line2D([0], [0], marker=K_MARKERS[k], linestyle='None', markersize=6, label=f'$K={k}$',
                             markerfacecolor='none', markeredgecolor='dimgray', markeredgewidth=1.0) for k in K_ORDER]
    if (df['method'] == 'global').any():
        k_handles.append(plt.Line2D([0], [0], color=GLOBAL_COLOR, linestyle='--', linewidth=1, label='Global'))
    fig.legend(handles=k_handles, loc='upper center', ncol=len(k_handles),
               bbox_to_anchor=(0.5, 0.98), frameon=False, handletextpad=0.3, columnspacing=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    # Pin the axes' vertical extent to what tight_layout gave the smaller (pre-camera-ready) font
    # sizes: bigger fonts otherwise make tight_layout claim extra top/bottom margin for the taller
    # title/legend/x-tick text, shrinking the actual plot area within this fixed figsize.
    fig.subplots_adjust(wspace=0.1, top=0.8074358974358973, bottom=0.20751586518158627)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary_csv', default=None, help='Single summary CSV to load directly, instead of globbing --results_root/--model.')
    p.add_argument('--results_root', default='results/subset/test_results')
    p.add_argument('--model', default='bert_singlelabel')
    p.add_argument('--chart_estimates_csv', default='results/subset_selection_chart_estimates_from_image.csv')
    p.add_argument('--no_chart_estimates', action='store_true', help='Skip the US/China chart-estimate rows entirely.')
    p.add_argument('--metric', default='acc', choices=list(METRICS), help='Metric to plot (chart estimates only cover acc).')
    p.add_argument('--include_combos', action='store_true', help='Include +bert combo methods (excluded by default).')
    p.add_argument('--domains', nargs='+', default=None, help='Domains to plot, in order (default: DOMAINS, plus any extra domains found in the summary data).')
    p.add_argument('--methods', nargs='+', default=None, help='Methods to plot, in order (default: METHODS, or METHOD_ORDER if --include_combos).')
    p.add_argument('--out_dir', default='plot/plots')
    p.add_argument('--out_prefix', default='subset_selection_target_comparison')
    return p.parse_args()


def main():
    args = parse_args()
    include_chart_estimates = not args.no_chart_estimates and args.metric == 'acc'
    parts = [chart_estimate_df(args.chart_estimates_csv)] if include_chart_estimates else []
    parts.append(summary_domain_df(args.summary_csv, args.results_root, args.model, args.include_combos, args.metric))
    df = pd.concat(parts, ignore_index=True)

    methods = args.methods or (METHOD_ORDER if args.include_combos else METHODS)
    methods = [m for m in methods if m in df['method'].unique()]
    domains = args.domains or DOMAINS + [d for d in sorted(df['domain'].unique()) if d not in DOMAINS]
    domains = [d for d in domains if d in df['domain'].unique()]

    n_seeds = sorted(df['n_seeds'].dropna().unique().astype(int)) if 'n_seeds' in df else []
    print(f'{len(methods)} methods with results: {methods}')
    print(f'Seed counts per (domain, budget, K, method) cell: {n_seeds}')

    for budget in sorted(df['budget'].unique()):
        bdf = df[df['budget'] == budget]
        bdomains = [d for d in domains if d in bdf['domain'].unique()]
        methods_tag = '-'.join(methods)
        plot(bdf, bdomains, methods, f'{args.out_dir}/{args.out_prefix}_{methods_tag}_{args.metric}_N{budget}.png', METRICS[args.metric])


if __name__ == '__main__':
    main()
