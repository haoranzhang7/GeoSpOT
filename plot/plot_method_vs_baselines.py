"""Compare subset-selection methods against the random baseline, split out by K.

Reads a CSV (domain, budget, K, method, metric, value_est, std_est) -- e.g. the one
written by build_subset_selection_csv.py -- and, per domain, draws a dot plot: one
dodged point per method per K (no bars, no connecting lines, so no K-trend is
implied). Random is plotted the same way as the other methods (per-K, not averaged)
since it varies by K same as everything else. If a 'global' row (no-subsetting
baseline, no K) is present for a domain, it is drawn as a horizontal reference band;
otherwise it is silently omitted.

All K values always appear together on one plot. Saves one plot per available
(budget, metric) pair, with both appended to the filename (e.g. foo.png ->
foo_N2000_test_acc.png, foo_N5000_test_acc.png, ...).
"""

import os, argparse

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 34,
    'axes.labelsize': 34,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 30,
    'axes.linewidth': 2.2,
    'xtick.major.width': 2.2,
    'ytick.major.width': 2.2,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
    'lines.dash_capstyle': 'round',
})

METHOD_ORDER = ['random', 'bert', 'geoclip', 'satclip', 'geodesic', 'geoclip+bert', 'satclip+bert', 'geodesic+bert']
METHOD_LABELS = {
    'random': 'Random', 'bert': 'BERT', 'geoclip': 'GeoCLIP', 'satclip': 'SatCLIP', 'geodesic': 'Geodesic',
    'geoclip+bert': 'GeoCLIP+BERT', 'satclip+bert': 'SatCLIP+BERT', 'geodesic+bert': 'Geodesic+BERT',
}

# random is the neutral baseline (kept black, as in the original 4-method version); the
# other 7 use a size-8 validated categorical palette (see dataviz skill), taking its first
# 7 slots in order -- dropping the palette's 8th (red) slot keeps every adjacent pair validated.
METHOD_COLORS = {
    'random': 'black',
    'bert': '#2a78d6',
    'geoclip': '#eb6834',
    'satclip': '#1baf7a',
    'geodesic': '#eda100',
    'geoclip+bert': '#e87ba4',
    'satclip+bert': '#008300',
    'geodesic+bert': '#4a3aa7',
}

K_ORDER = ['1', '2', '5']
K_MARKERS = {'1': 'o', '2': 's', '5': '^'}
K_DODGE = {'1': -0.22, '2': 0.0, '5': 0.22}

GLOBAL_COLOR = 'grey'


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['K'] = df['K'].astype(str)
    df['budget'] = df['budget'].astype(int)
    return df


METRIC_YLABELS = {
    'test_acc': 'Test Accuracy on Target domain',
    'test_top3_acc': 'Test Top-3 Accuracy on Target domain',
    'test_top5_acc': 'Test Top-5 Accuracy on Target domain',
}


def plot(df, domains, metric, out_path):
    # Each subplot needs enough width for len(METHOD_ORDER) rotated tick labels; with 8
    # methods (vs. the original 4) a plain 8in-per-domain panel is too narrow and forces
    # the top legend to overflow into the y-axis label, so widen per panel accordingly.
    panel_width = max(8, 1.7 * len(METHOD_ORDER))
    fig, axes = plt.subplots(1, len(domains), figsize=(panel_width * len(domains), 9), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    x = np.arange(len(METHOD_ORDER))

    for ax, domain in zip(axes, domains):
        domain_df = df[df['domain'] == domain]

        for method_idx, method in enumerate(METHOD_ORDER):
            for k in K_ORDER:
                matches = domain_df[(domain_df['method'] == method) & (domain_df['K'] == k)]
                if matches.empty:
                    continue  # e.g. an in-progress eval grid that hasn't covered this (method, K) yet
                row = matches.iloc[0]
                _, _, (barlinecol,) = ax.errorbar(
                    x[method_idx] + K_DODGE[k], row['value_est'], yerr=row['std_est'],
                    fmt=K_MARKERS[k], color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method],
                    elinewidth=3.5, capsize=8, capthick=3.5, markersize=22,
                    markeredgecolor='white', markeredgewidth=2, zorder=3)
                barlinecol.set_capstyle('round')

        global_rows = domain_df[domain_df['method'] == 'global']
        if len(global_rows):
            global_row = global_rows.iloc[0]
            ax.axhspan(global_row['value_est'] - global_row['std_est'], global_row['value_est'] + global_row['std_est'],
                       color=GLOBAL_COLOR, alpha=0.15, zorder=1)
            ax.axhline(global_row['value_est'], color=GLOBAL_COLOR, linestyle='--', linewidth=3, zorder=2, label=r'global ($K=\infty$)')

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in METHOD_ORDER], rotation=30, ha='right')
        ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
        ax.set_title(f'Target domain: {domain}')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.tick_params(axis='both', which='major', pad=8)

    axes[0].set_ylabel(METRIC_YLABELS.get(metric, metric))

    k_handles = [plt.Line2D([0], [0], marker=K_MARKERS[k], linestyle='None', markersize=22, label=rf'$K={k}$',
                             markerfacecolor='none', markeredgecolor='dimgray', markeredgewidth=3.5) for k in K_ORDER]
    baseline_handles, baseline_labels = axes[-1].get_legend_handles_labels()
    all_handles = k_handles + baseline_handles
    fig.legend(handles=all_handles, labels=[h.get_label() for h in k_handles] + baseline_labels,
               loc='upper center', ncol=min(5, len(all_handles)), bbox_to_anchor=(0.5, 1.1), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='plot/plots/subset_selection_results.csv')
    parser.add_argument('--out', default='plot/plots/subset_selection_method_vs_baselines.png')
    parser.add_argument('--domains', nargs='+', default=None, help='Subset of domains to plot, e.g. --domains China US')
    parser.add_argument('--metrics', nargs='+', default=None, help='Subset of metrics to plot, e.g. --metrics test_acc. Defaults to every metric in the CSV.')
    args = parser.parse_args()

    df = load_data(args.csv)
    domains = args.domains if args.domains else [d for d in df['domain'].unique()]
    metrics = args.metrics if args.metrics else [m for m in df['metric'].unique()]

    root, ext = os.path.splitext(args.out)
    for metric in metrics:
        metric_df = df[df['metric'] == metric]
        for budget in sorted(metric_df['budget'].unique()):
            plot(metric_df[metric_df['budget'] == budget], domains, metric, f'{root}_N{budget}_{metric}{ext}')


if __name__ == '__main__':
    main()
