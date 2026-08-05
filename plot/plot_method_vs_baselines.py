"""Compare subset-selection methods against random/global baselines, split out by K.

Reads the chart-estimated CSV (domain, budget, K, method, value_est, std_est) and,
per domain, draws a dot plot: one dodged point per method per K (no bars, no
connecting lines, so no K-trend is implied). Random is plotted the same way as the
other methods (per-K, not averaged) since it varies by K same as everything else;
global is shown as a horizontal reference band since it has no K. BERT is excluded
-- the comparison of interest is GeoCLIP/SatCLIP vs. Geodesic vs. random vs. global.

All K values always appear together on one plot. Saves one plot per available
budget, with the budget appended to the filename (e.g. foo.png -> foo_N2000.png,
foo_N5000.png, ...).
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

METHOD_ORDER = ['random', 'Geodesic', 'SatCLIP', 'GeoCLIP']
METHOD_LABELS = {'random': 'Random'}

METHOD_COLORS = {
    'random': 'black',
    'GeoCLIP': 'cornflowerblue',
    'SatCLIP': 'green',
    'Geodesic': 'darkorange',
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


def plot(df, domains, out_path):
    fig, axes = plt.subplots(1, len(domains), figsize=(8 * len(domains), 9), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    x = np.arange(len(METHOD_ORDER))

    for ax, domain in zip(axes, domains):
        domain_df = df[df['domain'] == domain]

        for method_idx, method in enumerate(METHOD_ORDER):
            for k in K_ORDER:
                row = domain_df[(domain_df['method'] == method) & (domain_df['K'] == k)].iloc[0]
                _, _, (barlinecol,) = ax.errorbar(
                    x[method_idx] + K_DODGE[k], row['value_est'], yerr=row['std_est'],
                    fmt=K_MARKERS[k], color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method],
                    elinewidth=3.5, capsize=8, capthick=3.5, markersize=22,
                    markeredgecolor='white', markeredgewidth=2, zorder=3)
                barlinecol.set_capstyle('round')

        global_row = domain_df[domain_df['method'] == 'global'].iloc[0]
        ax.axhspan(global_row['value_est'] - global_row['std_est'], global_row['value_est'] + global_row['std_est'],
                   color=GLOBAL_COLOR, alpha=0.15, zorder=1)
        ax.axhline(global_row['value_est'], color=GLOBAL_COLOR, linestyle='--', linewidth=3, zorder=2, label=r'global ($K=\infty$)')

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in METHOD_ORDER], rotation=30, ha='right')
        ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
        ax.set_title(f'Target domain: {domain}')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.tick_params(axis='both', which='major', pad=8)

    axes[0].set_ylabel('Test Accuracy on Target domain')

    k_handles = [plt.Line2D([0], [0], marker=K_MARKERS[k], linestyle='None', markersize=22, label=rf'$K={k}$',
                             markerfacecolor='none', markeredgecolor='dimgray', markeredgewidth=3.5) for k in K_ORDER]
    baseline_handles, baseline_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles=k_handles + baseline_handles, labels=[h.get_label() for h in k_handles] + baseline_labels,
               loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.1), frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='plot/plots/subset_selection_chart_estimates_from_image.csv')
    parser.add_argument('--out', default='plot/plots/subset_selection_method_vs_baselines.png')
    parser.add_argument('--domains', nargs='+', default=None, help='Subset of domains to plot, e.g. --domains China US')
    args = parser.parse_args()

    df = load_data(args.csv)
    domains = args.domains if args.domains else [d for d in df['domain'].unique()]

    root, ext = os.path.splitext(args.out)
    for budget in sorted(df['budget'].unique()):
        plot(df[df['budget'] == budget], domains, f'{root}_N{budget}{ext}')


if __name__ == '__main__':
    main()
