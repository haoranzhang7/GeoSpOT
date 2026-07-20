#!/usr/bin/env python
"""Plot the distribution of per-domain Spearman's rho values for one or more
(embedding_type, distance_type) combinations, as a bar histogram and/or KDE curve.

Example:
  python plot/plot_rho_histogram.py --embedding_types bert geoclip satclip --distance_types ot
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from trend_common import EMBEDDING_LABELS, EMBEDDING_COLORS
from trend_summary_common import load_rho_values

plt.rcParams['font.family'] = 'Times New Roman'

DISTANCE_LABELS = {'ot': 'OT', 'mmd': 'MMD', 'fid': 'FID', 'cosine': 'Cosine', 'geodesic': 'Geodesic'}
DISTANCE_COLORS = {'ot': 'firebrick', 'mmd': 'darkorange', 'fid': 'teal', 'cosine': 'slateblue', 'geodesic': 'goldenrod'}
AVERAGE_DISTANCES = {'cosine', 'geodesic'}  # simple pairwise averages, not true distributional distances
EMBEDDING_INDEPENDENT = {'geodesic'}        # raw geographic distance -- same value for every embedding
SIGN_FLIP = {'cosine'}                      # similarity, not distance -- correlates the opposite way


def load_series(plots_dir, embedding_types, distance_types, use_abs):
    combos = [(e, d) for d in distance_types
              for e in (embedding_types[:1] if d in EMBEDDING_INDEPENDENT else embedding_types)]
    multi_e, multi_d = len(embedding_types) > 1, len(distance_types) > 1
    if multi_d and not multi_e:
        colors = {c: DISTANCE_COLORS.get(c[1], 'steelblue') for c in combos}
    elif multi_e and not multi_d:
        colors = {c: EMBEDDING_COLORS.get(c[0], 'steelblue') for c in combos}
    else:
        palette = sns.color_palette('tab20', len(combos))
        colors = {c: palette[i] for i, c in enumerate(combos)}

    series = []
    for e, d in combos:
        values = np.array([])
        for candidate in (embedding_types if d in EMBEDDING_INDEPENDENT else [e]):
            values = load_rho_values(plots_dir, candidate, d, ["src"])
            if len(values):
                break
        if len(values) == 0:
            continue
        if d in SIGN_FLIP:
            values = -values
        if use_abs:
            values = np.abs(values)
        label = ('Avg. ' if d in AVERAGE_DISTANCES else '') + DISTANCE_LABELS.get(d, d.upper())
        if d not in EMBEDDING_INDEPENDENT:
            label += f"-{EMBEDDING_LABELS.get(e, e)}"
        print(f"Loaded {len(values)} rho values for {label} "
              f"(mean={values.mean():.4f}, median={np.median(values):.4f})")
        series.append((label, colors[(e, d)], values))

    if not series:
        raise ValueError("No rho summary CSVs found for the requested embedding_types/distance_types.")
    return series


def plot_histogram(series, out_path, kind, use_abs, bins=15, density=False, figsize=(9, 6)):
    rho_range = (0.0, 1.0) if use_abs else (-1.0, 1.0)
    all_values = np.concatenate([v for _, _, v in series])
    lo, hi = all_values.min(), all_values.max()
    pad = (hi - lo) * 0.1 or 0.05
    xlim = max(rho_range[0], lo - pad), min(rho_range[1], hi + pad)

    fig, ax = plt.subplots(figsize=figsize)
    if kind == 'bar':
        bin_edges = np.linspace(*xlim, bins + 1)
        for label, color, values in series:
            ax.hist(values, bins=bin_edges, density=density, alpha=0.55, label=label,
                     color=color, edgecolor='black', linewidth=0.8)
        y_label = 'Density' if density else 'Frequency'
    else:
        for label, color, values in series:
            sns.kdeplot(values, ax=ax, fill=False, linewidth=2.5, color=color, label=label, clip=rho_range)
        y_label = 'Density'

    if xlim[0] <= 0 <= xlim[1]:
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"Spearman's $|\rho|$" if use_abs else r"Spearman's $\rho$", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params('both', labelsize=14)
    ax.legend(fontsize=13)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {kind} histogram to {out_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    series = load_series(args.plots_dir, args.embedding_types, args.distance_types, args.abs_rho)

    combo = f"{'-'.join(args.embedding_types)}_{'-'.join(args.distance_types)}"
    suffix = "_abs" if args.abs_rho else ""
    bar_name = f"{args.output_name}_bar{suffix}" if args.output_name else f"rho_hist_bar_{combo}{suffix}"
    density_name = f"{args.output_name}_density{suffix}" if args.output_name else f"rho_hist_density_{combo}{suffix}"

    if args.plot_type in ('bar', 'both'):
        plot_histogram(series, os.path.join(args.output_dir, f"{bar_name}.png"),
                        'bar', args.abs_rho, args.bins, args.density, tuple(args.figsize))
    if args.plot_type in ('density', 'both'):
        plot_histogram(series, os.path.join(args.output_dir, f"{density_name}.png"),
                        'density', args.abs_rho, figsize=tuple(args.figsize))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots_dir', default='plot/plots')
    parser.add_argument('--embedding_types', nargs='+', default=['bert', 'geoclip', 'satclip'])
    parser.add_argument('--distance_types', nargs='+', default=['ot'])
    parser.add_argument('--output_dir', default='plot/plots/rho_histograms')
    parser.add_argument('--output_name', default=None)
    parser.add_argument('--plot_type', choices=['bar', 'density', 'both'], default='both')
    parser.add_argument('--bins', type=int, default=15)
    parser.add_argument('--density', action='store_true')
    parser.add_argument('--abs_rho', action='store_true')
    parser.add_argument('--figsize', type=float, nargs=2, default=(9, 6))
    args = parser.parse_args()
    main(args)
