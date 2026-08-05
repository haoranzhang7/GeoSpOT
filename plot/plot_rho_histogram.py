#!/usr/bin/env python
"""Plot the distribution of per-domain Spearman's rho values for one or more
(embedding_type, distance_type) combinations, as a bar histogram and/or KDE curve.

Example:
  python plot/plot_rho_histogram.py --embedding_types bert geoclip satclip --distance_types ot
"""
import os, argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from trend_common import EMBEDDING_LABELS
from trend_summary_common import load_rho_values

plt.rcParams['font.family'] = 'Times New Roman'

DISTANCE_LABELS = {'ot': 'OT', 'mmd': 'MMD', 'fid': 'FID', 'cosine': 'Cosine', 'geodesic': 'Geodesic'}
DISTANCE_COLORS = {'ot': 'firebrick', 'mmd': 'darkorange', 'fid': 'teal', 'cosine': 'slateblue', 'geodesic': 'goldenrod'}
AVERAGE_DISTANCES = {'cosine', 'geodesic'}  # simple pairwise averages, not true distributional distances
EMBEDDING_INDEPENDENT = {'geodesic'}        # raw geographic distance -- same value for every embedding
SIGN_FLIP = {'cosine'}                      # similarity, not distance -- correlates the opposite way

# One color per embedding, so color reads as "which embedding" at a glance rather than clashing
# across (embedding, distance) combos; distance/method is instead encoded by texture.
_TAB10 = plt.cm.tab10.colors
_PALETTE = [_TAB10[4], 'cornflowerblue', _TAB10[2], _TAB10[3]]
EMBEDDING_SHADES = {'bert': _PALETTE[0], 'geoclip': _PALETTE[1], 'satclip': _PALETTE[2]}
COLOR_ALPHA = 0.7

# Textures keyed by distance type, so the same method always draws with the same texture
# regardless of which embedding it's paired with.
DISTANCE_LINESTYLES = {
    'ot': '-', 'mmd': (0, (6, 2)), 'fid': (0, (1, 1)), 'cosine': (0, (3, 1, 1, 1)), 'geodesic': (0, (5, 5)),
}
DISTANCE_HATCHES = {'ot': '', 'mmd': '///', 'fid': '\\\\\\', 'cosine': 'xxx', 'geodesic': '...'}

FONT_SIZE_LABEL = 22
FONT_SIZE_TICK = 18
FONT_SIZE_LEGEND = 17


def load_series(plots_dir, embedding_types, distance_types, use_abs):
    combos = [(e, d) for d in distance_types
              for e in (embedding_types[:1] if d in EMBEDDING_INDEPENDENT else embedding_types)]
    multi_e, multi_d = len(embedding_types) > 1, len(distance_types) > 1

    def color_for(e, d):
        if d in EMBEDDING_INDEPENDENT or not multi_e:
            return DISTANCE_COLORS.get(d, 'steelblue')
        return EMBEDDING_SHADES.get(e, 'steelblue')

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
        series.append((label, color_for(e, d), DISTANCE_LINESTYLES.get(d, '-'),
                       DISTANCE_HATCHES.get(d, ''), values, e, d))

    if not series:
        raise ValueError("No rho summary CSVs found for the requested embedding_types/distance_types.")
    return series


def method_label(d):
    return ('Avg. ' if d in AVERAGE_DISTANCES else '') + DISTANCE_LABELS.get(d, d.upper())


def add_legend(ax, series, kind):
    """Single legend, laid out as two side-by-side columns: method (texture) on the left,
    embedding (color) on the right. Falls back to a plain per-series legend when only one of
    method/embedding actually varies, since there'd be nothing to split."""
    method_handles, seen_d = [], set()
    embed_handles, seen_e = [], set()
    for _, color, linestyle, hatch, _, e, d in series:
        if d not in seen_d:
            seen_d.add(d)
            if kind == 'bar':
                method_handles.append(Patch(facecolor='white', edgecolor='black', hatch=hatch, label=method_label(d)))
            else:
                method_handles.append(Line2D([], [], color='black', linewidth=2.75, linestyle=linestyle, label=method_label(d)))
        if d not in EMBEDDING_INDEPENDENT and e not in seen_e:
            seen_e.add(e)
            embed_handles.append(Line2D([], [], color=color, linewidth=2.75, label=EMBEDDING_LABELS.get(e, e)))

    if len(method_handles) <= 1 or len(embed_handles) <= 1:
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        return

    # matplotlib fills a multi-column legend column-major, so padding each group up to the same
    # row count and concatenating (not interleaving) keeps method in column 1, embedding in column 2.
    blank = Line2D([], [], color='none', label='')
    n_rows = max(len(method_handles), len(embed_handles))
    method_handles += [blank] * (n_rows - len(method_handles))
    embed_handles += [blank] * (n_rows - len(embed_handles))
    ax.legend(handles=method_handles + embed_handles, ncol=2, fontsize=FONT_SIZE_LEGEND,
              loc='upper right', columnspacing=1.2, handletextpad=0.6)


def plot_histogram(series, out_path, kind, use_abs, bins=15, density=False, figsize=(9, 6)):
    rho_range = (0.0, 1.0) if use_abs else (-1.0, 1.0)
    all_values = np.concatenate([v for _, _, _, _, v, _, _ in series])
    lo, hi = all_values.min(), all_values.max()
    pad = (hi - lo) * 0.1 or 0.05
    xlim = max(rho_range[0], lo - pad), min(rho_range[1], hi + pad)

    fig, ax = plt.subplots(figsize=figsize)
    if kind == 'bar':
        bin_edges = np.linspace(*xlim, bins + 1)
        for label, color, _, hatch, values, _, _ in series:
            ax.hist(values, bins=bin_edges, density=density, alpha=COLOR_ALPHA, label=label,
                     color=color, edgecolor='black', linewidth=0.8, hatch=hatch)
        y_label = 'Density' if density else 'Frequency'
    else:
        for label, color, linestyle, _, values, _, _ in series:
            sns.kdeplot(values, ax=ax, fill=False, linewidth=2.75, color=color, alpha=COLOR_ALPHA,
                        label=label, clip=rho_range, linestyle=linestyle)
        y_label = 'Density'

    if xlim[0] <= 0 <= xlim[1]:
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"Spearman's $|\rho|$" if use_abs else r"Spearman's $\rho$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    ax.tick_params('both', labelsize=FONT_SIZE_TICK)
    add_legend(ax, series, kind)

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
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 7))
    args = parser.parse_args()
    main(args)
