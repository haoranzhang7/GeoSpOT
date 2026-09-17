#!/usr/bin/env python
"""Two choropleth world maps for one source country in GeoYFCC:

1. GeoSpOT distance from the source to every other country, normalized by the source's mean
   distance to all other countries (so 1.0 = "average" dissimilarity, <1 = more similar than
   average, >1 = less similar than average) and drawn on a diverging blue -> red colormap
   centered at 1.0 (smaller = more similar).
2. Relative transfer performance: zero-shot accuracy of the source-pretrained model on every
   other country, rescaled as % change from that target's own self-pair accuracy
   (trend_common.rescale_metric) and then min-max normalized to [0, 1] over the source's row
   (0 = worst target for this source, 1 = best), drawn on a diverging red -> blue colormap
   (higher = better).

Countries with no GeoYFCC data are left uncolored; the source country is outlined in black and
marked with a star.

Requires geopandas (not in the project's default env as of this writing -- see the `mapping`
env note in plot_country_distance_map.py).

Example:
  python plot/plot_country_metric_maps.py --embedding-type geoclip
  python plot/plot_country_metric_maps.py --embedding-type geoclip+bert --lambda-weight 0.5 \
      --source-domain-idx 57 --maps distance
"""
import argparse, sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_country_distance_map import (  # noqa: E402
    DEFAULT_SHAPEFILE, NAME_ALIASES, NO_DATA_COLOR, BORDER_COLOR, load_country_mapping,
)
from rho_csv_common import DATA_ROOT, LOCATION_EMBEDDING_TYPES, find_ot_sinkhorn_log_file, find_ot_lambda_combo_files  # noqa: E402
from trend_common import load_results  # noqa: E402

plt.rcParams['font.family'] = 'Times New Roman'

SOURCE_EDGE_COLOR = "black"
SOURCE_EDGE_WIDTH = 1.2
STAR_COLOR = "black"
STAR_SIZE = 140


def find_ot_distance_file(dataset, embedding_type, lambda_weight):
    """Return the Path to the sinkhorn_log OT distance matrix for embedding_type, resolving the
    f"{location}+bert" combined-embedding lambda sweep (see rho_csv_common.py) when needed."""
    if "+bert" in embedding_type:
        location_embedding_type = embedding_type.split("+")[0]
        if location_embedding_type not in LOCATION_EMBEDDING_TYPES:
            raise ValueError(f"Unknown combined embedding_type: {embedding_type}")
        if lambda_weight is None:
            raise ValueError(f"--lambda-weight is required for combined embedding_type={embedding_type}")
        combo_files = find_ot_lambda_combo_files(dataset, location_embedding_type)
        matches = [f for lam, f in combo_files if lam == lambda_weight]
        if not matches:
            available = [lam for lam, _ in combo_files]
            raise FileNotFoundError(
                f"No OT distance file for {embedding_type} at lambda={lambda_weight} "
                f"(available lambdas: {available})")
        return matches[0]

    distance_file, note = find_ot_sinkhorn_log_file(dataset, embedding_type)
    if distance_file is None:
        raise FileNotFoundError(f"No sinkhorn_log OT distance matrix found for embedding_type={embedding_type}")
    if note:
        print(f"[NOTE] {embedding_type}: {note} ({distance_file})")
    return distance_file


def aliased_name(country_mapping, domain_idx):
    return NAME_ALIASES.get(country_mapping[domain_idx], country_mapping[domain_idx])

def load_distance_series(dataset, embedding_type, lambda_weight, source_domain_idx, country_mapping):
    """Return a Series of GeoSpOT distance from source_domain_idx to every other domain,
    normalized by the source's mean distance so 1.0 = average dissimilarity, indexed by
    (shapefile-aliased) country name."""
    distance_file = find_ot_distance_file(dataset, embedding_type, lambda_weight)
    matrix = pd.read_csv(distance_file, index_col=0)
    matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)

    row = matrix.loc[str(source_domain_idx)].drop(labels=["all", str(source_domain_idx)], errors="ignore").dropna()
    normalized = row / row.mean()

    names = {int(idx): aliased_name(country_mapping, int(idx)) for idx in row.index}
    return pd.Series(normalized.values, index=[names[int(idx)] for idx in row.index], name="distance")


def load_performance_series(results_file, metric, source_domain_idx, country_mapping):
    """Return a Series of relative transfer performance (min-max normalized over the source's
    row, 0=worst target, 1=best target) from source_domain_idx to every other domain, indexed by
    (shapefile-aliased) country name."""
    results_df = load_results(results_file, metric, rescale_acc_flag=True)
    row = results_df[(results_df["src_domain_idx"] == source_domain_idx)
                      & (results_df["tgt_domain_idx"] != source_domain_idx)].dropna(subset=["acc_value"])
    if row.empty:
        raise ValueError(f"No results for src_domain_idx={source_domain_idx} in {results_file}")

    normalized = (row["acc_value"] - row["acc_value"].min()) / (row["acc_value"].max() - row["acc_value"].min())
    names = row["tgt_domain_idx"].map(lambda idx: aliased_name(country_mapping, int(idx)))
    return pd.Series(normalized.values, index=names, name="performance")


def plot_panel(ax, world, values, source_name, cmap, norm):
    world.plot(ax=ax, color=NO_DATA_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2)

    merged = world.merge(values.rename("value"), left_on="NAME", right_index=True, how="inner")
    merged.plot(ax=ax, column="value", cmap=cmap, norm=norm, edgecolor=BORDER_COLOR, linewidth=0.2)

    source_geom = world[world["NAME"] == source_name]
    source_geom.plot(ax=ax, facecolor="none", edgecolor=SOURCE_EDGE_COLOR, linewidth=SOURCE_EDGE_WIDTH)
    star_point = source_geom.geometry.representative_point().iloc[0]
    ax.scatter([star_point.x], [star_point.y], marker="*", s=STAR_SIZE, color=STAR_COLOR, zorder=5)

    ax.set_axis_off()
    ax.set_aspect("equal")


def load_world(shapefile_path):
    world = gpd.read_file(shapefile_path)
    world = world[world["CONTINENT"] != "Antarctica"]
    # Clip to just inside +/-180 before reprojecting -- otherwise antimeridian-crossing
    # geometries (Russia, Fiji, the US's Aleutian Islands) grow spurious edge-to-edge slivers
    # under Robinson.
    world = world.clip(box(-179.9999, -90, 179.9999, 90))
    return world.to_crs("+proj=robin")


def make_map_figure(world, values, source_name, scale_note, cbar_label, cmap, norm, cbar_ticks, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    plot_panel(ax, world, values, source_name, cmap, norm)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(f"{cbar_label}\n{scale_note}", fontsize=19, linespacing=1.6)
    cbar.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)
    return fig


def make_distance_figure(world, distances, source_name, figsize):
    # Plain linear normalize rather than a TwoSlopeNorm centered at 1.0: TwoSlopeNorm always puts
    # vcenter at the colorbar's midpoint regardless of how asymmetric [vmin, 1.0] vs. [1.0, vmax]
    # actually are, so the tick spacing on the two halves doesn't match the data spacing -- e.g. a
    # 0.20-wide left half and 0.15-wide right half would each get drawn as exactly half the
    # colorbar. A plain linear scale doesn't have that problem. RdYlBu maps its low end to red, so
    # with distances increasing from min (most similar) to max, red lands on "most similar" as
    # wanted without needing to reverse the colormap.
    norm = Normalize(vmin=distances.min(), vmax=distances.max())
    return make_map_figure(
        world, distances, source_name,
        scale_note="(smaller = more similar)",
        cbar_label="GeoSpOT distance (normalized)", cmap="RdYlBu", norm=norm,
        cbar_ticks=None, figsize=figsize)


def make_performance_figure(world, performance, source_name, figsize):
    # RdYlBu_r (reversed) so red lands on 1.0 (best performance), matching "red = most similar"
    # in the distance map above.
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    return make_map_figure(
        world, performance, source_name,
        scale_note="(higher = better)",
        cbar_label="Relative transfer performance", cmap="RdYlBu_r", norm=norm,
        cbar_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], figsize=figsize)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="geoyfcc_text")
    parser.add_argument("--embedding-type", type=str, required=True,
                         help="e.g. geoclip, satclip, geodesic, bert, or a combined "
                              "'{location}+bert' embedding (requires --lambda-weight)")
    parser.add_argument("--lambda-weight", type=float, default=None,
                         help="Required when --embedding-type is a combined '{location}+bert' embedding")
    parser.add_argument("--source-domain-idx", type=int, default=57,
                         help="Domain index of the source country (default: 57 = United States)")
    parser.add_argument("--results-file", type=str, default="results/geoyfcc/combined_zeroshot_results_geoyfcc_bert.csv")
    parser.add_argument("--metric", type=str, default="avg_test_acc")
    parser.add_argument("--shapefile", type=str, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--maps", choices=["distance", "performance", "both"], default="both")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.0, 4.6))
    parser.add_argument("--out-dir", type=str, default="plot/plots")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    country_mapping = load_country_mapping(args.dataset)
    source_name = aliased_name(country_mapping, args.source_domain_idx)
    world = load_world(args.shapefile)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_tag = args.embedding_type + (f"_lambda{args.lambda_weight}" if args.lambda_weight is not None else "")

    if args.maps in ("distance", "both"):
        distances = load_distance_series(args.dataset, args.embedding_type, args.lambda_weight,
                                          args.source_domain_idx, country_mapping)
        fig = make_distance_figure(world, distances, source_name, tuple(args.figsize))
        out_path = out_dir / f"country_distance_map_{embedding_tag}_source{args.source_domain_idx}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if args.maps in ("performance", "both"):
        performance = load_performance_series(args.results_file, args.metric, args.source_domain_idx, country_mapping)
        fig = make_performance_figure(world, performance, source_name, tuple(args.figsize))
        out_path = out_dir / f"country_performance_map_{args.dataset}_source{args.source_domain_idx}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
