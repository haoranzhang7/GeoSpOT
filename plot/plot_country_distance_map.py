#!/usr/bin/env python
"""Choropleth world maps of GeoSpOT (OT) distance from one source country to every other
country in GeoYFCC, one panel per location embedding (default: GeoCLIP, SatCLIP).

The source country is hatched (its distance-to-self isn't meaningful); countries with no
GeoYFCC data are left uncolored; every other country is shaded by its OT distance to the
source, normalized per-panel to [most similar -> least similar] so the two panels share one
colorbar even though their raw distance scales differ.

Requires geopandas + seaborn (not in the project's default env as of this writing -- the
`mapping` conda env has both, e.g. `conda run -n mapping python plot/plot_country_distance_map.py`).

Example:
  python plot/plot_country_distance_map.py
  python plot/plot_country_distance_map.py --embedding-types geoclip satclip --source-domain-idx 57
"""
import argparse, sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rho_csv_common import DATA_ROOT, find_ot_sinkhorn_log_file  # noqa: E402
from trend_common import EMBEDDING_LABELS  # noqa: E402

plt.rcParams['font.family'] = 'Times New Roman'

# Natural Earth 1:10m admin-0 shapefile -- needs this resolution (not the 1:110m one) because
# some GeoYFCC countries (Hong Kong, Singapore) are dropped/merged at 1:110m.
DEFAULT_SHAPEFILE = "/home/libe2152/projects/optimizedsampling/0_data/boundaries/world/ne_10m_admin_0_countries.shp"

# GeoYFCC country names that don't match the shapefile's NAME field verbatim.
NAME_ALIASES = {
    "United States": "United States of America",
    "The Bahamas": "Bahamas",
    "Czech Republic": "Czechia",
}

NO_DATA_COLOR = "#e6e6e6"
BORDER_COLOR = "white"


def load_country_mapping(dataset):
    """Return {country_id: country_name} from the dataset's raw metadata CSV."""
    meta_path = DATA_ROOT / dataset / "geoyfcc_all_metadata_before_cleaning.csv"
    df = pd.read_csv(meta_path, usecols=["country_id", "country"]).dropna()
    df["country_id"] = df["country_id"].astype(int)
    return df.groupby("country_id")["country"].first().to_dict()


def load_source_distances(dataset, embedding_type, source_domain_idx, country_mapping):
    """Return a Series of OT distance from source_domain_idx to every other domain, indexed by
    (shapefile-aliased) country name."""
    distance_file, note = find_ot_sinkhorn_log_file(dataset, embedding_type)
    if distance_file is None:
        raise FileNotFoundError(f"No sinkhorn_log OT distance matrix found for embedding_type={embedding_type}")
    if note:
        print(f"[NOTE] {embedding_type}: {note} ({distance_file})")

    matrix = pd.read_csv(distance_file, index_col=0)
    matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)

    row = matrix.loc[str(source_domain_idx)].drop(labels=["all", str(source_domain_idx)], errors="ignore").dropna()

    names = {int(idx): NAME_ALIASES.get(country_mapping[int(idx)], country_mapping[int(idx)]) for idx in row.index}
    return pd.Series(row.values, index=[names[int(idx)] for idx in row.index], name="distance")


def plot_panel(ax, world, distances, source_name, cmap):
    world.plot(ax=ax, color=NO_DATA_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2)

    merged = world.merge(distances.rename("distance"), left_on="NAME", right_index=True, how="inner")
    norm = Normalize(vmin=distances.min(), vmax=distances.max())
    merged.plot(ax=ax, column="distance", cmap=cmap, norm=norm, edgecolor=BORDER_COLOR, linewidth=0.2)

    source_geom = world[world["NAME"] == source_name]
    source_geom.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.4, hatch="....")

    ax.set_axis_off()
    ax.set_aspect("equal")


def make_figure(dataset, embedding_types, source_domain_idx, shapefile_path, cmap_name, figsize):
    country_mapping = load_country_mapping(dataset)
    source_name = NAME_ALIASES.get(country_mapping[source_domain_idx], country_mapping[source_domain_idx])

    world = gpd.read_file(shapefile_path)
    world = world[world["CONTINENT"] != "Antarctica"]
    # Clip to just inside +/-180 before reprojecting -- otherwise antimeridian-crossing
    # geometries (Russia, Fiji, the US's Aleutian Islands) grow spurious edge-to-edge slivers
    # under Robinson.
    world = world.clip(box(-179.9999, -90, 179.9999, 90))
    world = world.to_crs("+proj=robin")
    cmap = sns.color_palette(cmap_name, as_cmap=True)

    fig, axes = plt.subplots(len(embedding_types), 1, figsize=figsize)
    if len(embedding_types) == 1:
        axes = [axes]

    for ax, embedding_type in zip(axes, embedding_types):
        distances = load_source_distances(dataset, embedding_type, source_domain_idx, country_mapping)
        plot_panel(ax, world, distances, source_name, cmap)
        label = EMBEDDING_LABELS.get(embedding_type, embedding_type)
        ax.set_title(f"GeoSpOT Distance ({label})", fontsize=9, pad=4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.1, hspace=0.15)

    cbar_ax = fig.add_axes([0.3, 0.045, 0.4, 0.018])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.4)
    cbar_ax.text(-0.02, 0.5, "most similar", transform=cbar_ax.transAxes, ha="right", va="center", fontsize=10)
    cbar_ax.text(1.02, 0.5, "least similar", transform=cbar_ax.transAxes, ha="left", va="center", fontsize=10)

    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="geoyfcc_text")
    parser.add_argument("--embedding-types", type=str, nargs="+", default=["geoclip", "satclip"])
    parser.add_argument("--source-domain-idx", type=int, default=57, help="Domain index of the source country (default: 57 = United States)")
    parser.add_argument("--shapefile", type=str, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--cmap", type=str, default="flare", help="Any seaborn/matplotlib colormap name")
    parser.add_argument("--figsize", type=float, nargs=2, default=(4.2, 4.6))
    parser.add_argument("--out", type=str, default="plot/plots/country_distance_map.png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    fig = make_figure(args.dataset, args.embedding_types, args.source_domain_idx,
                       args.shapefile, args.cmap, tuple(args.figsize))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
