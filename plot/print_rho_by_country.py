#!/usr/bin/env python
"""Look up a single country's fixed-source-domain rho/R2 (per embedding/distance combo) from the
CSVs built by build_rho_by_src_domain_csv.py, and print the across-source-domain average
rho/R2 for every embedding/distance combo, without recomputing anything.

Requires plot/plots/rho_by_src_domain_detail.csv and plot/plots/rho_avg_by_src_domain.csv to
already exist -- build them first with:

  python plot/build_rho_by_src_domain_csv.py

Example:
  python plot/print_rho_by_country.py --country "United States"
  python plot/print_rho_by_country.py --src_domain 57
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def load_country_mapping(dataset):
    """Return {country_id: country_name} from the dataset's raw metadata CSV. Kept as a local,
    geopandas-free copy of plot_country_distance_map.load_country_mapping so this script doesn't
    pull in the (not-default-env) mapping dependencies just to resolve a country name."""
    meta_path = DATA_ROOT / dataset / "geoyfcc_all_metadata_before_cleaning.csv"
    df = pd.read_csv(meta_path, usecols=["country_id", "country"]).dropna()
    df["country_id"] = df["country_id"].astype(int)
    return df.groupby("country_id")["country"].first().to_dict()


def resolve_src_domain(dataset, country, src_domain):
    if src_domain is not None:
        return src_domain

    country_mapping = load_country_mapping(dataset)
    name_to_id = {name.lower(): domain_id for domain_id, name in country_mapping.items()}
    match = name_to_id.get(country.lower())
    if match is None:
        available = ", ".join(sorted(country_mapping.values()))
        print(f"[ERROR] Country '{country}' not found in {dataset}'s metadata. Available countries: {available}",
              file=sys.stderr)
        sys.exit(1)
    return match


def print_country_table(detail_df, src_domain, country_label):
    sub = detail_df[detail_df['src_domain_idx'] == src_domain]
    print(f"\n=== Fixed source domain = {country_label} (src_domain_idx={src_domain}) ===")
    if sub.empty:
        print("  (no rows -- this source domain had < min_pairs target domains for every combo, "
              "or was excluded as an outlier)")
        return

    sub = sub.sort_values(['embedding_type', 'distance_type', 'lambda_weight'])
    for _, row in sub.iterrows():
        label = (f"{row['embedding_type']}/{row['distance_type']}"
                 + (f"/{row['method']}" if pd.notna(row['method']) and row['method'] else "")
                 + (f"/lambda={row['lambda_weight']}" if pd.notna(row['lambda_weight']) else ""))
        print(f"  {label}: rho={row['rho']:.4f}, p_value={row['p_value']:.4g}, r2={row['r2']:.4f} "
              f"(n_pairs={row['n_pairs']})")


def print_embedding_type_averages(summary_df):
    print("\n=== Average rho / R2 across source domains, per embedding/distance combo ===")
    summary_df = summary_df.sort_values(['embedding_type', 'distance_type', 'lambda'])
    for _, row in summary_df.iterrows():
        label = (f"{row['embedding_type']}/{row['distance_type']}"
                 + (f"/{row['method']}" if isinstance(row['method'], str) and row['method'] else "")
                 + (f"/lambda={row['lambda']}" if pd.notna(row['lambda']) else ""))
        print(f"  {label}: rho_mean={row['rho_mean']:.4f} (std={row['rho_std']:.4f}), "
              f"r2_mean={row['r2_mean']:.4f} (std={row['r2_std']:.4f}) (n_src_domains={row['n_src_domains']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="geoyfcc_text")
    parser.add_argument("--detail_csv", default="plot/plots/rho_by_src_domain_detail.csv")
    parser.add_argument("--summary_csv", default="plot/plots/rho_avg_by_src_domain.csv")

    fixed_domain_group = parser.add_mutually_exclusive_group(required=True)
    fixed_domain_group.add_argument("--country", type=str, default=None,
                                     help="Country name as it appears in the dataset's metadata, e.g. 'United States'.")
    fixed_domain_group.add_argument("--src_domain", type=int, default=None,
                                     help="Domain index to use directly, skipping the country name lookup.")
    args = parser.parse_args()

    detail_path, summary_path = Path(args.detail_csv), Path(args.summary_csv)
    if not detail_path.exists() or not summary_path.exists():
        missing = [str(p) for p in (detail_path, summary_path) if not p.exists()]
        print(f"[ERROR] Missing {', '.join(missing)} -- build them first with:\n"
              f"  python plot/build_rho_by_src_domain_csv.py", file=sys.stderr)
        sys.exit(1)

    src_domain = resolve_src_domain(args.dataset, args.country, args.src_domain)
    country_label = args.country if args.country is not None else str(src_domain)

    detail_df = pd.read_csv(detail_path)
    summary_df = pd.read_csv(summary_path)

    print_country_table(detail_df, src_domain, country_label)
    print_embedding_type_averages(summary_df)


if __name__ == "__main__":
    main()
