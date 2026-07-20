#!/usr/bin/env python
"""
Aggregate the by_domain trend summary CSVs (produced by plot_trends_by_domain.py, one row per
fixed domain with rho/p_value/r2/n_pairs) into a single table: one row per
(embedding_type, distance_type, direction), reporting the mean/std of Spearman's rho and R^2
across domains.

Example:
  python plot/summarize_by_domain_trends.py --plots_dir plot/plots --output_file plot/plots/by_domain_trend_summary.csv
"""

import argparse
import os

import pandas as pd

from trend_summary_common import find_trend_csv

EMBEDDING_TYPES = ['bert', 'geoclip', 'satclip']
DISTANCE_TYPES = ['cosine', 'mmd', 'ot', 'fid', 'geodesic']
DIRECTIONS = ['src', 'tgt']


def main(args):
    rows = []
    for embedding_type in args.embedding_types:
        for distance_type in args.distance_types:
            for direction in DIRECTIONS:
                csv_path = find_trend_csv(args.plots_dir, embedding_type, distance_type, direction)
                if csv_path is None:
                    print(f"[WARNING] No by_{direction}_domain summary found for "
                          f"embedding={embedding_type}, distance_type={distance_type}, skipping")
                    continue

                df = pd.read_csv(csv_path)
                rows.append({
                    'embedding_type': embedding_type,
                    'distance_type': distance_type,
                    'direction': direction,
                    'rho_mean': df['rho'].mean(),
                    'rho_std': df['rho'].std(),
                    'r2_mean': df['r2'].mean(),
                    'r2_std': df['r2'].std(),
                    'n_domains': len(df),
                })

    summary_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    summary_df.to_csv(args.output_file, index=False)
    print(f"Saved {len(summary_df)} rows to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots_dir', type=str, default='plot/plots',
                         help="Root directory containing per-embedding subfolders with a by_domain/ folder.")
    parser.add_argument('--embedding_types', type=str, nargs='+', default=EMBEDDING_TYPES)
    parser.add_argument('--distance_types', type=str, nargs='+', default=DISTANCE_TYPES)
    parser.add_argument('--output_file', type=str, default='plot/plots/by_domain_trend_summary.csv')
    args = parser.parse_args()

    main(args)
