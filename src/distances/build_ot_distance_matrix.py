#!/usr/bin/env python
"""
Consolidate the per-source-domain OT distance CSVs (produced by ot_distance.py with k=1)
into a single N x N distance matrix CSV (rows=src, cols=tgt), matching the format
produced by mmd_distance.py / fid_distance.py so it can be used with plot_trends_by_domain.py.

Example:
  python src/distances/build_ot_distance_matrix.py --dataset geoyfcc_text --embedding-type bert
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distances.utils import DATA_ROOT


def build_ot_matrix(source_dir, embedding_type, k, config_suffix, n):
    matrix = np.full((n, n), np.nan)
    for src in range(n):
        src_file = source_dir / f"distances_source{src}_k{k}_{embedding_type}_all_combinations_{config_suffix}.csv"
        if not src_file.exists():
            print(f"[WARNING] Missing {src_file}, leaving row {src} as NaN")
            continue
        rows = pd.read_csv(src_file)
        for _, row in rows.iterrows():
            tgt = int(row["tgt_domain_1_idx"])
            if tgt == src:
                continue
            matrix[src, tgt] = row["distance"]
    return matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geoyfcc_text", choices=["geoyfcc_text", "fmow", "geoyfcc_image"])
    p.add_argument("--total-domains", type=int, default=62)
    p.add_argument("--embedding-type", default="bert")
    p.add_argument("--k", type=int, default=1, help="Only k=1 (single target domain) can be reshaped into an N x N matrix.")
    p.add_argument("--method", default="sinkhorn_log")
    p.add_argument("--reg-e", type=float, default=0.01)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--metric", default="euclidean", help="Distance metric used for non-geodesic embeddings.")
    p.add_argument("--normalize-cost", default="max_per_domain")
    args = p.parse_args()

    if args.k != 1:
        raise ValueError("build_ot_distance_matrix.py only supports k=1 (one target domain per source file).")

    metric = "geodesic" if args.embedding_type == "geodesic" else args.metric
    config_suffix = (f"method_{args.method}_reg_{args.reg_e}_iter_{args.max_iter}_"
                      f"metric_{metric}_norm_{args.normalize_cost}")

    source_dir = DATA_ROOT / args.dataset / "distances" / "ot_distance"
    n = args.total_domains
    matrix = build_ot_matrix(source_dir, args.embedding_type, args.k, config_suffix, n)

    out = source_dir / f"ot_{args.embedding_type}_k{args.k}_{config_suffix}.csv"
    pd.DataFrame(matrix, index=range(n), columns=range(n)).to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
