#!/usr/bin/env python
"""
Compute pairwise average geodesic (great-circle) distance over domain coordinates: for a domain
pair (s, t), avg_geo_dist(s, t) = mean over all (i in s, j in t) of haversine(coord_i, coord_j).

Computed in batches (never materializing the full N_s x N_t distance matrix at once) so it scales
to domains with tens of thousands of samples. Geodesic distance only depends on lat/lon
coordinates, not on any embedding type, so this produces a single matrix per dataset.

Saves an N x N matrix CSV (rows=src, cols=tgt), matching the format produced by
mmd_distance.py / fid_distance.py / cosine_similarity_distance.py.

Example:
  python src/distances/geodesic_distance.py --dataset geoyfcc_text
"""

import argparse, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distances.utils import DATA_ROOT, load_coordinates_and_domains, get_embs
from src.distances.cost_matrix import haversine_distance


def avg_geodesic_distance(src_coords, tgt_coords, batch_size=5000):
    n_src, n_tgt = len(src_coords), len(tgt_coords)
    total = 0.0
    for i in range(0, n_src, batch_size):
        batch_i = src_coords[i:i + batch_size]
        for j in range(0, n_tgt, batch_size):
            batch_j = tgt_coords[j:j + batch_size]
            total += haversine_distance(batch_i, batch_j).sum().item()
    return total / (n_src * n_tgt)


def compute_avg_geodesic_matrix(coords, domains, active, n, max_samples, device, batch_size):
    matrix = np.full((n, n), np.nan)
    domain_coords = {idx: get_embs(coords, domains, idx, max_samples, device) for idx in active}

    for src in tqdm(active, desc="src domain", unit="domain"):
        src_coords = domain_coords[src]
        for tgt in active:
            if tgt == src:
                continue
            matrix[src, tgt] = avg_geodesic_distance(src_coords, domain_coords[tgt], batch_size)
    return matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geoyfcc_text", choices=["geoyfcc_text", "fmow", "geoyfcc_image"])
    p.add_argument("--total-domains", type=int, default=62)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = args.total_domains
    coords, domains = load_coordinates_and_domains(args.dataset)
    active = [i for i in range(n) if (domains == i).sum() > 0]
    print(f"{len(active)} non-empty domains, {len(active) * (len(active) - 1)} pairs to compute")

    matrix = compute_avg_geodesic_matrix(coords, domains, active, n, args.max_samples, device, args.batch_size)

    out_dir = DATA_ROOT / args.dataset / "distances" / "geodesic_distance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "geodesic_avg_distance.csv"
    pd.DataFrame(matrix, index=range(n), columns=range(n)).to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
