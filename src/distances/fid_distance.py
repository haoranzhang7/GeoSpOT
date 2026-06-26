#!/usr/bin/env python
"""
Compute pairwise FID distances over domain embeddings. Saves an N×N matrix CSV (rows=src, cols=tgt).

Example:
  python src/distances/fid_distance.py --dataset geoyfcc_text --embedding-type satclip
"""

import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from scipy import linalg
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distances.utils import DATA_ROOT, load_embeddings_and_domains, get_embs
from src.distances.cost_matrix import (compute_cost_matrix, normalize_cost_matrix,
                                        load_cost_constants, compute_and_cache_cost_constants)


def stats(features):
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def mean_dist(mu1, mu2, metric="cosine", normalize_args=None):
    """Distance between two domain mean vectors using the given metric (not squared)."""
    t1 = torch.as_tensor(mu1, dtype=torch.float32).reshape(1, -1)
    t2 = torch.as_tensor(mu2, dtype=torch.float32).reshape(1, -1)
    dist = compute_cost_matrix(t1, t2, metric)
    if normalize_args is not None:
        dist = normalize_cost_matrix(dist, normalize_args)
    return dist.item()


def fid(mu1, sigma1, mu2, sigma2, metric="cosine", normalize_args=None):
    mean_term = mean_dist(mu1, mu2, metric, normalize_args) ** 2
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return mean_term + np.trace(sigma1 + sigma2 - 2 * covmean)


def calculate_fid(embeddings1, embeddings2, metric="cosine", normalize_args=None):
    mu1, s1 = stats(embeddings1)
    mu2, s2 = stats(embeddings2)
    return fid(mu1, s1, mu2, s2, metric=metric, normalize_args=normalize_args)


def compute_fid_matrix(embeddings, domains, active, n, max_samples, device, metric="cosine", normalize_args=None):
    matrix = np.full((n, n), np.nan)
    domain_stats = {}
    for idx in active:
        feats = get_embs(embeddings, domains, idx, max_samples, device).cpu().numpy()
        domain_stats[idx] = stats(feats)

    for src in tqdm(active, desc="src domain", unit="domain"):
        src_mu, src_sig = domain_stats[src]
        for tgt in active:
            if tgt == src:
                continue
            tgt_mu, tgt_sig = domain_stats[tgt]
            matrix[src, tgt] = float(fid(src_mu, src_sig, tgt_mu, tgt_sig, metric=metric, normalize_args=normalize_args))
    return matrix


def build_normalize_args(args, embeddings):
    """Build the normalize_cost_matrix args namespace, fetching/caching global constants if needed."""
    ns = SimpleNamespace(normalize_cost=args.normalize_cost, max_constant=None, min_constant=None)
    if args.normalize_cost in ("max", "minmax"):
        cached = load_cost_constants(str(DATA_ROOT), args.dataset, args.embedding_type, args.metric)
        if cached is None:
            if args.metric == "geodesic":
                raise FileNotFoundError(
                    f"No cached geodesic cost constants found for dataset={args.dataset}. "
                    "Run ot_distance.py with --embedding-type geodesic first to generate them."
                )
            print(f"Computing global {args.metric} cost constants for normalization...")
            cached = compute_and_cache_cost_constants(str(DATA_ROOT), args.dataset, args.embedding_type,
                                                       args.metric, embeddings)
        ns.max_constant, ns.min_constant = cached
    return ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geoyfcc_text", choices=["geoyfcc_text", "fmow", "geoyfcc_image"])
    p.add_argument("--total-domains", type=int, default=62)
    p.add_argument("--embedding-type", default="bert")
    p.add_argument("--metric", default="cosine", choices=["euclidean", "cosine", "geodesic"],
                   help="Distance used for the mean term (diff @ diff); 'geodesic' requires 2D "
                        "[lat, lon] embeddings. The covariance/trace term is unaffected.")
    p.add_argument("--normalize-cost", default="none", choices=["none", "max", "minmax"],
                   help="Normalize the mean-term distance before squaring it, the same way OT "
                        "normalizes its cost matrix (see cost_matrix.normalize_cost_matrix). "
                        "'max_per_domain' is not supported for FID: unlike OT/MMD, FID has no full "
                        "sample-pairwise distance matrix to take a per-pair max over.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = args.total_domains
    embeddings, domains = load_embeddings_and_domains(args.dataset, args.embedding_type)
    active = [i for i in range(n) if (domains == i).sum() > 0]
    print(f"{len(active)} non-empty domains, {len(active) * (len(active) - 1)} pairs to compute")

    normalize_args = build_normalize_args(args, embeddings)

    matrix = compute_fid_matrix(embeddings, domains, active, n, args.max_samples, device,
                                 metric=args.metric, normalize_args=normalize_args)

    out_dir = DATA_ROOT / args.dataset / "distances" / "fid_distance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fid_{args.embedding_type}_m{args.metric}_n{args.normalize_cost}.csv"
    pd.DataFrame(matrix, index=range(n), columns=range(n)).to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
