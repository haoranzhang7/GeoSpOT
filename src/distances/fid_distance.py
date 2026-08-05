#!/usr/bin/env python
"""
Compute pairwise FID distances over domain embeddings. Saves an N×N matrix CSV (rows=src, cols=tgt).

Example:
  python src/distances/fid_distance.py --dataset geoyfcc_text --embedding-type satclip
"""

import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import linalg
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distances.utils import DATA_ROOT, load_embeddings_and_domains, get_embs
from src.distances.cost_matrix import compute_cost_matrix, normalize_cost_matrix, build_normalize_args


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
    # Matrix square root of the covariance product; equivalent (under the trace) to
    # (Sigma1^(1/2) Sigma2 Sigma1^(1/2))^(1/2) for PSD covariances.
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return mean_term + np.trace(sigma1 + sigma2 - 2 * covmean)


def calculate_fid(embeddings1, embeddings2, metric="cosine", normalize_args=None):
    mu1, s1 = stats(embeddings1)
    mu2, s2 = stats(embeddings2)
    return fid(mu1, s1, mu2, s2, metric=metric, normalize_args=normalize_args)


def fid_cache_path(cache_dir, src, tgt):
    """FID(P, Q) is symmetric (it's the squared Wasserstein-2 distance between Gaussians), so
    src/tgt share one cache file regardless of order."""
    return cache_dir / f"{min(src, tgt)}_{max(src, tgt)}.txt"


def cached_fid(cache_dir, src, tgt, compute_fn, force_recompute=False):
    path = fid_cache_path(cache_dir, src, tgt)
    if path.exists() and not force_recompute:
        return float(path.read_text()), True
    dist = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(str(dist))
    return dist, False


def compute_fid_matrix(embeddings, domains, active, n, max_samples, device, metric="cosine",
                        normalize_args=None, cache_dir=None, force_recompute=False):
    matrix = np.full((n, n), np.nan)
    domain_stats = {idx: stats(get_embs(embeddings, domains, idx, max_samples, device).cpu().numpy()) for idx in active}

    for src in tqdm(active, desc="src domain", unit="domain"):
        src_mu, src_sig = domain_stats[src]
        for tgt in active:
            if tgt == src or not np.isnan(matrix[src, tgt]):
                continue
            tgt_mu, tgt_sig = domain_stats[tgt]
            compute_fn = lambda: float(fid(src_mu, src_sig, tgt_mu, tgt_sig, metric=metric,
                                            normalize_args=normalize_args))
            dist, cached = cached_fid(cache_dir, src, tgt, compute_fn, force_recompute)
            matrix[src, tgt] = matrix[tgt, src] = dist
    return matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geoyfcc_text", choices=["geoyfcc_text", "fmow", "geoyfcc_image"])
    p.add_argument("--total-domains", type=int, default=62)
    p.add_argument("--embedding-type", default="bert")
    p.add_argument("--metric", default="cosine", choices=["euclidean", "cosine", "geodesic"], help="Distance used for the mean term (diff @ diff); 'geodesic' requires 2D [lat, lon] embeddings. The covariance/trace term is unaffected.")
    p.add_argument("--normalize-cost", default="none", choices=["none", "max", "minmax"], help="Normalize the mean-term distance before squaring it, the same way OT normalizes its cost matrix (see cost_matrix.normalize_cost_matrix). 'max_per_domain' is not supported for FID: unlike OT/MMD, FID has no full sample-pairwise distance matrix to take a per-pair max over.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force-recompute", action="store_true", help="Recompute even if a cached pair distance exists")
    args = p.parse_args()

    out_dir = DATA_ROOT / args.dataset / "distances" / "fid_distance"
    out = out_dir / f"fid_{args.embedding_type}_m{args.metric}_n{args.normalize_cost}.csv"
    cache_dir = out_dir / "cache" / out.stem

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = args.total_domains
    embeddings, domains = load_embeddings_and_domains(args.dataset, args.embedding_type)
    active = [i for i in range(n) if (domains == i).sum() > 0]
    print(f"{len(active)} non-empty domains, {len(active) * (len(active) - 1)} pairs to compute")

    normalize_args = build_normalize_args(DATA_ROOT, args, embeddings)
    matrix = compute_fid_matrix(embeddings, domains, active, n, args.max_samples, device,
                                 metric=args.metric, normalize_args=normalize_args,
                                 cache_dir=cache_dir, force_recompute=args.force_recompute)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=range(n), columns=range(n)).to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
