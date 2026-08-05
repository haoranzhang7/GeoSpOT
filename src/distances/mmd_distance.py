#!/usr/bin/env python
"""Run MMD baseline distances. Saves an N×N matrix CSV (rows=src, cols=tgt)."""

import argparse, gc, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distances.utils import DATA_ROOT, load_embeddings_and_domains, get_embs
from src.distances.cost_matrix import compute_cost_matrix, build_normalize_args


def metric_dist(X, Y, metric="cosine"):
    """Pairwise distance matrix (not squared), optionally normalized the same way OT normalizes its cost matrix."""
    return compute_cost_matrix(X, Y, metric)


def gaussian_kernel(X, Y, sigma=1.0, metric="cosine", **kernel_kwargs):
    """K_ij = exp(-d(x_i, y_j)^2 / (2 sigma^2))"""
    return torch.exp(-metric_dist(X, Y, metric) ** 2 / (2 * sigma ** 2))

def linear_kernel(X, Y):
    """K_ij = x_i . y_j"""
    return X @ Y.T

def mmd(X, Y, kernel_fn, **kernel_kwargs):
    """MMD distance between two sets of samples X and Y using the specified kernel function."""
    K_xx, K_yy, K_xy = kernel_fn(X, X, **kernel_kwargs), kernel_fn(Y, Y, **kernel_kwargs), kernel_fn(X, Y, **kernel_kwargs)
    normalize_args = kernel_kwargs.get("normalize_args")
    if normalize_args is not None and normalize_args.normalize_cost.startswith("max_per_domain"):
        c = max(K_xx.max(), K_yy.max(), K_xy.max())
        K_xx, K_yy, K_xy = K_xx / c, K_yy / c, K_xy / c
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def median_heuristic_sigma(X, Y, max_samples=2000, metric="cosine", normalize_args=None):
    Z = torch.cat([X, Y], dim=0)
    if len(Z) > max_samples:
        Z = Z[torch.randperm(len(Z))[:max_samples]]
    dists = metric_dist(Z, Z, metric)
    return dists[~torch.eye(len(Z), dtype=torch.bool, device=Z.device)].median().item()


def calculate_mmd(X, Y, kernel="multiscale", sigma=None, scales=(0.1, 0.5, 1.0, 2.0, 5.0),
                   metric="cosine", normalize_args=None):
    """
    kernel="multiscale": sum of Gaussian MMDs at sigma * scales
    kernel="gaussian": single Gaussian; sigma defaults to median heuristic.
    kernel="linear": linear kernel.
    metric: distance used inside the Gaussian kernel ("euclidean", "cosine", or "geodesic").
    normalize_args: optional namespace with .normalize_cost/.max_constant/.min_constant, applied to the
    distance matrix the same way OT normalizes its cost matrix (see cost_matrix.normalize_cost_matrix).
    """
    X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    Y = Y if isinstance(Y, torch.Tensor) else torch.tensor(Y, dtype=torch.float32)

    if kernel == "linear":
        return mmd(X, Y, linear_kernel)
    base_sigma = sigma if sigma is not None else median_heuristic_sigma(X, Y, metric=metric, normalize_args=normalize_args)
    if kernel == "gaussian":
        return mmd(X, Y, gaussian_kernel, sigma=base_sigma, metric=metric, normalize_args=normalize_args)
    if kernel == "multiscale":
        return sum(mmd(X, Y, gaussian_kernel, sigma=base_sigma * s, metric=metric, normalize_args=normalize_args)
                   for s in scales)
    raise ValueError(f"Unknown kernel: {kernel}")


def mmd_cache_path(cache_dir, src, tgt):
    """MMD(X, Y) is symmetric, so src/tgt share one cache file regardless of order."""
    return cache_dir / f"{min(src, tgt)}_{max(src, tgt)}.txt"


def cached_mmd(cache_dir, src, tgt, compute_fn, force_recompute=False):
    path = mmd_cache_path(cache_dir, src, tgt)
    if path.exists() and not force_recompute:
        return float(path.read_text()), True
    dist = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(str(dist))
    return dist, False


def compute_mmd_matrix(embeddings, domains, active, n, kernel, sigma, max_samples, device,
                        metric="cosine", normalize_args=None, cache_dir=None, force_recompute=False):
    matrix = np.full((n, n), np.nan)
    for src in tqdm(active, desc="src domain", unit="domain"):
        src_embs = get_embs(embeddings, domains, src, max_samples, device)
        for tgt in active:
            if tgt == src or not np.isnan(matrix[src, tgt]):
                continue
            tgt_embs = get_embs(embeddings, domains, tgt, max_samples, device)
            compute_fn = lambda: float(calculate_mmd(src_embs, tgt_embs, kernel=kernel, sigma=sigma,
                                                       metric=metric, normalize_args=normalize_args))
            dist, cached = cached_mmd(cache_dir, src, tgt, compute_fn, force_recompute)
            matrix[src, tgt] = matrix[tgt, src] = dist
            print(f"  ({src},{tgt}): {dist:.6f}{' [cached]' if cached else ''}")
            del tgt_embs; gc.collect()
        del src_embs
        torch.cuda.empty_cache()
    return matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geoyfcc_text", choices=["geoyfcc_text", "fmow", "geoyfcc_image"])
    p.add_argument("--total-domains", type=int, default=62)
    p.add_argument("--embedding-type", default="bert")
    p.add_argument("--kernel", default="multiscale", choices=["multiscale", "gaussian", "linear"])
    p.add_argument("--metric", default="cosine", choices=["euclidean", "cosine", "geodesic"], help="Distance used inside the Gaussian kernel (ignored for kernel=linear). 'geodesic' requires 2D [lat, lon] embeddings.")
    p.add_argument("--normalize-cost", default="none", choices=["none", "max", "minmax", "max_per_domain", "max_per_domain_and_normalized_after"], help="Normalize the pairwise distance matrix before squaring it into the kernel, the same way OT normalizes its cost matrix (see cost_matrix.normalize_cost_matrix).")
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force-recompute", action="store_true", help="Recompute even if a cached pair distance exists")
    args = p.parse_args()

    out_dir = DATA_ROOT / args.dataset / "distances" / "mmd_distance"
    out = out_dir / f"mmd_{args.embedding_type}_k{args.kernel}_m{args.metric}_n{args.normalize_cost}.csv"
    cache_dir = out_dir / "cache" / out.stem

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = args.total_domains
    embeddings, domains = load_embeddings_and_domains(args.dataset, args.embedding_type)
    active = [i for i in range(n) if (domains == i).sum() > 0]
    print(f"{len(active)} non-empty domains, {len(active) * (len(active) - 1)} pairs to compute")

    normalize_args = build_normalize_args(DATA_ROOT, args, embeddings)
    matrix = compute_mmd_matrix(embeddings, domains, active, n, args.kernel, args.sigma, args.max_samples, device,
                                 metric=args.metric, normalize_args=normalize_args,
                                 cache_dir=cache_dir, force_recompute=args.force_recompute)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=range(n), columns=range(n)).to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
