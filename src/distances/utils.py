import math
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import ot

DATA_ROOT = Path("./data")
EMBEDDINGS_ROOT = Path("./data/embeddings")

NPZ_EMBEDDING_FILES = {"geoclip": "geoclip", "satclip": "satclip_L40"}


def _load_dataset_and_domains(dataset_name):
    """Load the dataset object (with its metadata df) and the per-row domain array."""
    if dataset_name == "geoyfcc_text":
        from datasets.geoyfcc.geoyfcc import GeoYFCCText
        ds = GeoYFCCText(root=str(DATA_ROOT / dataset_name), split=None)
        mask = ds.df["split"].isin(["train", "val"]).to_numpy()
        return ds, np.where(mask, ds.df["country_id"].to_numpy(), -1)
    if dataset_name == "fmow":
        from datasets.fmow.fmow import FMoW
        ds = FMoW(root=str(DATA_ROOT / "fmow"), split="train")
        return ds, ds.df["domain_idx"].to_numpy()
    if dataset_name == "geoyfcc_image":
        from datasets.geoyfcc_image.geoyfcc_image import GeoYFCCImage
        ds = GeoYFCCImage(root=str(DATA_ROOT / "geoyfcc_image"), split="train")
        return ds, ds.df["domain_idx"].to_numpy()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_train_dataset_and_domains(dataset_name):
    """Like _load_dataset_and_domains, but pre-filtered to the train+val split (row-aligned with
    embeddings that were themselves extracted from the train split only, e.g. bert.pt) while still
    excluding test."""
    if dataset_name != "geoyfcc_text":
        return _load_dataset_and_domains(dataset_name)
    from datasets.geoyfcc.geoyfcc import GeoYFCCText
    ds = GeoYFCCText(root=str(DATA_ROOT / dataset_name), split=None)
    ds.df = ds.df[ds.df["split"].isin(["train", "val"])].reset_index(drop=True)
    return ds, ds.df["country_id"].to_numpy()


_LATLON_COLUMNS = [("lat", "lon"), ("latitude", "longitude")]


def load_coordinates_and_domains(dataset_name):
    """Load lat/lon coordinates as [N,2] 'embeddings' (for the geodesic metric) and per-sample domains.
    Rows with missing lat/lon are dropped from both arrays so they stay aligned."""
    ds, domains = _load_dataset_and_domains(dataset_name)
    lat_col, lon_col = next(
        (c for c in _LATLON_COLUMNS if set(c).issubset(ds.df.columns)), (None, None)
    )
    if lat_col is None:
        raise ValueError(f"Dataset {dataset_name} has no lat/lon columns; geodesic distance is unsupported.")
    valid = ds.df[[lat_col, lon_col]].notna().all(axis=1).to_numpy()
    coords = torch.tensor(ds.df.loc[valid, [lat_col, lon_col]].to_numpy(), dtype=torch.float32)
    return coords, domains[valid]


def load_embeddings_and_domains(dataset_name, embedding_type):
    if embedding_type == "geodesic":
        return load_coordinates_and_domains(dataset_name)

    if embedding_type in NPZ_EMBEDDING_FILES:
        npz_path = EMBEDDINGS_ROOT / f"{dataset_name}_train_{NPZ_EMBEDDING_FILES[embedding_type]}.npz"
        data = np.load(npz_path)
        return torch.tensor(data["embeddings"], dtype=torch.float32), data["domains"]

    emb_path = DATA_ROOT / dataset_name / "embeddings" / f"{embedding_type}.pt"
    if not emb_path.exists():
        emb_path = DATA_ROOT / dataset_name / f"{dataset_name}_{embedding_type}_embeddings.pt"
    _, domains = _load_train_dataset_and_domains(dataset_name)
    return torch.load(emb_path, map_location="cpu"), domains


def domain_slice(tensor, domains_array, domain_idx):
    return tensor[domains_array >= 0] if domain_idx == "all" else tensor[domains_array == domain_idx]


def get_embs(embeddings, domains, idx, max_n, device):
    embs = embeddings[domains == idx]
    if max_n and len(embs) > max_n:
        embs = embs[torch.randperm(len(embs))[:max_n]]
    return embs.to(device)


def format_domain_identifier(domain_idx):
    if isinstance(domain_idx, (list, tuple)):
        return "+".join(str(i) for i in sorted(domain_idx))
    return str(domain_idx)


def get_ot_distance_cache_path(result_dir, embedding_type, src_idx, tgt_idx, ot_args,
                               include_greedy_sequential_str=False, lambda_param=None):
    metric = "geodesic" if embedding_type == "geodesic" else ot_args.metric
    src_str, tgt_str = format_domain_identifier(src_idx), format_domain_identifier(tgt_idx)
    lambda_str = f"_lambda_{lambda_param}" if lambda_param is not None else ""
    greedy_str = "_greedy_sequential" if include_greedy_sequential_str else ""
    debiased_str = "_debiased" if getattr(ot_args, "debiased", False) else ""
    filename = (f"ot_distance_{ot_args.normalize_cost}_{embedding_type}_"
                f"{src_str}_to_{tgt_str}_{ot_args.method}_eps_{ot_args.reg_e}_"
                f"maxIter_{ot_args.max_iter}_{metric}{greedy_str}{lambda_str}{debiased_str}.pkl")
    return os.path.join(result_dir, "ot_distance_cache", filename)


def save_ot_distance(cache_path, distance, metadata):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'distance': distance, 'metadata': metadata}, f)
    print(f"[INFO] OT distance saved to {cache_path}")


def load_ot_distance(cache_path):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load cache from {cache_path}: {e}")
        return None


def uniform_weights(n, device):
    return torch.ones(n, device=device) / n


def _cache_blocks_if_small(cost, blocks, n, m, device, budget_gb=15.0):
    """Caches every block once instead of recomputing cost(j0, j1) (e.g. the cosine matmul) on
    every Sinkhorn iteration, when the full N x M matrix comfortably fits in memory."""
    if device.type == "cpu" or n * m * 4 > budget_gb * 1e9:
        return cost
    cached = {(j0, j1): cost(j0, j1) for j0, j1 in blocks}
    return lambda j0, j1: cached[(j0, j1)]


def _sinkhorn_log_blocks(n, m, reg, device, cost, blocks, max_iter, stop_thr) -> float:
    """Chunked log-domain Sinkhorn given a cost(j0, j1) -> (n, j1-j0) block function; shared by
    sinkhorn_log_chunked and combined_sinkhorn_log_chunked so neither ever holds more than the
    full N x M matrix at once (one (n, chunk_size) block at a time when that wouldn't fit)."""
    cost = _cache_blocks_if_small(cost, blocks, n, m, device)
    u, v = torch.zeros(n, device=device), torch.zeros(m, device=device)
    log_a, log_b = -math.log(n), -math.log(m)  # uniform weights
    for _ in range(max_iter):
        prev_u = u
        run_max, run_sum = torch.full((n,), -torch.inf, device=device), torch.zeros(n, device=device)
        for j0, j1 in blocks:
            # Each block's cost only depends on (src, tgt), not on u/v, so compute it once here
            # and reuse it for both the v-update and the u-accumulation below (was computed twice).
            c = cost(j0, j1)
            v[j0:j1] = log_b - torch.logsumexp(-c / reg + u[:, None], dim=0)
            vals = -c / reg + v[j0:j1]
            new_max = torch.maximum(run_max, vals.max(dim=1).values)
            run_sum = run_sum * torch.exp(run_max - new_max) + torch.exp(vals - new_max[:, None]).sum(dim=1)
            run_max = new_max
        u = log_a - (run_max + torch.log(run_sum))
        if (u - prev_u).abs().max() < stop_thr:
            break

    return sum((torch.exp(-cost(j0, j1) / reg + u[:, None] + v[j0:j1]) * cost(j0, j1)).sum().item()
                for j0, j1 in blocks)


def _normalized_block_fn(src, tgt, metric, cost_args, blocks):
    """Returns cost(j0, j1) equal to normalize_cost_matrix(compute_cost_matrix(src, tgt, metric),
    cost_args)[:, j0:j1] -- i.e. normalized exactly as the full matrix would be -- computed one
    block at a time. "max_per_domain" needs the *global* max, so blocks (each spanning all of src
    and a slice of tgt that together partition tgt) are pre-scanned once for it; "max"/"minmax" use
    cost_args' precomputed constants directly, no scan needed."""
    from src.distances.cost_matrix import compute_cost_matrix, normalize_cost_matrix
    dmax = (max(compute_cost_matrix(src, tgt[j0:j1], metric).max() for j0, j1 in blocks)
            if cost_args.normalize_cost in ("max_per_domain", "max_per_domain_and_normalized_after") else None)

    def cost(j0, j1):
        c = compute_cost_matrix(src, tgt[j0:j1], metric)
        return c / dmax if dmax is not None else normalize_cost_matrix(c, cost_args)

    return cost


def sinkhorn_log_chunked(src, tgt, metric, ot_args, chunk_size=4096) -> float:
    """ot.sinkhorn2(method="sinkhorn_log"), but caps memory at O(N*chunk_size) by recomputing
    (N, chunk_size) cost blocks instead of keeping the full N x M matrix in memory."""
    n, m, device = src.shape[0], tgt.shape[0], src.device
    blocks = [(j, min(j + chunk_size, m)) for j in range(0, m, chunk_size)]
    cost = _normalized_block_fn(src, tgt, metric, ot_args, blocks)
    return _sinkhorn_log_blocks(n, m, ot_args.reg_e, device, cost, blocks, ot_args.max_iter, ot_args.stop_thr)


def combined_sinkhorn_log_chunked(src1, tgt1, src2, tgt2, cost_args1, cost_args2, ot_args, chunk_size=4096) -> float:
    """Chunked counterpart of combined_ot_distance._combine_cost_matrices + solve_ot: lambda-weighted
    sum of two per-embedding-type cost blocks, never materializing the full N x M combined matrix."""
    n, m, device = src1.shape[0], tgt1.shape[0], src1.device
    blocks = [(j, min(j + chunk_size, m)) for j in range(0, m, chunk_size)]
    cost1 = _normalized_block_fn(src1, tgt1, cost_args1.metric, cost_args1, blocks)
    cost2 = _normalized_block_fn(src2, tgt2, cost_args2.metric, cost_args2, blocks)

    def combo(j0, j1):
        return ot_args.lambda_param * cost1(j0, j1) + (1 - ot_args.lambda_param) * cost2(j0, j1)

    # Same global-max reasoning as _normalized_block_fn's dmax, applied to the combined matrix.
    dmax = max(combo(j0, j1).max() for j0, j1 in blocks) if ot_args.normalize_after else None
    cost = (lambda j0, j1: combo(j0, j1) / dmax) if dmax is not None else combo

    return _sinkhorn_log_blocks(n, m, ot_args.reg_e, device, cost, blocks, ot_args.max_iter, ot_args.stop_thr)


def _check_sinkhorn_log_chunked_matches_reference():
    """Sanity check: chunked result should match plain ot.sinkhorn2 up to solver tolerance."""
    from types import SimpleNamespace
    from src.distances.cost_matrix import compute_cost_matrix, normalize_cost_matrix
    torch.manual_seed(0)
    src, tgt = torch.rand(50, 8), torch.rand(130, 8)
    args = SimpleNamespace(reg_e=0.05, max_iter=1000, stop_thr=1e-9, normalize_cost="max_per_domain")
    cost = normalize_cost_matrix(compute_cost_matrix(src, tgt, "cosine"), args)
    a, b = uniform_weights(50, "cpu"), uniform_weights(130, "cpu")
    ref = _sinkhorn2(a, b, cost, "sinkhorn_log", args)
    got = sinkhorn_log_chunked(src, tgt, "cosine", args, chunk_size=17)  # uneven chunk_size on purpose
    assert abs(ref - got) < 1e-4, f"chunked sinkhorn diverged: ref={ref:.6f} got={got:.6f}"
    print(f"OK: sinkhorn_log_chunked matches ot.sinkhorn2 (ref={ref:.6f}, got={got:.6f})")


def _check_combined_sinkhorn_log_chunked_matches_reference():
    """Sanity check: chunked combined result should match the dense _combine_cost_matrices + solve_ot path."""
    from types import SimpleNamespace
    from src.distances.combined_ot_distance import _combine_cost_matrices
    torch.manual_seed(1)
    src1, tgt1, src2, tgt2 = torch.rand(50, 8), torch.rand(130, 8), torch.rand(50, 4), torch.rand(130, 4)
    cost_args1 = SimpleNamespace(metric="cosine", normalize_cost="max_per_domain")
    cost_args2 = SimpleNamespace(metric="cosine", normalize_cost="max_per_domain")
    ot_args = SimpleNamespace(reg_e=0.05, max_iter=1000, stop_thr=1e-9, lambda_param=0.5, normalize_after=True)
    combined = _combine_cost_matrices(src1, tgt1, src2, tgt2, cost_args1, cost_args2, ot_args)
    a, b = uniform_weights(50, "cpu"), uniform_weights(130, "cpu")
    ref = _sinkhorn2(a, b, combined, "sinkhorn_log", ot_args)
    got = combined_sinkhorn_log_chunked(src1, tgt1, src2, tgt2, cost_args1, cost_args2, ot_args, chunk_size=17)
    assert abs(ref - got) < 1e-4, f"combined chunked sinkhorn diverged: ref={ref:.6f} got={got:.6f}"
    print(f"OK: combined_sinkhorn_log_chunked matches reference (ref={ref:.6f}, got={got:.6f})")


def _sinkhorn2(a, b, cost_matrix, method, ot_args):
    return float(ot.sinkhorn2(a, b, cost_matrix, method=method, reg=ot_args.reg_e,
                               numItermax=ot_args.max_iter, verbose=False, stopThr=ot_args.stop_thr))


def solve_ot(a, b, cost_matrix, ot_args) -> float:
    if ot_args.method == "sinkhorn":
        # Exp-domain sinkhorn can silently underflow/diverge on this pipeline's cost matrices;
        # POT catches that internally and warns rather than returning NaN, so detect the warning
        # and fall back to the numerically-stable log-domain solve instead of trusting the result.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = _sinkhorn2(a, b, cost_matrix, "sinkhorn", ot_args)
        if math.isfinite(value) and not any("numerical errors" in str(w.message) for w in caught):
            return value
        print("[WARNING] sinkhorn hit numerical errors; retrying with sinkhorn_log")
        return _sinkhorn2(a, b, cost_matrix, "sinkhorn_log", ot_args)
    if ot_args.method == "sinkhorn_log":
        return _sinkhorn2(a, b, cost_matrix, "sinkhorn_log", ot_args)
    if ot_args.method == "emd":
        return float(ot.emd2(a, b, cost_matrix, verbose=True))
    raise ValueError(f"Unsupported method: {ot_args.method}")


if __name__ == "__main__":
    _check_sinkhorn_log_chunked_matches_reference()
    _check_combined_sinkhorn_log_chunked_matches_reference()
