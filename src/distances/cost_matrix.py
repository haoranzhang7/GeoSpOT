import json
from pathlib import Path
from types import SimpleNamespace

import torch
import ot


def haversine_distance(src, tgt, radius=6371.0):
    """Pairwise great-circle distance (km) between src[N,2] and tgt[M,2] lat/lon points."""
    src_rad, tgt_rad = torch.deg2rad(src), torch.deg2rad(tgt)
    lat1, lon1 = src_rad[:, 0:1], src_rad[:, 1:2]
    lat2, lon2 = tgt_rad[:, 0], tgt_rad[:, 1]
    a = (torch.sin((lat1 - lat2) / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon1 - lon2) / 2) ** 2)
    return radius * 2 * torch.arcsin(torch.sqrt(a.clamp(max=1.0)))


def compute_cost_matrix(src_embeddings, tgt_embeddings, metric: str):
    """Pairwise distance matrix D_ij = d(src_i, tgt_j) for the given metric (not squared)."""
    if metric == "cosine":
        src_norm = src_embeddings / src_embeddings.norm(dim=1, keepdim=True)
        tgt_norm = tgt_embeddings / tgt_embeddings.norm(dim=1, keepdim=True)
        return 1 - torch.matmul(src_norm, tgt_norm.T)
    if metric == "geodesic":
        if src_embeddings.shape[1] != 2 or tgt_embeddings.shape[1] != 2:
            raise ValueError(f"Geodesic requires 2D coords, got src: {src_embeddings.shape[1]}, tgt: {tgt_embeddings.shape[1]}")
        return haversine_distance(src_embeddings, tgt_embeddings)
    if metric == "euclidean":
        return torch.cdist(src_embeddings, tgt_embeddings)
    return ot.dist(src_embeddings, tgt_embeddings, metric=metric)


def pairwise_minmax(src_embeddings, tgt_embeddings, metric: str):
    """Global min/max of a pairwise distance matrix, used to derive normalization constants."""
    distances = compute_cost_matrix(src_embeddings, tgt_embeddings, metric)
    return distances.min().item(), distances.max().item()


def cosine_distance_minmax(src_embeddings, tgt_embeddings):
    return pairwise_minmax(src_embeddings, tgt_embeddings, "cosine")


def normalize_cost_matrix(cost_matrix, args):
    if args.normalize_cost in ("none", None):
        return cost_matrix
    if args.normalize_cost == "max":
        return cost_matrix / args.max_constant
    if args.normalize_cost == "minmax":
        return (cost_matrix - args.min_constant) / (args.max_constant - args.min_constant)
    if args.normalize_cost in ("max_per_domain", "max_per_domain_and_normalized_after"):
        return cost_matrix / torch.max(cost_matrix)
    raise ValueError(f"Unknown normalize_cost: {args.normalize_cost}")


def cost_constants_cache_path(data_root, dataset_name, embedding_type, metric):
    """Cache path for global min/max cost constants. 'cosine' and 'geodesic' keep the legacy
    (unsuffixed) filename so existing caches stay valid; other metrics get a metric suffix."""
    suffix = "" if metric in ("cosine", "geodesic") else f"_{metric}"
    return Path(data_root) / dataset_name / f"{embedding_type}{suffix}_cost_matrix_data.json"


def load_cost_constants(data_root, dataset_name, embedding_type, metric):
    """Load cached (max, min) cost constants, or None if not cached yet."""
    path = cost_constants_cache_path(data_root, dataset_name, embedding_type, metric)
    if not path.exists():
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data['cost_max'], data['cost_min']


def compute_and_cache_cost_constants(data_root, dataset_name, embedding_type, metric, embeddings):
    """Compute global pairwise min/max cost over `embeddings` and cache it to disk."""
    min_val, max_val = pairwise_minmax(embeddings, embeddings, metric)
    path = cost_constants_cache_path(data_root, dataset_name, embedding_type, metric)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'cost_min': float(min_val), 'cost_max': float(max_val)}, f, indent=2)
    return max_val, min_val


def build_normalize_args(data_root, args, embeddings):
    """Build the normalize_cost_matrix args namespace, fetching/caching global constants if needed."""
    ns = SimpleNamespace(normalize_cost=args.normalize_cost, max_constant=None, min_constant=None)
    if args.normalize_cost in ("max", "minmax"):
        cached = load_cost_constants(str(data_root), args.dataset, args.embedding_type, args.metric)
        if cached is None:
            if args.metric == "geodesic":
                raise FileNotFoundError(
                    f"No cached geodesic cost constants found for dataset={args.dataset}. "
                    "Run ot_distance.py with --embedding-type geodesic first to generate them.")
            print(f"Computing global {args.metric} cost constants for normalization...")
            cached = compute_and_cache_cost_constants(str(data_root), args.dataset, args.embedding_type,
                                                       args.metric, embeddings)
        ns.max_constant, ns.min_constant = cached
    return ns
