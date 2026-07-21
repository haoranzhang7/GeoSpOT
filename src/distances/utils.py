import os
import pickle
from pathlib import Path

import numpy as np
import torch
import ot

DATA_ROOT = Path("./data")
EMBEDDINGS_ROOT = Path("./data/embeddings")

NPZ_EMBEDDING_FILES = {
    "geoclip": "geoclip",
    "satclip": "satclip_L40",
}


def _load_dataset_and_domains(dataset_name):
    """Load the dataset object (with its metadata df) and the per-row domain array."""
    if dataset_name == "geoyfcc_text":
        from datasets.geoyfcc.geoyfcc import GeoYFCCText
        ds = GeoYFCCText(root=str(DATA_ROOT / dataset_name), split=None)
        train_mask = np.array(ds.df["split"].isin(["train", "val"]))
        domains = np.where(train_mask, np.array(ds.df["country_id"]), -1)
    elif dataset_name == "fmow":
        from datasets.fmow.fmow import FMoW
        ds = FMoW(root=str(DATA_ROOT / "fmow"), split="train")
        domains = np.array(ds.df["domain_idx"])
    elif dataset_name == "geoyfcc_image":
        from datasets.geoyfcc_image.geoyfcc_image import GeoYFCCImage
        ds = GeoYFCCImage(root=str(DATA_ROOT / "geoyfcc_image"), split="train")
        domains = np.array(ds.df["domain_idx"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ds, domains


def _load_train_dataset_and_domains(dataset_name):
    """Like _load_dataset_and_domains, but pre-filtered to the train+val split (row-aligned with
    embeddings that were themselves extracted from the train split only, e.g. bert.pt) while still
    excluding test."""
    if dataset_name == "geoyfcc_text":
        from datasets.geoyfcc.geoyfcc import GeoYFCCText
        ds = GeoYFCCText(root=str(DATA_ROOT / dataset_name), split=None)
        ds.df = ds.df[ds.df["split"].isin(["train", "val"])].reset_index(drop=True)
        domains = np.array(ds.df["country_id"])
        return ds, domains
    return _load_dataset_and_domains(dataset_name)


_LATLON_COLUMNs = [("lat", "lon"), ("latitude", "longitude")]


def load_coordinates_and_domains(dataset_name):
    """Load lat/lon coordinates as [N,2] 'embeddings' (for the geodesic metric) and per-sample domains.
    Rows with missing lat/lon are dropped from both arrays so they stay aligned."""
    ds, domains = _load_dataset_and_domains(dataset_name)
    lat_col, lon_col = next(
        (cols for cols in _LATLON_COLUMNs if set(cols).issubset(ds.df.columns)), (None, None)
    )
    if lat_col is None:
        raise ValueError(f"Dataset {dataset_name} has no lat/lon columns; geodesic distance is unsupported.")

    valid = ds.df[[lat_col, lon_col]].notna().all(axis=1).to_numpy()
    coords = torch.tensor(ds.df.loc[valid, [lat_col, lon_col]].to_numpy(), dtype=torch.float32)
    domains = domains[valid]
    return coords, domains


def load_embeddings_and_domains(dataset_name, embedding_type):
    if embedding_type == "geodesic":
        return load_coordinates_and_domains(dataset_name)

    if embedding_type in NPZ_EMBEDDING_FILES:
        npz_path = EMBEDDINGS_ROOT / f"{dataset_name}_train_{NPZ_EMBEDDING_FILES[embedding_type]}.npz"
        data = np.load(npz_path)
        embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
        domains = data["domains"]
        return embeddings, domains

    emb_path = DATA_ROOT / dataset_name / "embeddings" / f"{embedding_type}.pt"
    if not emb_path.exists():
        emb_path = DATA_ROOT / dataset_name / f"{dataset_name}_{embedding_type}_embeddings.pt"
    embeddings = torch.load(emb_path, map_location="cpu")

    _, domains = _load_train_dataset_and_domains(dataset_name)
    return embeddings, domains


def domain_slice(tensor, domains_array, domain_idx):
    if domain_idx == "all":
        return tensor[domains_array >= 0]
    return tensor[domains_array == domain_idx]


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


def solve_ot(a, b, cost_matrix, ot_args) -> float:
    if ot_args.method in ("sinkhorn", "sinkhorn_log"):
        return float(ot.sinkhorn2(a, b, cost_matrix, method=ot_args.method, reg=ot_args.reg_e,
                                   numItermax=ot_args.max_iter, verbose=False, stopThr=1e-8))
    if ot_args.method == "emd":
        return float(ot.emd2(a, b, cost_matrix, verbose=True))
    raise ValueError(f"Unsupported method: {ot_args.method}")
