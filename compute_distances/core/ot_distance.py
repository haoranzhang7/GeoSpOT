"""
General utilities for computing Optimal Transport distances between embeddings.
This module provides dataset-agnostic functions for OT computation.
"""

import os
import pickle
import time
import numpy as np
import torch
import ot
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class OTConfig:
    """Configuration for OT computation."""
    reg_e: float
    max_iter: int
    normalize_cost: str  # "max" or "minmax"
    method: str  # "sinkhorn", "sinkhorn_log", "emd"
    metric: str  # "cosine", "euclidean", "geodesic"
    
    def __post_init__(self):
        if self.normalize_cost not in ["none", "max", "minmax", "max_per_domain", "max_per_domain_and_normalized_after"]:
            raise ValueError(f"normalize_cost must be 'max' or 'minmax', got {self.normalize_cost}")
        if self.method not in ["sinkhorn", "sinkhorn_log", "emd"]:
            raise ValueError(f"method must be 'sinkhorn', 'sinkhorn_log', or 'emd', got {self.method}")


def haversine_distance_torch_optimized(src, tgt, radius=6371.0):
    """
    Optimized pairwise haversine distances between two sets of lat/lon coords.
    
    Optimizations:
    1. Pre-compute trigonometric values
    2. Use more efficient tensor operations
    3. Reduce intermediate computations
    
    Args:
        src: (n, 2) tensor [lat, lon] in degrees
        tgt: (m, 2) tensor [lat, lon] in degrees
        radius: Earth radius in kilometers (default: 6371.0)

    Returns:
        (n, m) tensor of distances in km
    """
    # Convert to radians once
    src_rad = torch.deg2rad(src)
    tgt_rad = torch.deg2rad(tgt)
    
    # Extract coordinates
    lat1, lon1 = src_rad[:, 0:1], src_rad[:, 1:2]  # (n, 1)
    lat2, lon2 = tgt_rad[:, 0], tgt_rad[:, 1]      # (m,)
    
    # Pre-compute trigonometric values
    cos_lat1 = torch.cos(lat1)  # (n, 1)
    cos_lat2 = torch.cos(lat2)  # (m,)
    sin_lat1 = torch.sin(lat1)  # (n, 1)
    sin_lat2 = torch.sin(lat2)  # (m,)
    
    # Compute differences
    dlat = lat1 - lat2  # (n, m)
    dlon = lon1 - lon2  # (n, m)
    
    # Optimized haversine formula
    sin_dlat_half = torch.sin(dlat / 2)  # (n, m)
    sin_dlon_half = torch.sin(dlon / 2)  # (n, m)
    
    # More efficient computation
    a = sin_dlat_half * sin_dlat_half + cos_lat1 * cos_lat2 * sin_dlon_half * sin_dlon_half
    c = 2 * torch.arcsin(torch.sqrt(a.clamp(max=1.0)))
    
    return radius * c


def haversine_distance_torch(src, tgt, radius=6371.0):
    """
    Compute pairwise haversine distances between two sets of lat/lon coords.
    
    Args:
        src: (n, 2) tensor [lat, lon] in degrees
        tgt: (m, 2) tensor [lat, lon] in degrees
        radius: Earth radius in kilometers (default: 6371.0)

    Returns:
        (n, m) tensor of distances in km
    """
    src_rad = torch.deg2rad(src)
    tgt_rad = torch.deg2rad(tgt)

    lat1, lon1 = src_rad[:, 0:1], src_rad[:, 1:2]
    lat2, lon2 = tgt_rad[:, 0], tgt_rad[:, 1]

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a.clamp(max=1.0)))
    return radius * c


def compute_cost_matrix(src_embeddings, tgt_embeddings, metric: str):
    """
    Compute cost matrix between source and target embeddings.
    
    Args:
        src_embeddings: Source embeddings tensor
        tgt_embeddings: Target embeddings tensor
        metric: Distance metric ("cosine", "euclidean", "geodesic")
    
    Returns:
        Cost matrix tensor
    """
    if metric == "cosine":
        src_norm = src_embeddings / src_embeddings.norm(dim=1, keepdim=True)
        tgt_norm = tgt_embeddings / tgt_embeddings.norm(dim=1, keepdim=True)
        return 1 - torch.matmul(src_norm, tgt_norm.T)
    
    elif metric == "euclidean":
        return ot.dist(src_embeddings, tgt_embeddings, metric='euclidean')
    
    elif metric == "geodesic":
        if src_embeddings.shape[1] != 2 or tgt_embeddings.shape[1] != 2:
            raise ValueError(f"Geodesic requires 2D coords, got src: {src_embeddings.shape[1]}, tgt: {tgt_embeddings.shape[1]}")
        return haversine_distance_torch_optimized(src_embeddings, tgt_embeddings)
    
    else:
        return ot.dist(src_embeddings, tgt_embeddings, metric=metric)


def normalize_cost_matrix(cost_matrix, normalize_cost: str, max_constant: float, min_constant: Optional[float] = None):
    """
    Normalize cost matrix.
    
    Args:
        cost_matrix: Cost matrix to normalize
        normalize_cost: "max" or "minmax"
        max_constant: Maximum value for normalization
        min_constant: Minimum value for minmax normalization (required if normalize_cost="minmax")
    
    Returns:
        Normalized cost matrix
    """
    if normalize_cost == "none" or normalize_cost is None:
        return cost_matrix
    elif normalize_cost == "max":
        return cost_matrix / max_constant
    elif normalize_cost == "minmax":
        if min_constant is None:
            raise ValueError("min_constant is required when normalize_cost='minmax'")
        return (cost_matrix - min_constant) / (max_constant - min_constant)
    elif normalize_cost == "max_per_domain" or "max_per_domain_and_normalized_after":
        max_per_domain = torch.max(cost_matrix)
        return cost_matrix / max_per_domain
    else:
        raise ValueError(f"normalize_cost must be 'max', 'minmax', 'max_per_domain' or none, got {normalize_cost}")


def compute_ot_distance(src_embeddings, tgt_embeddings, 
                       config: OTConfig,
                       max_constant: float, 
                       min_constant: Optional[float] = None,) -> Tuple[float, float]:
    """
    Compute OT distance between two sets of embeddings.
    
    Args:
        src_embeddings: Source embeddings tensor
        tgt_embeddings: Target embeddings tensor
        config: OTConfig with all required parameters
        max_constant: Maximum value for cost normalization
        min_constant: Minimum value for minmax normalization (required if config.normalize_cost="minmax")
    
    Returns:
        Tuple of (distance, computation_time)
    """
    start_time = time.time()
    
    # Compute and normalize cost matrix
    cost_matrix = compute_cost_matrix(src_embeddings, tgt_embeddings, config.metric)
    cost_matrix = normalize_cost_matrix(cost_matrix, config.normalize_cost, max_constant, min_constant)
    
    # Uniform distributions
    a = torch.ones(src_embeddings.shape[0], device=src_embeddings.device) / src_embeddings.shape[0]
    b = torch.ones(tgt_embeddings.shape[0], device=tgt_embeddings.device) / tgt_embeddings.shape[0]
    
    # Compute optimal transport distance
    if config.method in ["sinkhorn", "sinkhorn_log"]:
        run_method = config.method
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance = ot.sinkhorn2(
                a, b, cost_matrix,
                method=run_method,
                reg=config.reg_e,
                numItermax=config.max_iter,
                verbose=False,
                stopThr=1e-8
            )
        numerical_warning = any(
            isinstance(wi.message, Warning) and "numerical errors" in str(wi.message).lower()
            for wi in w
        )
        if (numerical_warning or not np.isfinite(float(distance))) and run_method != "sinkhorn_log":
            print("[INFO] Numerical errors detected in Sinkhorn; retrying with sinkhorn_log", flush=True)
            run_method = "sinkhorn_log"
            distance = ot.sinkhorn2(
                a, b, cost_matrix,
                method=run_method,
                reg=config.reg_e,
                numItermax=config.max_iter,
                verbose=False,
                stopThr=1e-8
            )
    elif config.method == "emd":
        distance = ot.emd2(a, b, cost_matrix, verbose=True)
    else:
        raise ValueError(f"Unsupported method: {config.method}")
    
    computation_time = time.time() - start_time
    
    if float(distance) == 0.0:
        from IPython import embed; embed()
    return float(distance), computation_time

def compute_combined_ot_distance(src_embeddings_dict, tgt_embeddings_dict, 
                       config: OTConfig,
                       max_constant_dict: Dict[str, float] = None, 
                       min_constant_dict: Dict[str, float] = None,
                       lambda_param: float = 0.5) -> Tuple[float, float]:
    """Compute OT distance using combined embeddings"""
    start_time = time.time()

    # Assert exactly 2 embedding types
    assert len(src_embeddings_dict) == 2, f"Expected exactly 2 embedding types, got {len(src_embeddings_dict)}"
    assert len(tgt_embeddings_dict) == 2, f"Expected exactly 2 embedding types, got {len(tgt_embeddings_dict)}"
    assert set(src_embeddings_dict.keys()) == set(tgt_embeddings_dict.keys()), "Source and target embedding types must match"

    cost_matrices = {}
    for embedding_type, src_embeddings in src_embeddings_dict.items():
        tgt_embeddings = tgt_embeddings_dict[embedding_type]
        max_constant = max_constant_dict[embedding_type]
        min_constant = min_constant_dict[embedding_type]

        if embedding_type == "geodesic":
            metric = "geodesic"
        else:
            metric = config.metric
        
        cost_matrix = compute_cost_matrix(src_embeddings, tgt_embeddings, metric)
        cost_matrix = normalize_cost_matrix(cost_matrix, config.normalize_cost, max_constant, min_constant)
        cost_matrices[embedding_type] = cost_matrix

    # Get the two cost matrices in a deterministic order
    embedding_types = sorted(cost_matrices.keys())
    first_cost_matrix = cost_matrices[embedding_types[0]]
    second_cost_matrix = cost_matrices[embedding_types[1]]
    
    # Combine with lambda: lambda * (first) + (1 - lambda) * (second)
    print(f"Combining cost matrices for {embedding_types[0]} and {embedding_types[1]} with lambda {lambda_param}")
    combined_cost_matrix = lambda_param * first_cost_matrix + (1 - lambda_param) * second_cost_matrix

    if config.normalize_cost == "max_per_domain_and_normalized_after":
        combined_cost_matrix = normalize_cost_matrix(combined_cost_matrix, 'max_per_domain', max_constant, min_constant)

    # Uniform distributions
    a = torch.ones(combined_cost_matrix.shape[0], device=combined_cost_matrix.device) / combined_cost_matrix.shape[0]
    b = torch.ones(combined_cost_matrix.shape[1], device=combined_cost_matrix.device) / combined_cost_matrix.shape[1]
    
    # Compute optimal transport distance
    if config.method in ["sinkhorn", "sinkhorn_log"]:
        run_method = config.method
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance = ot.sinkhorn2(
                a, b, combined_cost_matrix,
                method=run_method,
                reg=config.reg_e,
                numItermax=config.max_iter,
                verbose=False,
                stopThr=1e-8
            )
        numerical_warning = any(
            isinstance(wi.message, Warning) and "numerical errors" in str(wi.message).lower()
            for wi in w
        )
        if (numerical_warning or not np.isfinite(float(distance))) and run_method != "sinkhorn_log":
            print("[INFO] Numerical errors detected in Sinkhorn; retrying with sinkhorn_log", flush=True)
            run_method = "sinkhorn_log"
            distance = ot.sinkhorn2(
                a, b, combined_cost_matrix,
                method=run_method,
                reg=config.reg_e,
                numItermax=config.max_iter,
                verbose=False,
                stopThr=1e-8
            )
    elif config.method == "emd":
        distance = ot.emd2(a, b, combined_cost_matrix, verbose=True)
    else:
        raise ValueError(f"Unsupported method: {config.method}")
    
    # Cleanup intermediate tensors
    del cost_matrices, combined_cost_matrix, a, b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    computation_time = time.time() - start_time
    return float(distance), computation_time


def compute_ot_coupling(src_embeddings, tgt_embeddings, 
                       config: OTConfig,
                       max_constant: float, 
                       min_constant: Optional[float] = None):
    """
    Compute OT coupling (transport plan) between two sets of embeddings.
    
    Args:
        src_embeddings: Source embeddings tensor
        tgt_embeddings: Target embeddings tensor
        config: OTConfig with all required parameters
        max_constant: Maximum value for cost normalization
        min_constant: Minimum value for minmax normalization
    
    Returns:
        Transport plan (coupling matrix)
    """
    # Compute and normalize cost matrix
    cost_matrix = compute_cost_matrix(src_embeddings, tgt_embeddings, config.metric)
    cost_matrix = normalize_cost_matrix(cost_matrix, config.normalize_cost, max_constant, min_constant)
    
    # Uniform distributions
    a = torch.ones(src_embeddings.shape[0], device=src_embeddings.device) / src_embeddings.shape[0]
    b = torch.ones(tgt_embeddings.shape[0], device=tgt_embeddings.device) / tgt_embeddings.shape[0]
    
    # Compute transport plan
    if config.method == "sinkhorn":
        coupling = ot.sinkhorn(a, b, cost_matrix, reg=config.reg_e, numItermax=config.max_iter, verbose=True)
    elif config.method == "emd":
        coupling = ot.emd(a, b, cost_matrix, verbose=True)
    else:
        raise ValueError(f"Unsupported method: {config.method}")
    
    return coupling


# ==================== Domain Union Utilities ====================

def format_domain_identifier(domain_idx: Union[int, List[int]]) -> str:
    """Format domain identifier for cache filenames (e.g., "55" or "55+57+60")."""
    if isinstance(domain_idx, (list, tuple)):
        return "+".join(str(idx) for idx in sorted(domain_idx))
    return str(domain_idx)


def combine_domain_embeddings(embeddings_dict: Dict[int, torch.Tensor], 
                             domain_indices: Union[int, List[int]]) -> torch.Tensor:
    """
    Combine embeddings from one or more domains.
    
    Args:
        embeddings_dict: Dictionary mapping domain index to embeddings tensor
        domain_indices: Single domain index or list of domain indices to combine
        
    Returns:
        Combined embeddings tensor (concatenated along sample dimension)
    """
    if isinstance(domain_indices, int):
        domain_indices = [domain_indices]
    
    combined = []
    for idx in domain_indices:
        if idx not in embeddings_dict:
            raise ValueError(f"Domain {idx} not found in embeddings dictionary")
        combined.append(embeddings_dict[idx])
    
    return torch.cat(combined, dim=0)


def compute_ot_distance_with_unions(embeddings_dict: Dict[int, torch.Tensor],
                                   src_domains: Union[int, List[int]],
                                   tgt_domains: Union[int, List[int]],
                                   config: OTConfig,
                                   max_constant: float,
                                   min_constant: Optional[float] = None) -> Tuple[float, float]:
    """
    Compute OT distance between unions of domains.
    
    Args:
        embeddings_dict: Dictionary mapping domain index to embeddings tensor
        src_domains: Single domain index or list of source domain indices
        tgt_domains: Single domain index or list of target domain indices
        config: OTConfig with all required parameters
        max_constant: Maximum value for cost normalization
        min_constant: Minimum value for minmax normalization
        
    Returns:
        Tuple of (distance, computation_time)
    """
    src_embeddings = combine_domain_embeddings(embeddings_dict, src_domains)
    tgt_embeddings = combine_domain_embeddings(embeddings_dict, tgt_domains)
    
    return compute_ot_distance(src_embeddings, tgt_embeddings, config, max_constant, min_constant)


# ==================== Caching Utilities ====================

def get_ot_distance_cache_path(result_dir: str, 
                              embedding_type: str,
                              src_idx: Union[int, List[int]], 
                              tgt_idx: Union[int, List[int]], 
                              config: OTConfig,
                              include_greedy_sequential_str: bool = False,
                              lambda_param: Optional[float] = None) -> str:
    """
    Generate cache file path for OT distance results.
    
    Args:
        result_dir: Base directory for results
        embedding_type: Type of embeddings (overridden to 'geodesic' if metric is geodesic)
        src_idx: Single domain index or list of domain indices for union
        tgt_idx: Single domain index or list of domain indices for union
        config: OTConfig with all required parameters
    
    Returns:
        Full path to cache file
    """
    if config.metric == 'geodesic':
        embedding_type = 'geodesic'
    if embedding_type == 'geodesic':
        config.metric = 'geodesic'
    
    src_str = format_domain_identifier(src_idx)
    tgt_str = format_domain_identifier(tgt_idx)

    if lambda_param is not None:
        lambda_str = f"_lambda_{lambda_param}"
    else:
        lambda_str = ""
    
    if include_greedy_sequential_str:
        filename = (f"ot_distance_{config.normalize_cost}_{embedding_type}_"
                    f"{src_str}_to_{tgt_str}_{config.method}_eps_{config.reg_e}_"
                    f"maxIter_{config.max_iter}_{config.metric}_greedy_sequential{lambda_str}.pkl")
    else:   
        filename = (f"ot_distance_{config.normalize_cost}_{embedding_type}_"
                    f"{src_str}_to_{tgt_str}_{config.method}_eps_{config.reg_e}_"
                    f"maxIter_{config.max_iter}_{config.metric}{lambda_str}.pkl")

    return os.path.join(result_dir, "ot_distance_cache", filename)


def save_ot_distance(cache_path: str, distance: float, metadata: dict):
    """Save OT distance and metadata to cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    cache_data = {
        'distance': distance,
        'metadata': metadata
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"[INFO] OT distance saved to {cache_path}")


def load_ot_distance(cache_path: str) -> Optional[dict]:
    """Load cached OT distance. Returns None if file doesn't exist or is corrupted."""
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load cache from {cache_path}: {e}")
        return None


def precheck_cache_status(pairs: List[Tuple[Union[int, List[int]], Union[int, List[int]]]], 
                         get_cache_path_fn,
                         force_recompute: bool = False,
                         detailed_check: bool = True) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Check which pairs already have cached distance results.
    
    Args:
        pairs: List of (source_idx, target_idx) tuples (each can be int or list of ints)
        get_cache_path_fn: Function to get cache path for a pair
        force_recompute: If True, treat all as missing
        detailed_check: If True, validate cache contents
    
    Returns:
        Tuple of (cached_pairs, missing_pairs)
    """
    cached_pairs = []
    missing_pairs = []
    
    print("Checking distance cache status...")
    
    for i, (src_idx, tgt_idx) in enumerate(pairs):
        if i % 500 == 0 and i > 0:
            print(f"Checked {i}/{len(pairs)} pairs - Cached: {len(cached_pairs)}, Missing: {len(missing_pairs)}")
            
        if force_recompute:
            missing_pairs.append((src_idx, tgt_idx))
            continue
        
        cache_path = get_cache_path_fn(src_idx, tgt_idx)
        
        if not os.path.exists(cache_path):
            missing_pairs.append((src_idx, tgt_idx))
            continue
        
        if detailed_check:
            try:
                cached_data = load_ot_distance(cache_path)
                if cached_data is None:
                    missing_pairs.append((src_idx, tgt_idx))
                    continue
                
                distance = cached_data.get('distance')
                if distance is None or not isinstance(distance, (int, float)) or np.isnan(distance) or np.isinf(distance):
                    print(f"[WARNING] Invalid distance in cache: {src_idx} -> {tgt_idx}")
                    missing_pairs.append((src_idx, tgt_idx))
                    continue
                
                cached_pairs.append((src_idx, tgt_idx))
                
            except Exception as e:
                print(f"[WARNING] Cache validation failed for {src_idx} -> {tgt_idx}: {e}")
                missing_pairs.append((src_idx, tgt_idx))
        else:
            cached_pairs.append((src_idx, tgt_idx))
    
    print(f"\nCache status summary:")
    print(f"  ✓ Cached (valid): {len(cached_pairs)}")
    print(f"  ✗ Missing/Invalid: {len(missing_pairs)}")
    
    return cached_pairs, missing_pairs


# ==================== Utilities ====================

def get_cosine_cost_matrix_min_max_chunked(src_embeddings, tgt_embeddings, chunk_size=1000):
    """
    Compute min/max of cosine cost matrix without materializing the full matrix.
    Process in chunks to avoid OOM.
    
    Returns:
        (min_value, max_value)
    """
    src_norm = src_embeddings / src_embeddings.norm(dim=1, keepdim=True)
    tgt_norm = tgt_embeddings / tgt_embeddings.norm(dim=1, keepdim=True)
    n_src = src_norm.shape[0]
    
    global_min = float('inf')
    global_max = float('-inf')
    
    for i in range(0, n_src, chunk_size):
        end_idx = min(i + chunk_size, n_src)
        src_chunk = src_norm[i:end_idx]
        
        similarity_chunk = torch.matmul(src_chunk, tgt_norm.T)
        cost_chunk = 1 - similarity_chunk
        
        global_min = min(global_min, cost_chunk.min().item())
        global_max = max(global_max, cost_chunk.max().item())
        
        del similarity_chunk, cost_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if (i // chunk_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_src} source samples... "
                  f"Current min={global_min:.6f}, max={global_max:.6f}")
    
    return global_min, global_max


def transform_embeddings_with_ot(src_embeddings, tgt_embeddings, config: OTConfig):
    """
    Transform source embeddings to target domain using OT.
    
    Returns:
        Transformed source embeddings
    """
    cost_matrix = compute_cost_matrix(src_embeddings, tgt_embeddings, config.metric)
    
    a = torch.ones(src_embeddings.shape[0], device=src_embeddings.device) / src_embeddings.shape[0]
    b = torch.ones(tgt_embeddings.shape[0], device=tgt_embeddings.device) / tgt_embeddings.shape[0]
    
    if config.method == "sinkhorn":
        transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=config.reg_e, numItermax=config.max_iter)
    elif config.method == "emd":
        transport_plan = ot.emd(a, b, cost_matrix)
    else:
        raise ValueError(f"Unsupported method: {config.method}")
    
    transport_plan_normalized = transport_plan / a[:, None]
    return transport_plan_normalized @ tgt_embeddings


# ==================== Visualization ====================

def plot_ot_distance_matrix(distance_matrix: np.ndarray,
                           labels: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = "viridis",
                           title: Optional[str] = None,
                           save_path: Optional[str] = None,
                           show_values: bool = True) -> plt.Figure:
    """Plot OT distance matrix as a heatmap."""
    n_domains = distance_matrix.shape[0]
    
    if labels is None:
        labels = [f"Domain {i}" for i in range(n_domains)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.isnan(distance_matrix)
    sns.heatmap(distance_matrix, 
                annot=show_values,
                fmt='.3f',
                cmap=cmap,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                mask=mask,
                cbar_kws={'label': 'OT Distance'},
                ax=ax)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    ax.set_xlabel('Target Domain', fontsize=12)
    ax.set_ylabel('Source Domain', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig