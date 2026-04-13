#!/usr/bin/env python
"""
Compute OT distances from source domain to K target domains (K=1,2,3)
"""

import os
import json
import time
import gc
import argparse
import yaml
from pathlib import Path
from itertools import combinations
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.geoyfcc.geoyfcc import GeoYFCCText

from compute_distances.core.ot_distance import (
    OTConfig,
    compute_ot_distance,
    compute_combined_ot_distance,
    get_cosine_cost_matrix_min_max_chunked,
    save_ot_distance,
    load_ot_distance,
    get_ot_distance_cache_path,
    combine_domain_embeddings
)

# ==================== Configuration Loading ====================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_base_config() -> dict:
    """Load base configuration."""
    base_config_path = PROJECT_ROOT / "configs" / "base" / "common.yaml"
    return load_config(str(base_config_path))

def load_dataset_config() -> dict:
    """Load dataset configuration."""
    dataset_config_path = PROJECT_ROOT / "configs" / "datasets" / "geoyfcc.yaml"
    return load_config(str(dataset_config_path))

def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration."""
    return load_config(config_path)

def merge_configs(base_config: dict, dataset_config: dict, experiment_config: dict) -> dict:
    """Merge configurations with proper inheritance."""
    merged = {}
    
    # Start with base config
    merged.update(base_config)
    
    # Override with dataset config
    for key, value in dataset_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    
    # Override with experiment config
    for key, value in experiment_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    
    return merged

# Default config path
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "ot_distance.yaml"

# Load configurations
base_config = load_base_config()
dataset_config = load_dataset_config()

# Global config variable (will be set by main function)
config = None

# ==================== Constants ====================

COUNTRY_MAPPING_PATH = Path("./data/geoyfcc/country_mapping.json")

# ==================== Setup ====================

def initialize_from_config(config: dict):
    """Initialize global variables from configuration."""
    global device, dataset, domains, _coordinate_cache
    
    # Set device
    device_name = config.get('COMPUTATION', {}).get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data_dir = config['PATHS']['data_root']
    dataset_name = config['DATASET']['gen_name']
    dataset_path = f"{data_dir}/{dataset_name}"
    
    dataset = GeoYFCCText(root=dataset_path, split='train')
    domains = np.array(list(dataset.df["country_id"]))
    
    # Initialize coordinate cache
    _coordinate_cache = {}


def load_country_mapping() -> dict:
    country_mapping = {}
    if COUNTRY_MAPPING_PATH.exists():
        try:
            with open(COUNTRY_MAPPING_PATH, 'r') as f:
                country_mapping_str = json.load(f)
                country_mapping = {int(k): v for k, v in country_mapping_str.items()}
        except Exception as e:
            print(f"Warning: Could not load country mapping from {COUNTRY_MAPPING_PATH}: {e}")
    else:
        print(f"Warning: Country mapping file not found at {COUNTRY_MAPPING_PATH}")
    return country_mapping

def get_data_dir(config) -> Path:
    """Get data directory from config."""
    # Try different possible locations for data directory
    if 'PATHS' in config and 'data_root' in config['PATHS']:
        return Path(config['PATHS']['data_root'])
    elif 'DATASET' in config and 'data_dir' in config['DATASET']:
        return Path(config['DATASET']['data_dir'])
    else:
        return Path("./data")

def get_dataset_name(config) -> str:
    """Get dataset name from config."""
    return config['DATASET']['gen_name']

def get_result_dir(config: dict) -> Path:
    """Get result directory from config."""
    data_dir = get_data_dir(config)
    dataset_name = get_dataset_name(config)
    return data_dir / dataset_name / "distances" / "ot_distance"

def get_ot_config(config) -> OTConfig:
    """Get OT configuration from config."""
    ot_config_dict = config.get('OT_CONFIG', {})
    return OTConfig(
        reg_e=ot_config_dict.get('reg_e', 0.01),
        max_iter=ot_config_dict.get('max_iter', 1000),
        normalize_cost=ot_config_dict.get('normalize_cost', 'max'),
        method=ot_config_dict.get('method', 'sinkhorn'),
        metric=ot_config_dict.get('metric', 'cosine')
    )

def get_embedding_path_template(config) -> Path:
    """Get embedding path template from config."""
    data_dir = get_data_dir(config)
    dataset_name = get_dataset_name(config)
    return data_dir / dataset_name / "embeddings" / "{embedding_type}.pt"

def get_embedding_types(config: dict) -> list:
    """Get embedding types from config."""
    embedding_config = config.get('EMBEDDING_TYPES', {})
    return embedding_config.get('individual', ["bert", "geoclip", "satclip_L10", "satclip_L40"])

def get_combined_embedding_types(config: dict) -> list:
    """Get combined embedding types from config."""
    embedding_config = config.get('EMBEDDING_TYPES', {})
    return embedding_config.get('combined', ["bert+geoclip", "bert+satclip_L10", "bert+satclip_L40", "bert+geodesic"])

def should_include_geodesic(config: dict) -> bool:
    """Check if geodesic should be included."""
    embedding_config = config.get('EMBEDDING_TYPES', {})
    return embedding_config.get('include_geodesic', True)

def should_include_combined(config: dict) -> bool:
    """Check if combined embeddings should be included."""
    embedding_config = config.get('EMBEDDING_TYPES', {})
    return embedding_config.get('include_combined', True)

def get_k_value(config: dict) -> int:
    """Get K value from config."""
    domain_config = config.get('DOMAIN_SELECTION', {})
    return domain_config.get('k', 1)

def should_use_greedy_sequential(config: dict) -> bool:
    """Check if greedy sequential selection should be used."""
    domain_config = config.get('DOMAIN_SELECTION', {})
    return domain_config.get('greedy_sequential', False)

def should_force_recompute(config: dict) -> bool:
    """Check if recomputation should be forced."""
    ot_config_dict = config.get('OT_CONFIG', {})
    return ot_config_dict.get('force_recompute', False)

# ==================== Helper Functions ====================

def extract_domain_embeddings(embeddings, domains_array, domain_idx, embedding_type=None):
    """Extract embeddings for a specific domain"""
    if isinstance(embeddings, dict):
        assert embedding_type is not None
        if embedding_type == "geodesic":
            return extract_domain_coordinates(dataset, domains_array, domain_idx)
        embeddings = embeddings[embedding_type]
    domain_mask = domains_array == domain_idx
    domain_embeddings = embeddings[domain_mask]
    # print(f"  Domain {domain_idx}: {len(domain_embeddings)} samples")
    return domain_embeddings

def extract_domain_coordinates(dataset, domains_array, domain_idx):
    """Extract lat/lon coordinates for a specific domain with caching and GPU acceleration"""
    global _coordinate_cache
    
    # Check cache first
    if domain_idx in _coordinate_cache:
        return _coordinate_cache[domain_idx]
    
    domain_mask = domains_array == domain_idx
    domain_df = dataset.df[domain_mask]
    
    # Get coordinates and filter out NaN values
    coords = domain_df[['lat', 'lon']].dropna()
    if len(coords) == 0:
        print(f"Warning: No valid coordinates found for domain {domain_idx}")
        empty_tensor = torch.tensor([], dtype=torch.float32, device=device)
        _coordinate_cache[domain_idx] = empty_tensor
        return empty_tensor
    
    # Convert to tensor [lat, lon] format and move to GPU
    coords_tensor = torch.tensor(coords.values, dtype=torch.float32, device=device)
    print(f"  Domain {domain_idx}: {len(coords_tensor)} samples with coordinates")
    
    # Cache the result
    _coordinate_cache[domain_idx] = coords_tensor
    return coords_tensor

def clear_coordinate_cache():
    """Clear the coordinate cache to free GPU memory"""
    global _coordinate_cache
    for coords in _coordinate_cache.values():
        if coords.numel() > 0:  # Only delete non-empty tensors
            del coords
    _coordinate_cache.clear()
    torch.cuda.empty_cache()
    print("Coordinate cache cleared")

def get_cost_constants(embedding_type):
    """Get or compute min/max cost constants for normalization"""
    if embedding_type == "geodesic":
        return get_geodesic_cost_constants()
    

    
    # Get paths from config
    data_dir = get_data_dir(config)
    dataset_name = get_dataset_name(config)
    embedding_path_template = get_embedding_path_template(config)
    
    cost_matrix_data_path = data_dir / dataset_name / f"{embedding_type}_cost_matrix_data.json"
    
    if cost_matrix_data_path.exists():
        with open(cost_matrix_data_path, 'r') as f:
            data = json.load(f)
            return data['cost_max'], data['cost_min']
    
    # Compute if not cached
    embedding_path = Path(str(embedding_path_template).format(embedding_type=embedding_type))
    embeddings = torch.load(embedding_path, map_location="cuda")
    min_val, max_val = get_cosine_cost_matrix_min_max_chunked(embeddings, embeddings)
    
    cost_matrix_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_matrix_data_path, 'w') as f:
        json.dump({'cost_min': float(min_val), 'cost_max': float(max_val)}, f, indent=2)
    
    # Cleanup embeddings immediately after use
    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    
    return max_val, min_val

def get_geodesic_cost_constants(batch_size=20000, force_recompute=False):
    """Get or compute min/max cost constants for geodesic distance"""
    # Get paths from config
    print("Getting Geodesice Cost Constants...")
    data_dir = get_data_dir(config)
    dataset_name = get_dataset_name(config)
    cost_matrix_data_path = data_dir / dataset_name / "geodesic_cost_matrix_data.json"
    
    if cost_matrix_data_path.exists() and not force_recompute:
        with open(cost_matrix_data_path, 'r') as f:
            data = json.load(f)
        return data['cost_max'], data['cost_min']
    
    # If not found, compute from all domains using coordinates
    print("Computing geodesic cost constants from lat/lon coordinates...")
    total_domains = config['DATASET']['total_domains']
    all_domain_indices = list(range(total_domains))
    
    # Collect all coordinates from all domains
    all_coords = []
    for domain_idx in all_domain_indices:
        coords = extract_domain_coordinates(dataset, domains, domain_idx)
        if len(coords) > 0:
            all_coords.append(coords)
    
    if not all_coords:
        raise ValueError("Warning: No valid coordinates found for any domain")
    
    # Combine all coordinates into one tensor
    all_coords_tensor = torch.cat(all_coords, dim=0)
    device = all_coords_tensor.device
    n = len(all_coords_tensor)

    print(f"Computing geodesic distances for {n} coordinates (batched)...")

    from compute_distances.core.ot_distance import haversine_distance_torch_optimized

    # Instead of allocating full n×n, compute min/max incrementally
    global_min = float("inf")
    global_max = float("-inf")

    with torch.no_grad():
        outer_range = range(0, n, batch_size)
        for i in tqdm(outer_range, desc="Computing geodesic batches (outer)", leave=True):
            end_i = min(i + batch_size, n)
            batch_i = all_coords_tensor[i:end_i]
            
            inner_range = range(0, n, batch_size)
            for j in tqdm(inner_range, desc=f"  Inner loop for batch {i//batch_size+1}", leave=False):
                end_j = min(j + batch_size, n)
                batch_j = all_coords_tensor[j:end_j]

                dists = haversine_distance_torch_optimized(batch_i, batch_j)

                # Remove self-distances if diagonal block
                if i == j:
                    mask = ~torch.eye(end_i - i, dtype=torch.bool, device=device)
                    dists = dists[mask]

                block_min = dists.min().item()
                block_max = dists.max().item()

                if block_min < global_min:
                    global_min = block_min
                if block_max > global_max:
                    global_max = block_max

                del dists
                torch.cuda.empty_cache()

    print(f"Geodesic cost constants: min={global_min:.2f} km, max={global_max:.2f} km")

    # Save for future use
    cost_data = {'cost_min': global_min, 'cost_max': global_max}
    cost_matrix_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_matrix_data_path, 'w') as f:
        json.dump(cost_data, f, indent=2)
    
    del all_coords_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    return global_max, global_min


def compute_or_load_distance(src_data, tgt_domain_indices, embeddings_or_dataset, domains,
                            embedding_type, max_const, min_const, ot_config, source_domain_idx, force_recompute=False,
                            include_greedy_sequential_str=False, lambda_param=None):
    """Compute or load cached OT distance"""

    result_dir = get_result_dir(config)
    cache_path = get_ot_distance_cache_path(
        result_dir=str(result_dir),
        embedding_type=embedding_type,
        src_idx=source_domain_idx,
        tgt_idx=tgt_domain_indices,
        config=ot_config,
        include_greedy_sequential_str=include_greedy_sequential_str,
        lambda_param=lambda_param
    )
    
    # Check cache
    if os.path.exists(cache_path) and not force_recompute:
        cache_data = load_ot_distance(cache_path)
        print(f"Loaded from cache {cache_path}", flush=True)
        if cache_data:
            return cache_data['distance'], True
    elif force_recompute and os.path.exists(cache_path):
        print(f"Force recompute enabled - ignoring cache {cache_path}", flush=True)
    
    # Compute distance based on embedding type
    if embedding_type == "geodesic":
        # Use coordinates for geodesic distance
        tgt_coords_list = [extract_domain_coordinates(embeddings_or_dataset, domains, idx) 
                          for idx in tgt_domain_indices]
        tgt_coords = torch.cat(tgt_coords_list, dim=0)
        
        distance, comp_time = compute_ot_distance(
            src_data, tgt_coords,
            config=ot_config,
            max_constant=max_const,
            min_constant=min_const
        )
        
        # Cleanup
        del tgt_coords_list, tgt_coords
    elif "+" in embedding_type:
        embedding_types = embedding_type.split("+")
        tgt_embeddings_dict = {}
        for embedding_type in embedding_types:
            tgt_embeddings_list = [extract_domain_embeddings(embeddings_or_dataset, domains, idx, embedding_type) 
                              for idx in tgt_domain_indices]
            tgt_embeddings = torch.cat(tgt_embeddings_list, dim=0)
            tgt_embeddings_dict[embedding_type] = tgt_embeddings
            
        # Handle combined embedding types
        # Get lambda_param from global scope (set in main)
        distance, comp_time = compute_combined_ot_distance(src_data, tgt_embeddings_dict, ot_config, max_const, min_const, lambda_param=lambda_param)
        print(f"Computed combined distance: {distance}", flush=True)
        
        # Cleanup combined embeddings
        del tgt_embeddings_dict
    else:
        # Use embeddings for other distance types
        tgt_embeddings_list = [extract_domain_embeddings(embeddings_or_dataset, domains, idx) 
                              for idx in tgt_domain_indices]
        tgt_embeddings = torch.cat(tgt_embeddings_list, dim=0)
        
        distance, comp_time = compute_ot_distance(
            src_data, tgt_embeddings,
            config=ot_config,
            max_constant=max_const,
            min_constant=min_const
        )
        
        # Cleanup
        del tgt_embeddings_list, tgt_embeddings
    
    # Save to cache
    metadata = {
        'src_domain_idx': source_domain_idx,
        'tgt_domain_indices': tgt_domain_indices,
        'embedding_type': embedding_type,
        'method': ot_config.method,
        'reg_e': ot_config.reg_e,
        'max_iter': ot_config.max_iter,
        'metric': ot_config.metric if embedding_type != "geodesic" else getattr(ot_config, 'metric_geodesic', 'geodesic'),
        'normalize_cost': ot_config.normalize_cost,
        'computation_time': comp_time,
        'src_shape': list(src_data.shape) if not isinstance(src_data, dict) else [src_data[embedding_type].shape for embedding_type in src_data.keys()],
        'timestamp': time.time()
    }
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_ot_distance(cache_path, distance, metadata)
    
    # Aggressive cleanup to free memory
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Distance computed {distance} for {embedding_type} from {source_domain_idx} to {tgt_domain_indices}")
    
    return distance, False

def greedy_sequential_ot_selection(src_data, all_domain_indices, data_source, domains, 
                                 embedding_type, max_const, min_const, ot_config, k, 
                                 source_domain_idx, force_recompute=False, lambda_param=None):
    """
    Implement greedy sequential OT distance computation.
    
    Args:
        src_data: Source domain data
        all_domain_indices: List of all available domain indices
        data_source: Data source for extracting embeddings
        domains: Domain array
        embedding_type: Type of embedding to use
        max_const, min_const: Cost constants
        ot_config: OT configuration (OTConfig object)
        k: Number of domains to select
        force_recompute: Whether to force recomputation
    
    Returns:
        tuple: (selected_domains, distances_at_each_step)
    """
    print(f"\n--- Greedy Sequential Selection for K={k} ---")
    
    # Load country mapping
    country_mapping = load_country_mapping()
    
    selected_domains = []
    distances_at_each_step = []
    remaining_domains = [idx for idx in all_domain_indices if idx != source_domain_idx]
    
    for step in range(k):
        rows = []
        print(f"\nStep {step + 1}: Selecting domain {step + 1}/{k}")
        
        best_distance = float('inf')
        best_domain = None

        if (step + 1) > 1:
            include_greedy_sequential_str = True
        else:
            include_greedy_sequential_str = False
        
        # Try each remaining domain
        for candidate_domain in remaining_domains:
            # Create target domain set: selected domains + candidate
            current_target_domains = selected_domains + [candidate_domain]
            
            print(f"  Evaluating: {current_target_domains}")
            
            # Compute OT distance
            distance, from_cache = compute_or_load_distance(
                src_data, current_target_domains, data_source, domains,
                embedding_type, max_const, min_const, ot_config, source_domain_idx, force_recompute,
                include_greedy_sequential_str=include_greedy_sequential_str, lambda_param=lambda_param
            )
            
            status = "cached" if from_cache else "computed"
            print(f"    Distance: {distance:.6f} ({status})")

            torch.cuda.empty_cache()
            gc.collect()
            
            # Update best if this is better
            if distance < best_distance:
                best_distance = distance
                best_domain = candidate_domain

            # Get country names
            source_domain_name = country_mapping.get(source_domain_idx, f"Unknown-{source_domain_idx}")
            tgt_domain_names = [country_mapping.get(tgt_idx, f"Unknown-{tgt_idx}") for tgt_idx in current_target_domains]

            row = {
                "source_domain_idx": source_domain_idx,
                "source_domain_name": source_domain_name,
                "k": step + 1,
                "tgt_domains": current_target_domains,
                "tgt_domain_names": tgt_domain_names,
                "distance": distance,
            }
            rows.append(row)

        os.makedirs(f"greedy_sequential_distances/{embedding_type}", exist_ok=True)
        distance_df = pd.DataFrame(rows)
        distance_df.to_csv(f"greedy_sequential_distances/{embedding_type}/k{step+1}_source{source_domain_idx}.csv", index=False)
        
        # Select the best domain
        selected_domains.append(best_domain)
        remaining_domains.remove(best_domain)
        distances_at_each_step.append(best_distance)
        
        print(f"  ✓ Selected domain {best_domain} with distance {best_distance:.6f}")
        print(f"  Selected so far: {selected_domains}")
    
    print(f"\n→ Final selection: {selected_domains}")
    print(f"• Distances at each step: {[f'{d:.6f}' for d in distances_at_each_step]}")

    return selected_domains, distances_at_each_step


def create_result_record(embedding_type, tgt_domain_indices, distance, source_domain_idx):
    """Create a result record with proper domain names"""
    record = {
        "embedding_type": embedding_type,
        "src_domain_idx": source_domain_idx,
        "k": len(tgt_domain_indices),
        "distance": distance
    }
    
    # Add individual target domains
    for i, idx in enumerate(tgt_domain_indices, 1):
        record[f"tgt_domain_{i}_idx"] = idx
    
    # Add combined identifier
    record["tgt_domains_combined"] = "+".join(str(idx) for idx in tgt_domain_indices)
    
    return record

# ==================== Main Computation ====================

def parse_args():
    """Parse command-line arguments"""
    # Default embedding types (will be overridden by config)
    default_embedding_types = ["bert", "geoclip", "satclip_L10", "satclip_L40", "geodesic"]
    default_combined_types = ["bert+geoclip", "bert+satclip_L10", "bert+satclip_L40", "bert+geodesic"]
    all_embedding_types = default_embedding_types + default_combined_types
    
    parser = argparse.ArgumentParser(description="Compute OT distances for GeoYFCC dataset")
    parser.add_argument("--embedding-type", type=str, required=True,
                       help="Embedding type to process (individual or combined)")
    parser.add_argument("--source-domain-idx", type=int, default=57,
                       help="Source domain index (default: 57)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file")
    parser.add_argument("--reg-e", type=float, default=None,
                       help="Regularization parameter (default: from config)")
    parser.add_argument("--max-iter", type=int, default=None,
                       help="Maximum iterations (default: from config)")
    parser.add_argument("--metric", type=str, default=None,
                       help="Distance metric (default: from config)")
    parser.add_argument("--method", type=str, default=None,
                       help="OT method (default: from config)")
    parser.add_argument("--normalize-cost", type=str, default=None,
                       help="Cost normalization (default: from config)")
    parser.add_argument("--k", type=int, default=None,
                       help="Number of target domains K (default: from config)")
    parser.add_argument("--greedy-sequential", action="store_true",
                       help="Use greedy sequential domain selection instead of all combinations")
    parser.add_argument("--force-recompute", action="store_true",
                       help="Force recomputation even if cached results exist")
    parser.add_argument("--lambda", type=float, default=None,
                       dest="lambda_param",
                       help="Lambda parameter for combined embeddings (default: 0.5)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    global config
    if args.config:
        experiment_config = load_experiment_config(args.config)
    else:
        experiment_config = load_experiment_config(str(DEFAULT_CONFIG_PATH))
    
    config = merge_configs(base_config, dataset_config, experiment_config)
    
    # Initialize from config
    initialize_from_config(config)
    
    # Override config with command-line arguments
    if args.reg_e is not None:
        config['OT_CONFIG']['reg_e'] = args.reg_e
    if args.max_iter is not None:
        config['OT_CONFIG']['max_iter'] = args.max_iter
    if args.normalize_cost is not None:
        config['OT_CONFIG']['normalize_cost'] = args.normalize_cost
    if args.method is not None:
        config['OT_CONFIG']['method'] = args.method
    if args.metric is not None:
        config['OT_CONFIG']['metric'] = args.metric
    if args.force_recompute:
        config['OT_CONFIG']['force_recompute'] = True
    if args.k is not None:
        config['DOMAIN_SELECTION']['k'] = args.k
    if args.greedy_sequential:
        config['DOMAIN_SELECTION']['greedy_sequential'] = True
    
    # Get configuration values
    ot_config = get_ot_config(config)
    source_domain_idx = args.source_domain_idx
    embedding_type = args.embedding_type
    k = get_k_value(config)
    force_recompute = should_force_recompute(config)
    use_greedy_sequential = should_use_greedy_sequential(config)
    lambda_param = args.lambda_param
    
    # Get paths and constants
    embedding_path_template = get_embedding_path_template(config)
    result_dir = get_result_dir(config)
    
    print(f"\n{'='*60}")
    print(f"Processing {embedding_type} embeddings")
    print(f"Source Domain ID: {source_domain_idx}")
    print(f"OT Config: {ot_config}")
    print(f"Force Recompute: {force_recompute}")
    print(f"K Value: {k}")
    print(f"Greedy Sequential: {use_greedy_sequential}")
    print(f"{'='*60}")
    
    # Get all domain indices
    total_domains = config['DATASET']['total_domains']
    all_domain_indices = list(range(total_domains))

    metric_to_use = getattr(ot_config, 'metric_geodesic', 'geodesic') if embedding_type == "geodesic" else ot_config.metric
    
    # Create descriptive filename suffix from OT config
    config_suffix = f"method_{ot_config.method}_reg_{ot_config.reg_e}_iter_{ot_config.max_iter}_metric_{metric_to_use}_norm_{ot_config.normalize_cost}"
    
    # Load data and constants based on embedding type
    if embedding_type == "geodesic":
        # For geodesic distance, use coordinates instead of embeddings
        print("Using lat/lon coordinates for geodesic distance computation")
        max_const, min_const = get_cost_constants(embedding_type)
        print(f"Cost constants: min={min_const:.6f} km, max={max_const:.6f} km")
        
        # Extract source coordinates
        src_data = extract_domain_coordinates(dataset, domains, source_domain_idx)
        if len(src_data) == 0:
            print(f"✗ No valid coordinates found for source domain {source_domain_idx}")
            return
        
        data_source = dataset  # Pass dataset for coordinate extraction
    elif "+" in embedding_type:
        embedding_types = embedding_type.split("+")
        src_data_dict = {}
        data_source_dict = {}
        max_const_dict = {}
        min_const_dict = {}
        
        for emb_type in embedding_types:
            if emb_type == "geodesic":
                src_data_dict[emb_type] = extract_domain_coordinates(dataset, domains, source_domain_idx)
            else:
                embedding_path = Path(str(embedding_path_template).format(embedding_type=emb_type))
                if not embedding_path.exists():
                    print(f"✗ Embedding file not found: {embedding_path}")
                    return
                embeddings = torch.load(embedding_path, map_location="cuda")
                data_source_dict[emb_type] = embeddings
                src_data_dict[emb_type] = extract_domain_embeddings(embeddings, domains, source_domain_idx)
                # Don't delete embeddings here as they're still needed in data_source_dict
            max_const_dict[emb_type], min_const_dict[emb_type] = get_cost_constants(emb_type)
        
        src_data = src_data_dict
        max_const = max_const_dict
        min_const = min_const_dict
        data_source = data_source_dict
    else:
        # For other embedding types, use embeddings
        embedding_path = Path(str(embedding_path_template).format(embedding_type=embedding_type))
        if not embedding_path.exists():
            print(f"✗ Embedding file not found: {embedding_path}")
            return
        
        embeddings = torch.load(embedding_path, map_location="cuda")
        max_const, min_const = get_cost_constants(embedding_type)
        
        print(f"Cost constants: min={min_const:.6f}, max={max_const:.6f}")
        
        # Extract source embeddings
        src_data = extract_domain_embeddings(embeddings, domains, source_domain_idx)
        data_source = embeddings  # Pass embeddings for embedding extraction
    
    # Process K target domains
    print(f"\n--- K={k} target domains ---")
    
    records = []
    
    if use_greedy_sequential:
        # Use greedy sequential selection
        print("Using greedy sequential domain selection")
        selected_domains, distances_at_each_step = greedy_sequential_ot_selection(
            src_data, all_domain_indices, data_source, domains,
            embedding_type, max_const, min_const, ot_config, k, source_domain_idx, force_recompute, lambda_param
        )
        
        # Create records for each step
        for step, (domain, distance) in enumerate(zip(selected_domains, distances_at_each_step)):
            # Create target domain set up to this step
            step_domains = selected_domains[:step+1]
            record = create_result_record(embedding_type, step_domains, distance, source_domain_idx)
            record["selection_step"] = step + 1
            record["greedy_sequential"] = True
            records.append(record)
            
        print(f"✓ Greedy sequential selection completed")
        
    else:
        # Use all combinations (original behavior)
        tgt_combinations = list(combinations(all_domain_indices, k))
        print(f"Computing {len(tgt_combinations)} combinations...")
        
        for idx, tgt_domain_indices in enumerate(tgt_combinations, 1):
            distance, from_cache = compute_or_load_distance(
                src_data, list(tgt_domain_indices), data_source, domains,
                embedding_type, max_const, min_const, ot_config, source_domain_idx, force_recompute,
                lambda_param=lambda_param
            )
            
            status = "cached" if from_cache else "computed"
            if idx % 100 == 0 or idx == len(tgt_combinations):
                print(f"  [{idx}/{len(tgt_combinations)}] {tgt_domain_indices}: {distance:.6f} ({status})")
            
            record = create_result_record(embedding_type, tgt_domain_indices, distance, source_domain_idx)
            record["greedy_sequential"] = False
            records.append(record)
    
    # Save results for this embedding type and K value
    df = pd.DataFrame(records)
    
    # Create filename suffix based on method
    method_suffix = "greedy" if use_greedy_sequential else "all_combinations"
    output_path = result_dir / f"distances_source{source_domain_idx}_k{k}_{embedding_type}_{method_suffix}_{config_suffix}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved K={k} results to {output_path.name}")
    
    # Cleanup
    del src_data
    if embedding_type == "geodesic":
        # Clear coordinate cache for geodesic computation
        clear_coordinate_cache()
    elif "+" in embedding_type:
        del data_source, max_const, min_const
    else:
        del embeddings, data_source
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()