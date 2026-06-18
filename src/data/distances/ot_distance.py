#!/usr/bin/env python
"""
Compute OT distances from source domain to K target domains (K=1,2,3)
"""

import os
import json
import time
import gc
import copy
import argparse
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

from compute_distances.ot_distance import (
    compute_ot_distance,
    compute_combined_ot_distance,
    cosine_distance_minmax,
    haversine_distance,
)
from compute_distances.utils import (
    save_ot_distance,
    load_ot_distance,
    get_ot_distance_cache_path,
)

device = dataset = domains = None
_coordinate_cache = {}

def initialize(args):
    """Initialize global dataset/device state from CLI args."""
    global device, dataset, domains, _coordinate_cache

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = GeoYFCCText(root=f"{args.data_root}/{args.dataset_name}", split='train')
    domains = np.array(list(dataset.df["country_id"]))
    _coordinate_cache = {}

def load_country_mapping(args) -> dict:
    path = Path(args.data_root) / "geoyfcc" / "country_mapping.json"
    if not path.exists():
        print(f"Warning: Country mapping file not found at {path}")
        return {}
    try:
        with open(path, 'r') as f:
            return {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print(f"Warning: Could not load country mapping from {path}: {e}")
        return {}

def get_embedding_path(args, embedding_type) -> Path:
    return Path(args.data_root) / args.dataset_name / "embeddings" / f"{embedding_type}.pt"

def get_result_dir(args) -> Path:
    return Path(args.data_root) / args.dataset_name / "distances" / "ot_distance"

def extract_domain_embeddings(embeddings, domains_array, domain_idx, embedding_type=None):
    """Extract embeddings for a specific domain"""
    if isinstance(embeddings, dict):
        assert embedding_type is not None
        if embedding_type == "geodesic":
            return extract_domain_coordinates(dataset, domains_array, domain_idx)
        embeddings = embeddings[embedding_type]
    return embeddings[domains_array == domain_idx]

def extract_domain_coordinates(dataset, domains_array, domain_idx):
    """Extract lat/lon coordinates for a specific domain with caching and GPU acceleration"""
    if domain_idx in _coordinate_cache:
        return _coordinate_cache[domain_idx]

    coords = dataset.df[domains_array == domain_idx][['lat', 'lon']].dropna()
    if len(coords) == 0:
        print(f"Warning: No valid coordinates found for domain {domain_idx}")
        coords_tensor = torch.tensor([], dtype=torch.float32, device=device)
    else:
        coords_tensor = torch.tensor(coords.values, dtype=torch.float32, device=device)
        print(f"  Domain {domain_idx}: {len(coords_tensor)} samples with coordinates")

    _coordinate_cache[domain_idx] = coords_tensor
    return coords_tensor

def clear_coordinate_cache():
    """Clear the coordinate cache to free GPU memory"""
    _coordinate_cache.clear()
    torch.cuda.empty_cache()
    print("Coordinate cache cleared")

def get_cost_constants(embedding_type, args):
    """Get or compute min/max cost constants for normalization"""
    if embedding_type == "geodesic":
        return get_geodesic_cost_constants(args)

    path = Path(args.data_root) / args.dataset_name / f"{embedding_type}_cost_matrix_data.json"
    if path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
            return data['cost_max'], data['cost_min']

    embeddings = torch.load(get_embedding_path(args, embedding_type), map_location="cuda")
    min_val, max_val = cosine_distance_minmax(embeddings, embeddings)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'cost_min': float(min_val), 'cost_max': float(max_val)}, f, indent=2)

    del embeddings
    torch.cuda.empty_cache()
    gc.collect()

    return max_val, min_val

def get_geodesic_cost_constants(args, batch_size=20000, force_recompute=False):
    """Get or compute min/max cost constants for geodesic distance"""
    print("Getting Geodesic Cost Constants...")
    path = Path(args.data_root) / args.dataset_name / "geodesic_cost_matrix_data.json"

    if path.exists() and not force_recompute:
        with open(path, 'r') as f:
            data = json.load(f)
        return data['cost_max'], data['cost_min']

    print("Computing geodesic cost constants from lat/lon coordinates...")
    all_coords = [c for idx in range(args.total_domains)
                  if len(c := extract_domain_coordinates(dataset, domains, idx)) > 0]

    if not all_coords:
        raise ValueError("Warning: No valid coordinates found for any domain")

    all_coords_tensor = torch.cat(all_coords, dim=0)
    device = all_coords_tensor.device
    n = len(all_coords_tensor)

    print(f"Computing geodesic distances for {n} coordinates (batched)...")

    global_min = float("inf")
    global_max = float("-inf")

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Computing geodesic batches (outer)", leave=True):
            end_i = min(i + batch_size, n)
            batch_i = all_coords_tensor[i:end_i]

            for j in tqdm(range(0, n, batch_size), desc=f"  Inner loop for batch {i//batch_size+1}", leave=False):
                end_j = min(j + batch_size, n)
                dists = haversine_distance(batch_i, all_coords_tensor[j:end_j])

                if i == j:
                    mask = ~torch.eye(end_i - i, dtype=torch.bool, device=device)
                    dists = dists[mask]

                global_min = min(global_min, dists.min().item())
                global_max = max(global_max, dists.max().item())

                del dists
                torch.cuda.empty_cache()

    print(f"Geodesic cost constants: min={global_min:.2f} km, max={global_max:.2f} km")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'cost_min': global_min, 'cost_max': global_max}, f, indent=2)

    del all_coords_tensor
    torch.cuda.empty_cache()
    gc.collect()

    return global_max, global_min

def compute_or_load_distance(args, src_data, tgt_domain_indices, embeddings_or_dataset, domains,
                              embedding_type, max_const, min_const, source_domain_idx,
                              include_greedy_sequential_str=False, lambda_param=None):
    """Compute or load cached OT distance"""

    result_dir = get_result_dir(args)
    lambda_for_cache = lambda_param if "+" in embedding_type else None

    ot_args = copy.copy(args)
    ot_args.metric = "geodesic" if embedding_type == "geodesic" else args.metric

    cache_path = get_ot_distance_cache_path(
        result_dir=str(result_dir),
        embedding_type=embedding_type,
        src_idx=source_domain_idx,
        tgt_idx=tgt_domain_indices,
        ot_args=ot_args,
        include_greedy_sequential_str=include_greedy_sequential_str,
        lambda_param=lambda_for_cache,
    )

    if os.path.exists(cache_path) and not args.force_recompute:
        cache_data = load_ot_distance(cache_path)
        print(f"Loaded from cache {cache_path}", flush=True)
        if cache_data:
            return cache_data['distance'], True
    elif args.force_recompute and os.path.exists(cache_path):
        print(f"Force recompute enabled - ignoring cache {cache_path}", flush=True)

    t0 = time.time()
    if embedding_type == "geodesic":
        tgt_coords = torch.cat([extract_domain_coordinates(embeddings_or_dataset, domains, idx)
                                 for idx in tgt_domain_indices], dim=0)
        ot_args.max_constant, ot_args.min_constant = max_const, min_const
        distance = compute_ot_distance(src_data, tgt_coords, ot_args)
    elif "+" in embedding_type:
        emb_type_1, emb_type_2 = embedding_type.split("+")

        tgt_emb_1 = torch.cat([extract_domain_embeddings(embeddings_or_dataset, domains, idx, emb_type_1)
                                for idx in tgt_domain_indices], dim=0)
        tgt_emb_2 = torch.cat([extract_domain_embeddings(embeddings_or_dataset, domains, idx, emb_type_2)
                                for idx in tgt_domain_indices], dim=0)

        cost_args_1 = copy.copy(args)
        cost_args_1.metric = "geodesic" if emb_type_1 == "geodesic" else args.metric
        cost_args_1.max_constant, cost_args_1.min_constant = max_const[emb_type_1], min_const[emb_type_1]

        cost_args_2 = copy.copy(args)
        cost_args_2.metric = "geodesic" if emb_type_2 == "geodesic" else args.metric
        cost_args_2.max_constant, cost_args_2.min_constant = max_const[emb_type_2], min_const[emb_type_2]

        ot_args.lambda_param = lambda_param if lambda_param is not None else 0.5
        ot_args.normalize_after = args.normalize_after

        distance = compute_combined_ot_distance(
            src_data[emb_type_1], src_data[emb_type_2],
            tgt_emb_1, tgt_emb_2,
            cost_args_1, cost_args_2, ot_args
        )
        print(f"Computed combined distance: {distance}", flush=True)
    else:
        tgt_embeddings = torch.cat([extract_domain_embeddings(embeddings_or_dataset, domains, idx)
                                     for idx in tgt_domain_indices], dim=0)
        ot_args.max_constant, ot_args.min_constant = max_const, min_const
        distance = compute_ot_distance(src_data, tgt_embeddings, ot_args)
    comp_time = time.time() - t0

    metadata = {
        'src_domain_idx': source_domain_idx,
        'tgt_domain_indices': tgt_domain_indices,
        'embedding_type': embedding_type,
        'method': args.method,
        'reg_e': args.reg_e,
        'max_iter': args.max_iter,
        'metric': ot_args.metric,
        'normalize_cost': args.normalize_cost,
        'computation_time': comp_time,
        'src_shape': list(src_data.shape) if not isinstance(src_data, dict) else {k: list(v.shape) for k, v in src_data.items()},
        'timestamp': time.time()
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_ot_distance(cache_path, distance, metadata)

    torch.cuda.empty_cache()
    gc.collect()

    print(f"Distance computed {distance} for {embedding_type} from {source_domain_idx} to {tgt_domain_indices}")

    return distance, False

def greedy_sequential_ot_selection(args, src_data, all_domain_indices, data_source, domains,
                                    embedding_type, max_const, min_const, k,
                                    source_domain_idx, lambda_param=None):
    """Greedy sequential OT domain selection: pick the best next target domain at each step."""
    print(f"\n--- Greedy Sequential Selection for K={k} ---")

    country_mapping = load_country_mapping(args)

    selected_domains = []
    distances_at_each_step = []
    remaining_domains = [idx for idx in all_domain_indices if idx != source_domain_idx]

    for step in range(k):
        rows = []
        print(f"\nStep {step + 1}: Selecting domain {step + 1}/{k}")

        best_distance, best_domain = float('inf'), None

        for candidate_domain in remaining_domains:
            current_target_domains = selected_domains + [candidate_domain]
            print(f"  Evaluating: {current_target_domains}")

            distance, from_cache = compute_or_load_distance(
                args, src_data, current_target_domains, data_source, domains,
                embedding_type, max_const, min_const, source_domain_idx,
                include_greedy_sequential_str=step > 0, lambda_param=lambda_param
            )
            print(f"    Distance: {distance:.6f} ({'cached' if from_cache else 'computed'})")

            torch.cuda.empty_cache()
            gc.collect()

            if distance < best_distance:
                best_distance, best_domain = distance, candidate_domain

            source_domain_name = country_mapping.get(source_domain_idx, f"Unknown-{source_domain_idx}")
            rows.append({
                "source_domain_idx": source_domain_idx,
                "source_domain_name": source_domain_name,
                "k": step + 1,
                "tgt_domains": current_target_domains,
                "tgt_domain_names": [country_mapping.get(i, f"Unknown-{i}") for i in current_target_domains],
                "distance": distance,
            })

        out_dir = Path(f"greedy_sequential_distances/{embedding_type}")
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"k{step+1}_source{source_domain_idx}.csv", index=False)

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
        "distance": distance,
    }
    for i, idx in enumerate(tgt_domain_indices, 1):
        record[f"tgt_domain_{i}_idx"] = idx
    record["tgt_domains_combined"] = "+".join(str(idx) for idx in tgt_domain_indices)
    return record

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Compute OT distances for GeoYFCC dataset")
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory")
    parser.add_argument("--dataset-name", type=str, default="geoyfcc_text", help="Dataset folder name under the data root")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (falls back to cpu if cuda is unavailable)")
    parser.add_argument("--embedding-type", type=str, required=True, help="Embedding type to process, individual (e.g. 'bert') or combined (e.g. 'bert+geoclip')")
    parser.add_argument("--source-domain-idx", type=int, default=57, help="Source domain index")
    parser.add_argument("--total-domains", type=int, default=62, help="Total number of domains in the dataset")
    parser.add_argument("--k", type=int, default=1, help="Number of target domains K")
    parser.add_argument("--reg-e", type=float, default=0.01, help="Sinkhorn regularization parameter")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric for non-geodesic embeddings")
    parser.add_argument("--method", type=str, default="sinkhorn", help="OT method: sinkhorn, sinkhorn_log, or emd")
    parser.add_argument("--normalize-cost", type=str, default="max", help="Cost matrix normalization: none, max, minmax, or max_per_domain")
    parser.add_argument("--normalize-after", action="store_true", help="Renormalize the combined cost matrix after weighting (combined embeddings only)")
    parser.add_argument("--lambda", type=float, default=None, dest="lambda_param", help="Lambda weight for combined embeddings (default: 0.5)")
    parser.add_argument("--greedy-sequential", action="store_true", help="Use greedy sequential domain selection instead of all combinations")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation even if cached results exist")
    return parser.parse_args()

def main():
    args = parse_args()
    initialize(args)

    source_domain_idx = args.source_domain_idx
    embedding_type = args.embedding_type
    k = args.k
    use_greedy_sequential = args.greedy_sequential
    lambda_param = args.lambda_param

    result_dir = get_result_dir(args)

    print(f"\n{'='*60}")
    print(f"Processing {embedding_type} embeddings")
    print(f"Source Domain ID: {source_domain_idx}")
    print(f"OT Config: method={args.method}, reg_e={args.reg_e}, max_iter={args.max_iter}, "
          f"metric={args.metric}, normalize_cost={args.normalize_cost}")
    print(f"Force Recompute: {args.force_recompute}")
    print(f"K Value: {k}")
    print(f"Greedy Sequential: {use_greedy_sequential}")
    print(f"{'='*60}")

    all_domain_indices = list(range(args.total_domains))

    metric_to_use = "geodesic" if embedding_type == "geodesic" else args.metric
    config_suffix = f"method_{args.method}_reg_{args.reg_e}_iter_{args.max_iter}_metric_{metric_to_use}_norm_{args.normalize_cost}"

    if embedding_type == "geodesic":
        print("Using lat/lon coordinates for geodesic distance computation")
        max_const, min_const = get_cost_constants(embedding_type, args)
        print(f"Cost constants: min={min_const:.6f} km, max={max_const:.6f} km")

        src_data = extract_domain_coordinates(dataset, domains, source_domain_idx)
        if len(src_data) == 0:
            print(f"✗ No valid coordinates found for source domain {source_domain_idx}")
            return

        data_source = dataset
    elif "+" in embedding_type:
        src_data, data_source, max_const, min_const = {}, {}, {}, {}

        for emb_type in embedding_type.split("+"):
            if emb_type == "geodesic":
                src_data[emb_type] = extract_domain_coordinates(dataset, domains, source_domain_idx)
            else:
                embedding_path = get_embedding_path(args, emb_type)
                if not embedding_path.exists():
                    print(f"✗ Embedding file not found: {embedding_path}")
                    return
                embeddings = torch.load(embedding_path, map_location="cuda")
                data_source[emb_type] = embeddings
                src_data[emb_type] = extract_domain_embeddings(embeddings, domains, source_domain_idx)
            max_const[emb_type], min_const[emb_type] = get_cost_constants(emb_type, args)
    else:
        embedding_path = get_embedding_path(args, embedding_type)
        if not embedding_path.exists():
            print(f"✗ Embedding file not found: {embedding_path}")
            return

        embeddings = torch.load(embedding_path, map_location="cuda")
        max_const, min_const = get_cost_constants(embedding_type, args)
        print(f"Cost constants: min={min_const:.6f}, max={max_const:.6f}")

        src_data = extract_domain_embeddings(embeddings, domains, source_domain_idx)
        data_source = embeddings

    print(f"\n--- K={k} target domains ---")

    records = []

    if use_greedy_sequential:
        print("Using greedy sequential domain selection")
        selected_domains, distances_at_each_step = greedy_sequential_ot_selection(
            args, src_data, all_domain_indices, data_source, domains,
            embedding_type, max_const, min_const, k, source_domain_idx, lambda_param
        )

        for step, distance in enumerate(distances_at_each_step):
            record = create_result_record(embedding_type, selected_domains[:step + 1], distance, source_domain_idx)
            record["selection_step"] = step + 1
            record["greedy_sequential"] = True
            records.append(record)

        print(f"✓ Greedy sequential selection completed")
    else:
        tgt_combinations = list(combinations(all_domain_indices, k))
        print(f"Computing {len(tgt_combinations)} combinations...")

        for idx, tgt_domain_indices in enumerate(tgt_combinations, 1):
            distance, from_cache = compute_or_load_distance(
                args, src_data, list(tgt_domain_indices), data_source, domains,
                embedding_type, max_const, min_const, source_domain_idx,
                lambda_param=lambda_param
            )

            if idx % 100 == 0 or idx == len(tgt_combinations):
                status = "cached" if from_cache else "computed"
                print(f"  [{idx}/{len(tgt_combinations)}] {tgt_domain_indices}: {distance:.6f} ({status})")

            record = create_result_record(embedding_type, tgt_domain_indices, distance, source_domain_idx)
            record["greedy_sequential"] = False
            records.append(record)

    method_suffix = "greedy" if use_greedy_sequential else "all_combinations"
    output_path = result_dir / f"distances_source{source_domain_idx}_k{k}_{embedding_type}_{method_suffix}_{config_suffix}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"✓ Saved K={k} results to {output_path.name}")

    del src_data
    if embedding_type == "geodesic":
        clear_coordinate_cache()
    elif "+" in embedding_type:
        del data_source, max_const, min_const
    else:
        del embeddings, data_source
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
