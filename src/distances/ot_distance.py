#!/usr/bin/env python
"""Compute OT distances from a source domain to K target domains."""

import os, json, time, gc, copy, argparse, sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distances.cost_matrix import (compute_cost_matrix, normalize_cost_matrix, haversine_distance,
                                        cost_constants_cache_path, load_cost_constants, compute_and_cache_cost_constants)
from src.distances.combined_ot_distance import compute_combined_distance
from src.distances.utils import (save_ot_distance, load_ot_distance, get_ot_distance_cache_path, uniform_weights,
                                  solve_ot, domain_slice, load_embeddings_and_domains, load_coordinates_and_domains)

NEEDS_GLOBAL_CONSTANTS = ("max", "minmax")


def compute_ot_distance(src_embeddings, tgt_embeddings, ot_args) -> float:
    cost_ab = normalize_cost_matrix(compute_cost_matrix(src_embeddings, tgt_embeddings, ot_args.metric), ot_args)
    a = uniform_weights(src_embeddings.shape[0], src_embeddings.device)
    b = uniform_weights(tgt_embeddings.shape[0], tgt_embeddings.device)
    ot_ab = solve_ot(a, b, cost_ab, ot_args)
    if not ot_args.debiased:
        return ot_ab
    # Sinkhorn divergence S_eps(a,b) = OT_eps(a,b) - 0.5*OT_eps(a,a) - 0.5*OT_eps(b,b) corrects
    # entropic OT's regularization bias; with "max_per_domain" normalization the cancellation is
    # only approximate since each term is scaled by its own local max.
    cost_aa = normalize_cost_matrix(compute_cost_matrix(src_embeddings, src_embeddings, ot_args.metric), ot_args)
    cost_bb = normalize_cost_matrix(compute_cost_matrix(tgt_embeddings, tgt_embeddings, ot_args.metric), ot_args)
    ot_aa = solve_ot(a, a, cost_aa, ot_args)
    ot_bb = solve_ot(b, b, cost_bb, ot_args)
    return ot_ab - 0.5 * (ot_aa + ot_bb)


def load_country_mapping(args) -> dict:
    path = Path(args.data_root) / "geoyfcc" / "country_mapping.json"
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return {int(k): v for k, v in json.load(f).items()}

def get_result_dir(args) -> Path:
    return Path(args.data_root) / args.dataset_name / "distances" / "ot_distance"

def get_cost_constants(embedding_type, args, tensor):
    if embedding_type == "geodesic":
        return get_geodesic_cost_constants(args, tensor)
    cached = load_cost_constants(args.data_root, args.dataset_name, embedding_type, args.metric)
    return cached if cached is not None else compute_and_cache_cost_constants(
        args.data_root, args.dataset_name, embedding_type, args.metric, tensor)

def get_geodesic_cost_constants(args, coords_tensor, batch_size=20000, force_recompute=False):
    path = cost_constants_cache_path(args.data_root, args.dataset_name, "geodesic", "geodesic")
    if path.exists() and not force_recompute:
        with open(path, 'r') as f:
            data = json.load(f)
        return data['cost_max'], data['cost_min']

    n = len(coords_tensor)
    if n == 0:
        raise ValueError("No valid coordinates found for any domain")
    device = coords_tensor.device
    global_min, global_max = float("inf"), float("-inf")
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Geodesic cost constants"):
            end_i = min(i + batch_size, n)
            batch_i = coords_tensor[i:end_i]
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                dists = haversine_distance(batch_i, coords_tensor[j:end_j])
                if i == j:
                    dists = dists[~torch.eye(end_i - i, dtype=torch.bool, device=device)]
                global_min, global_max = min(global_min, dists.min().item()), max(global_max, dists.max().item())
                del dists
                torch.cuda.empty_cache()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'cost_min': global_min, 'cost_max': global_max}, f, indent=2)
    return global_max, global_min

def load_type_data(args, device, embedding_type):
    loader = load_coordinates_and_domains(args.dataset_name) if embedding_type == "geodesic" \
        else load_embeddings_and_domains(args.dataset_name, embedding_type)
    tensor, doms = loader
    return tensor.to(device), doms

def load_source_data(args, device, source_domain_idx, embedding_type, cache=None):
    """Load target-side data, source-domain slices, and cost-normalization constants for each embedding type.
    Pass a shared `cache` dict across calls (e.g. when sweeping over multiple source domains/lambdas
    in one process) to avoid re-reading the same multi-GB embedding files from disk each time."""
    types = embedding_type.split("+")
    needs_constants = args.normalize_cost in NEEDS_GLOBAL_CONSTANTS
    data_source, domains_source, max_const, min_const, src_data = {}, {}, {}, {}, {}

    for t in types:
        if cache is not None and t in cache:
            tensor, doms, mx, mn = cache[t]
        else:
            tensor, doms = load_type_data(args, device, t)
            mx, mn = get_cost_constants(t, args, tensor) if needs_constants else (None, None)
            if cache is not None:
                cache[t] = (tensor, doms, mx, mn)
        data_source[t], domains_source[t], max_const[t], min_const[t] = tensor, doms, mx, mn
        src_data[t] = domain_slice(tensor, doms, source_domain_idx)
        if len(src_data[t]) == 0:
            raise ValueError(f"No data found for source domain {source_domain_idx} in embedding type {t}")
        if source_domain_idx == "all" and src_data[t].shape[0] > args.source_sample_cap:
            # Pooling every domain (~1e5-1e6 rows) makes the dense OT cost matrix too large for
            # GPU; subsample deterministically to keep it the size of a single-domain OT run.
            n_before = src_data[t].shape[0]
            gen = torch.Generator(device=src_data[t].device).manual_seed(args.source_sample_seed)
            keep = torch.randperm(n_before, generator=gen, device=src_data[t].device)[:args.source_sample_cap]
            src_data[t] = src_data[t][keep]
            print(f"[INFO] source-domain-idx=all, embedding_type={t}: subsampled pooled source "
                  f"from {n_before} to {src_data[t].shape[0]} points (source_sample_cap="
                  f"{args.source_sample_cap}, seed={args.source_sample_seed}) to keep the OT cost matrix tractable.")

    if len(types) == 1:
        t = types[0]
        return src_data[t], data_source[t], domains_source[t], max_const[t], min_const[t]
    return src_data, data_source, domains_source, max_const, min_const

def compute_or_load_distance(args, src_data, tgt_domain_indices, data_source, domains_source,
                              embedding_type, max_const, min_const, source_domain_idx,
                              include_greedy_sequential_str=False, lambda_param=None):
    result_dir = get_result_dir(args)
    lambda_for_cache = lambda_param if "+" in embedding_type else None

    ot_args = copy.copy(args)
    ot_args.metric = "geodesic" if embedding_type == "geodesic" else args.metric

    cache_path = get_ot_distance_cache_path(
        result_dir=str(result_dir), embedding_type=embedding_type, src_idx=source_domain_idx,
        tgt_idx=tgt_domain_indices, ot_args=ot_args,
        include_greedy_sequential_str=include_greedy_sequential_str, lambda_param=lambda_for_cache)

    # OT distance is symmetric for a plain single-domain pair, so also check the reverse-direction
    # cache (not the pooled "all" source, which is never a valid target -- see save_k1_matrix).
    reverse_cache_path = None
    if source_domain_idx != "all" and len(tgt_domain_indices) == 1 and tgt_domain_indices[0] != source_domain_idx:
        reverse_cache_path = get_ot_distance_cache_path(
            result_dir=str(result_dir), embedding_type=embedding_type, src_idx=tgt_domain_indices[0],
            tgt_idx=[source_domain_idx], ot_args=ot_args,
            include_greedy_sequential_str=include_greedy_sequential_str, lambda_param=lambda_for_cache)

    for path in (cache_path, reverse_cache_path):
        if path and os.path.exists(path) and not args.force_recompute:
            cache_data = load_ot_distance(path)
            if cache_data:
                return cache_data['distance'], True

    t0 = time.time()
    if "+" in embedding_type:
        distance = compute_combined_distance(args, src_data, tgt_domain_indices, data_source, domains_source,
                                              embedding_type, max_const, min_const, ot_args, lambda_param)
    else:
        tgt_tensor = torch.cat([domain_slice(data_source, domains_source, idx) for idx in tgt_domain_indices])
        ot_args.max_constant, ot_args.min_constant = max_const, min_const
        distance = compute_ot_distance(src_data, tgt_tensor, ot_args)
    comp_time = time.time() - t0

    metadata = {
        'src_domain_idx': source_domain_idx, 'tgt_domain_indices': tgt_domain_indices,
        'embedding_type': embedding_type, 'method': args.method, 'reg_e': args.reg_e,
        'max_iter': args.max_iter, 'metric': ot_args.metric, 'normalize_cost': args.normalize_cost,
        'debiased': args.debiased, 'computation_time': comp_time,
        'src_shape': list(src_data.shape) if not isinstance(src_data, dict) else {k: list(v.shape) for k, v in src_data.items()},
        'timestamp': time.time(),
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_ot_distance(cache_path, distance, metadata)
    return distance, False

def greedy_sequential_ot_selection(args, src_data, all_domain_indices, data_source, domains_source,
                                    embedding_type, max_const, min_const, k,
                                    source_domain_idx, lambda_param=None):
    country_mapping = load_country_mapping(args)
    selected_domains, distances_at_each_step = [], []
    remaining_domains = [idx for idx in all_domain_indices if idx != source_domain_idx]

    for step in range(k):
        rows, best_distance, best_domain = [], float('inf'), None
        for candidate_domain in remaining_domains:
            current_target_domains = selected_domains + [candidate_domain]
            distance, _ = compute_or_load_distance(
                args, src_data, current_target_domains, data_source, domains_source,
                embedding_type, max_const, min_const, source_domain_idx,
                include_greedy_sequential_str=step > 0, lambda_param=lambda_param)
            if distance < best_distance:
                best_distance, best_domain = distance, candidate_domain
            rows.append({
                "source_domain_idx": source_domain_idx,
                "source_domain_name": country_mapping.get(source_domain_idx, f"Unknown-{source_domain_idx}"),
                "k": step + 1, "tgt_domains": current_target_domains,
                "tgt_domain_names": [country_mapping.get(i, f"Unknown-{i}") for i in current_target_domains],
                "distance": distance,
            })

        lambda_suffix = f"_lambda_{lambda_param}" if lambda_param is not None else ""
        out_dir = Path(f"greedy_sequential_distances/{embedding_type}{lambda_suffix}")
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"k{step+1}_source{source_domain_idx}.csv", index=False)

        selected_domains.append(best_domain)
        remaining_domains.remove(best_domain)
        distances_at_each_step.append(best_distance)

    print(f"Greedy selection for source {source_domain_idx}: {selected_domains}")
    return selected_domains, distances_at_each_step

def create_result_record(embedding_type, tgt_domain_indices, distance, source_domain_idx, lambda_param=None):
    record = {"embedding_type": embedding_type, "src_domain_idx": source_domain_idx,
              "k": len(tgt_domain_indices), "distance": distance}
    for i, idx in enumerate(tgt_domain_indices, 1):
        record[f"tgt_domain_{i}_idx"] = idx
    record["tgt_domains_combined"] = "+".join(str(idx) for idx in tgt_domain_indices)
    if "+" in embedding_type:
        record["lambda"] = lambda_param
    return record

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute OT distances for GeoYFCC dataset")
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory")
    parser.add_argument("--dataset-name", type=str, default="geoyfcc_text", help="Dataset folder name under the data root")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (falls back to cpu if cuda is unavailable)")
    parser.add_argument("--embedding-type", type=str, required=True, help="Embedding type to process, individual (e.g. 'bert') or combined (e.g. 'bert+geoclip')")
    parser.add_argument("--source-domain-idx", type=str, default="57", help="Comma-separated source domain index/indices, or 'all' to pool embeddings across every domain. Multiple indices are looped over within this process, reusing loaded embeddings instead of reloading them per index.")
    parser.add_argument("--source-sample-cap", type=int, default=15000, help="When --source-domain-idx=all, randomly subsample the pooled source distribution down to this many points (keeps the dense OT cost matrix the same order of magnitude as a single-domain source; a single domain here has ~7.6k-15.6k samples). Ignored otherwise.")
    parser.add_argument("--source-sample-seed", type=int, default=42, help="Seed for the --source-sample-cap subsampling, for reproducibility.")
    parser.add_argument("--total-domains", type=int, default=62, help="Total number of domains in the dataset")
    parser.add_argument("--k", type=int, default=1, help="Number of target domains K")
    parser.add_argument("--reg-e", type=float, default=0.01, help="Sinkhorn regularization parameter")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations")
    parser.add_argument("--stop-thr", type=float, default=1e-5, help="Sinkhorn marginal-convergence threshold. Empirically on this data's real cost matrices, 1e-6 gives a ~2x speedup over 1e-8 with relative error <=3e-6 (negligible); 1e-5 gives ~3x with error ~4e-5; below that (e.g. 1e-3) errors grow to ~6e-4, which starts to be worth caution.")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric for non-geodesic embeddings")
    parser.add_argument("--method", type=str, default="sinkhorn_log", help="OT method: sinkhorn_log (numerically stable, and empirically converges faster than 'sinkhorn' on this data's real cost matrices -- see solve_ot), sinkhorn (falls back to sinkhorn_log on detected numerical errors), or emd")
    parser.add_argument("--normalize-cost", type=str, default="max", help="Cost matrix normalization: none, max, minmax, or max_per_domain")
    parser.add_argument("--debiased", action="store_true", help="Compute the debiased Sinkhorn divergence S_eps(a,b) = OT_eps(a,b) - 0.5*OT_eps(a,a) - 0.5*OT_eps(b,b) instead of raw entropic OT_eps(a,b).")
    parser.add_argument("--normalize-after", action="store_true", help="Renormalize the combined cost matrix after weighting (combined embeddings only)")
    parser.add_argument("--lambda", type=float, default=None, dest="lambda_param", help="Lambda weight for combined embeddings (default: 0.5)")
    parser.add_argument("--lambda-values", type=str, default=None, help="Comma-separated lambda values to loop over within this process (combined embeddings only), reusing loaded embeddings. Overrides --lambda.")
    parser.add_argument("--greedy-sequential", action="store_true", help="Use greedy sequential domain selection instead of all combinations")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation even if cached results exist")
    return parser.parse_args(argv)

def run_all_combinations(args, src_data, data_source, domains_source, all_domain_indices, embedding_type,
                          max_const, min_const, source_domain_idx, k, lambda_param):
    records = []
    for tgt_domain_indices in tqdm(list(combinations(all_domain_indices, k)), desc=f"src={source_domain_idx} k={k}"):
        distance, _ = compute_or_load_distance(
            args, src_data, list(tgt_domain_indices), data_source, domains_source,
            embedding_type, max_const, min_const, source_domain_idx, lambda_param=lambda_param)
        record = create_result_record(embedding_type, tgt_domain_indices, distance, source_domain_idx, lambda_param)
        record["greedy_sequential"] = False
        records.append(record)
    return records

def run_greedy_sequential(args, src_data, all_domain_indices, data_source, domains_source, embedding_type,
                           max_const, min_const, source_domain_idx, k, lambda_param):
    selected_domains, distances_at_each_step = greedy_sequential_ot_selection(
        args, src_data, all_domain_indices, data_source, domains_source,
        embedding_type, max_const, min_const, k, source_domain_idx, lambda_param)
    records = []
    for step, distance in enumerate(distances_at_each_step):
        record = create_result_record(embedding_type, selected_domains[:step + 1], distance, source_domain_idx, lambda_param)
        record["selection_step"] = step + 1
        record["greedy_sequential"] = True
        records.append(record)
    return records

def save_k1_matrix(records, output_path, source_domain_idx, total_domains):
    """Merge k=1 results into a single src x tgt distance matrix CSV, shared across every
    --source-domain-idx run. OT distance is symmetric, so each entry is mirrored into the
    transposed cell -- except for the pooled "all" source, which gets its own row only (it's
    never a valid target domain, so there's no matching column to mirror into)."""
    if output_path.exists():
        matrix = pd.read_csv(output_path, index_col=0)
        matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
    else:
        domain_keys = [str(i) for i in range(total_domains)]
        matrix = pd.DataFrame(np.nan, index=domain_keys, columns=domain_keys, dtype=float)

    src_key = str(source_domain_idx)
    for record in records:
        tgt_key = str(record["tgt_domain_1_idx"])
        matrix.loc[src_key, tgt_key] = record["distance"]
        if source_domain_idx != "all":
            matrix.loc[tgt_key, src_key] = record["distance"]

    domain_order = sorted(matrix.columns, key=int)
    row_order = domain_order + (["all"] if "all" in matrix.index else [])
    matrix = matrix.reindex(index=row_order, columns=domain_order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)


def save_long_records(records, output_path, source_domain_idx):
    """Merge k>1 results into a single long-format CSV shared across every --source-domain-idx
    run, replacing any existing rows for this source domain (idempotent on re-run)."""
    new_df = pd.DataFrame(records)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing = existing[existing["src_domain_idx"].astype(str) != str(source_domain_idx)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


def save_records(records, result_dir, source_domain_idx, k, total_domains, embedding_type, method_suffix, config_suffix):
    if k == 1:
        output_path = result_dir / f"ot_distance_matrix_{embedding_type}_{method_suffix}_{config_suffix}.csv"
        save_k1_matrix(records, output_path, source_domain_idx, total_domains)
    else:
        output_path = result_dir / f"distances_k{k}_{embedding_type}_{method_suffix}_{config_suffix}.csv"
        save_long_records(records, output_path, source_domain_idx)
    print(f"Saved K={k} results to {output_path}")

def cleanup_source_data(src_data, data_source):
    del src_data, data_source
    torch.cuda.empty_cache()
    gc.collect()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    source_domain_indices = [s if s == "all" else int(s) for s in args.source_domain_idx.split(",")]
    lambda_values = [float(x) for x in args.lambda_values.split(",")] if args.lambda_values else [args.lambda_param]
    embedding_type, k = args.embedding_type, args.k
    all_domain_indices = list(range(args.total_domains))
    metric_to_use = "geodesic" if embedding_type == "geodesic" else args.metric

    cache = {}
    for lambda_param in lambda_values:
        config_suffix = f"method_{args.method}_reg_{args.reg_e}_iter_{args.max_iter}_metric_{metric_to_use}_norm_{args.normalize_cost}"
        if args.debiased:
            config_suffix += "_debiased"
        if "+" in embedding_type and lambda_param is not None:
            config_suffix += f"_lambda_{lambda_param}"

        for source_domain_idx in source_domain_indices:
            src_data, data_source, domains_source, max_const, min_const = load_source_data(
                args, device, source_domain_idx, embedding_type, cache=cache)

            if args.greedy_sequential:
                records = run_greedy_sequential(args, src_data, all_domain_indices, data_source, domains_source, embedding_type,
                                                 max_const, min_const, source_domain_idx, k, lambda_param)
                method_suffix = "greedy"
            else:
                records = run_all_combinations(args, src_data, data_source, domains_source, all_domain_indices, embedding_type,
                                                max_const, min_const, source_domain_idx, k, lambda_param)
                method_suffix = "all_combinations"

            save_records(records, get_result_dir(args), source_domain_idx, k, args.total_domains, embedding_type, method_suffix, config_suffix)
            cleanup_source_data(src_data, None)

if __name__ == "__main__":
    main(parse_args())
