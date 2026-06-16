import os
import pickle
import torch
from typing import Dict, List, Optional, Union


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
    filename = (f"ot_distance_{ot_args.normalize_cost}_{embedding_type}_"
                f"{src_str}_to_{tgt_str}_{ot_args.method}_eps_{ot_args.reg_e}_"
                f"maxIter_{ot_args.max_iter}_{metric}{greedy_str}{lambda_str}.pkl")
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
