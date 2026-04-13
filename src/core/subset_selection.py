from typing import Tuple, List, Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd
import os
import csv
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '../..'))

from src.data.load_datasets import get_domain_split_mask


def _select_best_ot_combination(
    ot_params: Dict[str, Any],
    candidate_pool: Sequence[int],
    exclude_domains_list: Sequence[int],
    num_domains: int,
) -> List[int]:
    """
    Read the OT distances CSV specified by ot_params and return the K-target domain
    combination (as a list of ints) that yields the minimum distance for the given
    source domain, subject to filters (candidate pool, exclusions, exact K).

    Required ot_params keys:
      - ot_distance_dir, source_domain_idx, embedding_type, method, reg, iter, metric, norm
    """
    ot_dir = ot_params.get('ot_distance_dir')
    source_domain_idx = ot_params.get('source_domain_idx')
    embedding_type = ot_params.get('embedding_type')
    # Defaults for convenience
    ot_method = ot_params.get('method') or 'sinkhorn'
    reg = ot_params.get('reg') or '0.01'
    iters = ot_params.get('iter') or '1000'
    metric = ot_params.get('metric') or 'cosine'
    norm = ot_params.get('norm') or 'max'

    if None in [ot_dir, source_domain_idx, embedding_type, num_domains]:
        raise ValueError("Missing required OT parameters: ot_distance_dir, source_domain_idx, embedding_type, method, reg, iter, metric, norm, num_domains")

    filename = f"distances_source{source_domain_idx}_k{num_domains}_{embedding_type}_greedy_method_{ot_method}_reg_{reg}_iter_{iters}_metric_{metric}_norm_{norm}.csv"
    csv_path = os.path.join(ot_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OT distances file not found: {csv_path}")

    best_row = None
    best_distance = None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Validate source domain
            try:
                src_idx = int(row.get('src_domain_idx'))
                if src_idx != int(source_domain_idx):
                    continue
            except (ValueError, TypeError):
                continue
            
            # Parse target domains
            combo = row.get('tgt_domains_combined', '')
            if not combo:
                continue
            try:
                tgt_list = [int(tok) for tok in combo.split('+') if tok]
            except ValueError:
                continue
            
            # Apply filters
            if (len(tgt_list) != int(num_domains) or
                any(t in exclude_domains_list for t in tgt_list) or
                any(t not in candidate_pool for t in tgt_list)):
                continue
            
            # Get distance
            try:
                dist = float(row.get('distance'))
            except (ValueError, TypeError):
                continue
            
            # Update best if better
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_row = tgt_list

    if best_row is None:
        from IPython import embed; embed()
        raise ValueError(f"No valid OT candidate combination found in {csv_path} after applying filters")
    return best_row

def choose_candidate_domains(
    all_domains: Sequence[int],
    num_domains: Optional[int] = None,
    method: str = 'random',
    seed: int = 42,
    exclude_domains: Optional[Sequence[int]] = None,
    ot_params: Optional[Dict[str, Any]] = None,
    specific_domain: Optional[int] = None,
) -> List[int]:
    """
    Choose candidate domains to construct the pretraining pool.

    Args:
        all_domains: Pool of possible domains (e.g., config PRETRAIN_DOMAINS)
        num_domains: How many to choose. If None, choose all.
        method: 'random', 'ot', or 'specific_domain'
        seed: RNG seed
        exclude_domains: Domains to exclude (e.g., target domain)
        ot_params: Parameters for OT method
        specific_domain: Specific domain to select (used when method='specific_domain')

    Returns:
        List of chosen domain ids.
    """
    if exclude_domains is None:
        exclude_domains_list = []
    elif isinstance(exclude_domains, int):
        exclude_domains_list = [exclude_domains]
    else:
        exclude_domains_list = list(exclude_domains)

    # Default candidate pool after exclusion
    candidate_pool = [d for d in all_domains if d not in set(exclude_domains_list)]

    if method == 'in_distribution':
        if specific_domain is None:
            raise ValueError("in_distribution must be provided when method='in_distribution'")
        if specific_domain not in all_domains:
            raise ValueError(f"in_distribution {specific_domain} not found in all_domains {all_domains}")
        return [specific_domain]

    if method == 'global':
        # Return all domains except the excluded ones (typically the target domain)
        # This is used when K should equal max K (all domains except target)
        return list(candidate_pool)

    if method == 'random' or method is None:
        if num_domains is None or num_domains >= len(candidate_pool):
            return list(candidate_pool)
        rng = np.random.RandomState(seed)
        return list(rng.choice(candidate_pool, size=num_domains, replace=False))

    if method == 'ot':
        if ot_params is None:
            raise ValueError("ot_params must be provided when method='ot'")
        return _select_best_ot_combination(ot_params, candidate_pool, exclude_domains_list, int(num_domains))

    # Fallback to random
    if num_domains is None or num_domains >= len(candidate_pool):
        return list(candidate_pool)
    rng = np.random.RandomState(seed)
    return list(rng.choice(candidate_pool, size=num_domains, replace=False))


def uniformly_sample_across_domains(
    dataset_name: str,
    dataset,
    candidate_domains: Sequence[int],
    split: str,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Given a set of domains and total budget B, sample as uniformly as possible
    the same number from each domain until B samples are collected.

    - Handles domains with fewer available samples by redistributing the remainder
      to other domains.
    - If B >= total available, returns mask of all available samples.

    Returns:
        Boolean mask over the entire dataset with exactly min(B, total_available) True values.
    """
    rng = np.random.RandomState(seed)

    # Collect available indices per domain for the requested split
    domain_to_indices = {}
    for d in candidate_domains:
        mask_d = get_domain_split_mask(dataset_name, dataset, d, split=split)
        domain_to_indices[d] = np.where(mask_d)[0]

    total_available = sum(len(idxs) for idxs in domain_to_indices.values())
    if budget >= total_available:
        # Use all available
        full_mask = np.zeros(len(dataset), dtype=bool)
        for idxs in domain_to_indices.values():
            full_mask[idxs] = True
        return full_mask

    # Start with equal base quota per domain
    num_domains = len(candidate_domains)
    base_quota = budget // num_domains
    remainder = budget % num_domains

    # Initial quotas: distribute remainder round-robin
    quotas = {d: base_quota for d in candidate_domains}
    for i, d in enumerate(candidate_domains):
        if i < remainder:
            quotas[d] += 1

    # Ensure quotas don't exceed availability; collect deficit to redistribute
    surplus_needed = 0
    for d in candidate_domains:
        available = len(domain_to_indices[d])
        if quotas[d] > available:
            surplus_needed += quotas[d] - available
            quotas[d] = available

    # Redistribute remaining quota to domains with remaining capacity
    if surplus_needed > 0:
        # Create a list of domains with spare capacity
        expandable = [d for d in candidate_domains if quotas[d] < len(domain_to_indices[d])]
        while surplus_needed > 0 and expandable:
            made_progress = False
            for d in list(expandable):
                if quotas[d] < len(domain_to_indices[d]):
                    quotas[d] += 1
                    surplus_needed -= 1
                    made_progress = True
                    if quotas[d] == len(domain_to_indices[d]):
                        expandable.remove(d)
                    if surplus_needed == 0:
                        break
            if not made_progress:
                # No domain can absorb more
                break

    # Sample per domain according to quotas
    selected_indices = []
    for d in candidate_domains:
        idxs = domain_to_indices[d]
        q = quotas[d]
        if q <= 0:
            continue
        if q >= len(idxs):
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=q, replace=False)
        selected_indices.append(np.array(chosen, dtype=np.int64))

    if selected_indices:
        selected_indices = np.concatenate(selected_indices)
    else:
        selected_indices = np.array([], dtype=np.int64)

    # If due to rounding/availability we selected fewer than budget, top up uniformly
    selected_set = set(selected_indices.tolist())
    while len(selected_set) < min(budget, total_available):
        for d in candidate_domains:
            if len(selected_set) >= min(budget, total_available):
                break
            idxs = domain_to_indices[d]
            # Add one more if possible
            remaining = [i for i in idxs if i not in selected_set]
            if remaining:
                selected = int(rng.choice(remaining, size=1)[0])
                selected_set.add(selected)

    final_indices = np.fromiter(selected_set, dtype=np.int64)
    final_mask = np.zeros(len(dataset), dtype=bool)
    final_mask[final_indices] = True
    return final_mask