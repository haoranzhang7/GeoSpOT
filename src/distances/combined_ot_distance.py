"""OT distance for combined embedding types (e.g. 'bert+geoclip'): a weighted sum of per-type cost matrices."""

import copy
import torch

from src.distances.cost_matrix import compute_cost_matrix, normalize_cost_matrix
from src.distances.utils import uniform_weights, solve_ot


def compute_combined_ot_distance(src_emb1, src_emb2, tgt_emb1, tgt_emb2, cost_args1, cost_args2, ot_args) -> float:
    cost_matrix_1 = normalize_cost_matrix(compute_cost_matrix(src_emb1, tgt_emb1, cost_args1.metric), cost_args1)
    cost_matrix_2 = normalize_cost_matrix(compute_cost_matrix(src_emb2, tgt_emb2, cost_args2.metric), cost_args2)

    combined = ot_args.lambda_param * cost_matrix_1 + (1 - ot_args.lambda_param) * cost_matrix_2
    if ot_args.normalize_after:
        combined = combined / torch.max(combined)

    a = uniform_weights(combined.shape[0], combined.device)
    b = uniform_weights(combined.shape[1], combined.device)
    return solve_ot(a, b, combined, ot_args)


def compute_combined_distance(args, src_data, tgt_domain_indices, embeddings_or_dataset, domains,
                               embedding_type, max_const, min_const, ot_args, lambda_param=None) -> float:
    """Resolve the '+' branch of compute_or_load_distance: extract per-type target embeddings and combine."""
    from src.distances.ot_distance import extract_domain_embeddings

    emb_type_1, emb_type_2 = embedding_type.split("+")
    tgt_emb_1 = torch.cat([extract_domain_embeddings(embeddings_or_dataset, domains, idx, emb_type_1) for idx in tgt_domain_indices])
    tgt_emb_2 = torch.cat([extract_domain_embeddings(embeddings_or_dataset, domains, idx, emb_type_2) for idx in tgt_domain_indices])

    cost_args_1, cost_args_2 = copy.copy(args), copy.copy(args)
    cost_args_1.metric = "geodesic" if emb_type_1 == "geodesic" else args.metric
    cost_args_1.max_constant, cost_args_1.min_constant = max_const[emb_type_1], min_const[emb_type_1]
    cost_args_2.metric = "geodesic" if emb_type_2 == "geodesic" else args.metric
    cost_args_2.max_constant, cost_args_2.min_constant = max_const[emb_type_2], min_const[emb_type_2]

    ot_args.lambda_param = lambda_param if lambda_param is not None else 0.5
    ot_args.normalize_after = args.normalize_after

    distance = compute_combined_ot_distance(src_data[emb_type_1], src_data[emb_type_2], tgt_emb_1, tgt_emb_2,
                                              cost_args_1, cost_args_2, ot_args)
    print(f"Computed combined distance: {distance}", flush=True)
    return distance
