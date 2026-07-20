"""OT distance for combined embedding types (e.g. 'bert+geoclip'): a weighted sum of per-type cost matrices."""

import copy
import torch

from src.distances.cost_matrix import compute_cost_matrix, normalize_cost_matrix
from src.distances.utils import uniform_weights, solve_ot, domain_slice


def _combine_cost_matrices(emb1_x, emb1_y, emb2_x, emb2_y, cost_args1, cost_args2, ot_args):
    cost_matrix_1 = normalize_cost_matrix(compute_cost_matrix(emb1_x, emb1_y, cost_args1.metric), cost_args1)
    cost_matrix_2 = normalize_cost_matrix(compute_cost_matrix(emb2_x, emb2_y, cost_args2.metric), cost_args2)

    combined = ot_args.lambda_param * cost_matrix_1 + (1 - ot_args.lambda_param) * cost_matrix_2
    if ot_args.normalize_after:
        combined = combined / torch.max(combined)
    return combined


def compute_combined_ot_distance(src_emb1, src_emb2, tgt_emb1, tgt_emb2, cost_args1, cost_args2, ot_args) -> float:
    combined_ab = _combine_cost_matrices(src_emb1, tgt_emb1, src_emb2, tgt_emb2, cost_args1, cost_args2, ot_args)

    a = uniform_weights(combined_ab.shape[0], combined_ab.device)
    b = uniform_weights(combined_ab.shape[1], combined_ab.device)
    ot_ab = solve_ot(a, b, combined_ab, ot_args)

    if not ot_args.debiased:
        return ot_ab

    # Sinkhorn divergence: S_eps(a,b) = OT_eps(a,b) - 0.5*OT_eps(a,a) - 0.5*OT_eps(b,b),
    # applied to the same lambda-weighted combined cost matrix used for the a-b term.
    # Caveat: with normalize_cost="max_per_domain" each per-type cost matrix is scaled by its
    # own local max, so the bias cancellation is only approximate (exact under "max"/"minmax").
    combined_aa = _combine_cost_matrices(src_emb1, src_emb1, src_emb2, src_emb2, cost_args1, cost_args2, ot_args)
    combined_bb = _combine_cost_matrices(tgt_emb1, tgt_emb1, tgt_emb2, tgt_emb2, cost_args1, cost_args2, ot_args)
    ot_aa = solve_ot(a, a, combined_aa, ot_args)
    ot_bb = solve_ot(b, b, combined_bb, ot_args)
    return ot_ab - 0.5 * (ot_aa + ot_bb)


def compute_combined_distance(args, src_data, tgt_domain_indices, data_source, domains_source,
                               embedding_type, max_const, min_const, ot_args, lambda_param=None) -> float:
    """Resolve the '+' branch of compute_or_load_distance: extract per-type target embeddings and combine."""
    emb_type_1, emb_type_2 = embedding_type.split("+")
    tgt_emb_1 = torch.cat([domain_slice(data_source[emb_type_1], domains_source[emb_type_1], idx) for idx in tgt_domain_indices])
    tgt_emb_2 = torch.cat([domain_slice(data_source[emb_type_2], domains_source[emb_type_2], idx) for idx in tgt_domain_indices])

    cost_args_1, cost_args_2 = copy.copy(args), copy.copy(args)
    cost_args_1.metric = "geodesic" if emb_type_1 == "geodesic" else args.metric
    cost_args_1.max_constant, cost_args_1.min_constant = max_const[emb_type_1], min_const[emb_type_1]
    cost_args_2.metric = "geodesic" if emb_type_2 == "geodesic" else args.metric
    cost_args_2.max_constant, cost_args_2.min_constant = max_const[emb_type_2], min_const[emb_type_2]

    ot_args.lambda_param = lambda_param if lambda_param is not None else 0.5
    ot_args.normalize_after = args.normalize_after

    return compute_combined_ot_distance(src_data[emb_type_1], src_data[emb_type_2], tgt_emb_1, tgt_emb_2,
                                         cost_args_1, cost_args_2, ot_args)
