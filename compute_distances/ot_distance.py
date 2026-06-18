import torch
import ot


def haversine_distance(src, tgt, radius=6371.0):
    src_rad, tgt_rad = torch.deg2rad(src), torch.deg2rad(tgt)

    lat1, lon1 = src_rad[:, 0:1], src_rad[:, 1:2]
    lat2, lon2 = tgt_rad[:, 0], tgt_rad[:, 1]

    a = (torch.sin((lat1 - lat2) / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon1 - lon2) / 2) ** 2)
    
    return radius * 2 * torch.arcsin(torch.sqrt(a.clamp(max=1.0)))

def cosine_distance_minmax(src_embeddings, tgt_embeddings):
    src_norm = src_embeddings / src_embeddings.norm(dim=1, keepdim=True)
    tgt_norm = tgt_embeddings / tgt_embeddings.norm(dim=1, keepdim=True)

    distances = 1 - src_norm @ tgt_norm.T

    return distances.min().item(), distances.max().item()


def compute_cost_matrix(src_embeddings, tgt_embeddings, metric: str):
    if metric == "cosine":
        src_norm = src_embeddings / src_embeddings.norm(dim=1, keepdim=True)
        tgt_norm = tgt_embeddings / tgt_embeddings.norm(dim=1, keepdim=True)
        return 1 - torch.matmul(src_norm, tgt_norm.T)
    
    elif metric == "geodesic":
        if src_embeddings.shape[1] != 2 or tgt_embeddings.shape[1] != 2:
            raise ValueError(f"Geodesic requires 2D coords, got src: {src_embeddings.shape[1]}, tgt: {tgt_embeddings.shape[1]}")
        return haversine_distance(src_embeddings, tgt_embeddings)
    
    else:
        return ot.dist(src_embeddings, tgt_embeddings, metric=metric)


def normalize_cost_matrix(cost_matrix, args):
    
    if args.normalize_cost in ("none", None):
        return cost_matrix
    
    elif args.normalize_cost == "max":
        return cost_matrix / args.max_constant
    
    elif args.normalize_cost == "minmax":
        return (cost_matrix - args.min_constant) / (args.max_constant - args.min_constant)
    
    elif args.normalize_cost in ("max_per_domain", "max_per_domain_and_normalized_after"):
        return cost_matrix / torch.max(cost_matrix)
    
    else:
        raise ValueError(f"Unknown normalize_cost: {args.normalize_cost}")


def sinkhorn_distance(a, b, cost_matrix, ot_args) -> float:
    distance = ot.sinkhorn2(a, b, cost_matrix, method=ot_args.method,
                            reg=ot_args.reg_e, numItermax=ot_args.max_iter,
                            verbose=False, stopThr=1e-8)
    return float(distance)


def solve_ot(a, b, cost_matrix, ot_args) -> float:
    if ot_args.method in ("sinkhorn", "sinkhorn_log"):
        return sinkhorn_distance(a, b, cost_matrix, ot_args)
    
    elif ot_args.method == "emd":
        return float(ot.emd2(a, b, cost_matrix, verbose=True))
    
    raise ValueError(f"Unsupported method: {ot_args.method}")


def uniform_weights(n, device):
    return torch.ones(n, device=device) / n


def compute_ot_distance(src_embeddings, tgt_embeddings, ot_args) -> float:
    cost_matrix = compute_cost_matrix(src_embeddings, tgt_embeddings, ot_args.metric)
    cost_matrix = normalize_cost_matrix(cost_matrix, ot_args)

    a = uniform_weights(src_embeddings.shape[0], src_embeddings.device)
    b = uniform_weights(tgt_embeddings.shape[0], tgt_embeddings.device)

    return solve_ot(a, b, cost_matrix, ot_args)


def compute_combined_ot_distance(src_emb1, src_emb2, tgt_emb1, tgt_emb2, cost_args1, cost_args2, ot_args) -> float:
    cost_matrix_1 = compute_cost_matrix(src_emb1, tgt_emb1, cost_args1.metric)
    cost_matrix_1 = normalize_cost_matrix(cost_matrix_1, cost_args1)

    cost_matrix_2 = compute_cost_matrix(src_emb2, tgt_emb2, cost_args2.metric)
    cost_matrix_2 = normalize_cost_matrix(cost_matrix_2, cost_args2)

    combined = ot_args.lambda_param * cost_matrix_1 + (1 - ot_args.lambda_param) * cost_matrix_2

    if ot_args.normalize_after:
        combined = combined / torch.max(combined)

    a = uniform_weights(combined.shape[0], combined.device)
    b = uniform_weights(combined.shape[1], combined.device)

    distance = solve_ot(a, b, combined, ot_args)

    return distance


def compute_ot_coupling(src_embeddings, tgt_embeddings, ot_args):
    cost_matrix = compute_cost_matrix(src_embeddings, tgt_embeddings, ot_args.metric)
    cost_matrix = normalize_cost_matrix(cost_matrix, ot_args)

    a = uniform_weights(src_embeddings.shape[0], src_embeddings.device)
    b = uniform_weights(tgt_embeddings.shape[0], tgt_embeddings.device)
    
    if ot_args.method == "sinkhorn":
        return ot.sinkhorn(a, b, cost_matrix, reg=ot_args.reg_e, numItermax=ot_args.max_iter, verbose=True)
    
    elif ot_args.method == "emd":
        return ot.emd(a, b, cost_matrix, verbose=True)
    
    raise ValueError(f"Unsupported method: {ot_args.method}")

def combine_domain_embeddings(embeddings_dict, domain_indices):
    if isinstance(domain_indices, int):
        domain_indices = [domain_indices]
        
    return torch.cat([embeddings_dict[i] for i in domain_indices], dim=0)

def compute_ot_distance_with_unions(embeddings_dict, src_domains, tgt_domains, ot_args) -> float:
    src = combine_domain_embeddings(embeddings_dict, src_domains)
    tgt = combine_domain_embeddings(embeddings_dict, tgt_domains)

    return compute_ot_distance(src, tgt, ot_args)
