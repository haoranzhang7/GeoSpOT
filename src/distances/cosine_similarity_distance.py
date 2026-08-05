#!/usr/bin/env python
"""
Compute pairwise average cosine similarity over domain embeddings: for a domain pair (s, t),
avg_cos_sim(s, t) = mean over all (i in s, j in t) of cos_sim(x_i, y_j).

This is computed efficiently via the identity
  mean_{i,j} cos_sim(x_i, y_j) = mean_i(x_i / |x_i|) . mean_j(y_j / |y_j|)
i.e. the dot product of each domain's mean L2-normalized embedding, avoiding materializing the
full sample-by-sample similarity matrix.

Saves an N x N matrix CSV (rows=src, cols=tgt), matching the format produced by
mmd_distance.py / fid_distance.py.

Example:
  python src/distances/cosine_similarity_distance.py --dataset geoyfcc_text --embedding-type satclip
"""

import argparse, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distances.utils import DATA_ROOT, load_embeddings_and_domains, get_embs


def mean_normalized_vector(features):
    normed = features / features.norm(dim=1, keepdim=True)
    return normed.mean(dim=0)


def compute_avg_cosine_similarity_matrix(embeddings, domains, active, n, max_samples, device):
    matrix = np.full((n, n), np.nan)
    domain_dirs = {idx: mean_normalized_vector(get_embs(embeddings, domains, idx, max_samples, device)) for idx in active}

    for src in active:
        for tgt in active:
            if tgt == src:
                continue
            matrix[src, tgt] = torch.dot(domain_dirs[src], domain_dirs[tgt]).item()
    return matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geoyfcc_text", choices=["geoyfcc_text", "fmow", "geoyfcc_image"])
    p.add_argument("--total-domains", type=int, default=62)
    p.add_argument("--embedding-type", default="bert")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = args.total_domains
    embeddings, domains = load_embeddings_and_domains(args.dataset, args.embedding_type)
    active = [i for i in range(n) if (domains == i).sum() > 0]
    print(f"{len(active)} non-empty domains, {len(active) * (len(active) - 1)} pairs to compute")

    matrix = compute_avg_cosine_similarity_matrix(embeddings, domains, active, n, args.max_samples, device)

    out_dir = DATA_ROOT / args.dataset / "distances" / "cosine_distance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cosine_{args.embedding_type}_avg_similarity.csv"
    pd.DataFrame(matrix, index=range(n), columns=range(n)).to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
