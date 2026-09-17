#!/usr/bin/env python
"""Checks that every OT distance file 07_subset_selection_slurm.sh's array needs is already cached,
so a missing one doesn't trigger a slow mid-training recompute or crash. Keep TARGETS/K_VALUES/
EMBS/LOC_EMBS/LAMBDA in sync with that file. Usage: python experiments/check_subset_selection_data.py
"""
import sys
from pathlib import Path
import pandas as pd

OT_DIR = Path("data/geoyfcc_text/distances/ot_distance")
TARGETS, K_VALUES, LAMBDA, TOTAL_DOMAINS = [57, 12], [1, 2, 5], 0.5, 62
COMBOS = [(e, None) for e in ["bert", "geoclip", "satclip", "geodesic"]] + \
         [(f"{e}+bert", LAMBDA) for e in ["geoclip", "satclip", "geodesic"]]


def missing(embedding_type, lam, k, tgt):
    metric = "geodesic" if embedding_type == "geodesic" else "cosine"
    suffix = f"method_sinkhorn_log_reg_0.01_iter_1000_metric_{metric}_norm_max_per_domain"
    suffix += f"_lambda_{lam}" if lam is not None else ""
    if k == 1:
        path = OT_DIR / f"ot_distance_matrix_{embedding_type}_all_combinations_{suffix}.csv"
        if not path.exists():
            return f"MISSING FILE: {path}"
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str)
        row_ok = str(tgt) in df.index and df.loc[str(tgt)].reindex([str(i) for i in range(TOTAL_DOMAINS)]).notna().all()
        return None if row_ok else f"MISSING ROW src={tgt} in {path.name}"
    path = OT_DIR / f"distances_k{k}_{embedding_type}_greedy_{suffix}.csv"
    if not path.exists():
        return f"MISSING FILE: {path}"
    df = pd.read_csv(path)
    ok = ((df["src_domain_idx"].astype(str) == str(tgt)) & (df["k"] == k) & df["greedy_sequential"]).any()
    return None if ok else f"MISSING ROW src={tgt} k={k} in {path.name}"


problems = [m for tgt in TARGETS for k in K_VALUES for e, lam in COMBOS if (m := missing(e, lam, k, tgt))]
if problems:
    print(f"{len(problems)} gap(s) -- these jobs will recompute mid-training (file missing) or crash (row missing):")
    print("\n".join(f" - {p}" for p in problems))
    sys.exit(1)
print("All OT distance data required by 07_subset_selection_slurm.sh is present.")
