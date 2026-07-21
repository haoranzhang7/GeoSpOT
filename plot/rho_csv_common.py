"""Shared helpers for plot/build_overall_rho_csv.py and plot/build_rho_by_src_domain_csv.py:
locate the precomputed domain-pair distance matrix for a given (dataset, distance_type,
embedding_type), tolerating the several file-naming variants left behind by
src/distances/{cosine,mmd,ot,fid,geodesic}_distance.py as those scripts evolve.

If a distance type has more than one matching file (e.g. mmd computed under multiple metrics,
or a matrix mid-recompute alongside an older finished one) the most recently modified match is
used; rows/pairs not yet filled in (NaN) are dropped downstream by trend_common.build_combined_df.

OT is the one exception: experiments/06_compute_ot_distances.sh's --method default has changed
over time (sinkhorn_log -> sinkhorn, see ot_distance.py), so at any given moment some embeddings'
matrices on disk are sinkhorn and others sinkhorn_log -- picking "newest" would silently compare
rho values computed under different OT solvers. Only sinkhorn_log results are used (see
find_ot_sinkhorn_log_file); if an embedding doesn't have one yet, it falls back to the legacy
old_results/*_max_per_domain.csv matrix and flags that in the yielded `note`.
"""

import os
import re
from pathlib import Path

DATA_ROOT = Path("data")

EMBEDDING_TYPES = ["bert", "geoclip", "satclip", "geodesic"]
DISTANCE_TYPES = ["cosine", "mmd", "ot", "fid", "geodesic"]

# Preference order when a distance type has been computed under multiple ground metrics.
MMD_METRIC_PRIORITY = ["euclidean", "geodesic", "cosine"]


def _newest(paths):
    return max(paths, key=os.path.getmtime) if paths else None


def find_distance_file(dataset, distance_type, embedding_type):
    """Return the Path to the newest matching distance matrix CSV, or None if not computed yet."""
    dist_dir = DATA_ROOT / dataset / "distances" / f"{distance_type}_distance"

    if distance_type == "geodesic":
        # Embedding-agnostic: only meaningful paired with embedding_type "geodesic" itself.
        if embedding_type != "geodesic":
            return None
        f = dist_dir / "geodesic_avg_distance.csv"
        return f if f.exists() else None

    if distance_type == "cosine":
        f = dist_dir / f"cosine_{embedding_type}_avg_similarity.csv"
        return f if f.exists() else None

    if distance_type == "mmd":
        for metric in MMD_METRIC_PRIORITY:
            matches = list(dist_dir.glob(f"mmd_{embedding_type}_kmultiscale_m{metric}_nmax_per_domain*.csv"))
            if matches:
                return _newest(matches)
        return None

    if distance_type == "fid":
        return _newest(list(dist_dir.glob(f"fid_{embedding_type}_m*_n*.csv")))

    if distance_type == "ot":
        raise ValueError("find_distance_file doesn't handle 'ot' -- OT matrices are split by "
                          "solver method, use find_ot_distance_files/iter_available_combos instead.")

    raise ValueError(f"Unknown distance_type: {distance_type}")


def find_ot_sinkhorn_log_file(dataset, embedding_type):
    """Return (Path, note) for the sinkhorn_log OT distance matrix for this embedding. If none has
    been computed yet, falls back to the legacy old_results/*_max_per_domain.csv matrix (from
    before ot_distance.py recorded --method in its output filename) and sets `note` to flag the
    fallback; returns (None, "") if neither exists."""
    dist_dir = DATA_ROOT / dataset / "distances" / "ot_distance"
    matches = list(dist_dir.glob(
        f"ot_distance_matrix_{embedding_type}_all_combinations_method_sinkhorn_log_*_norm_max_per_domain.csv"))
    if matches:
        return _newest(matches), ""

    # satclip's OT matrices were computed at two resolutions; "satclip" elsewhere in the
    # pipeline resolves to the L40 embeddings (see NPZ_EMBEDDING_FILES in src/distances/utils.py).
    fallback_emb = "satclip_L40" if embedding_type == "satclip" else embedding_type
    fallback = dist_dir / "old_results" / f"ot_distance_{fallback_emb}_max_per_domain.csv"
    if fallback.exists():
        return fallback, "sinkhorn_log result not available yet; fell back to old_results max_per_domain csv"

    return None, ""


def iter_available_combos(dataset, embedding_types, distance_types):
    """Yield (embedding_type, distance_type, distance_file, method, note) for every combo with a
    resolvable distance file; prints a warning and skips combos that aren't computed yet.
    method is only meaningful for distance_type "ot" (always "sinkhorn_log" -- see
    find_ot_sinkhorn_log_file); note flags when an OT row is a fallback to old_results."""
    for embedding_type in embedding_types:
        for distance_type in distance_types:
            if distance_type == "geodesic" and embedding_type != "geodesic":
                continue

            if distance_type == "ot":
                distance_file, note = find_ot_sinkhorn_log_file(dataset, embedding_type)
                if distance_file is None:
                    print(f"[WARNING] No sinkhorn_log ot distance file (or old_results fallback) "
                          f"found for embedding={embedding_type}, skipping")
                    continue
                if note:
                    print(f"[NOTE] {embedding_type}/ot: {note} ({distance_file})")
                yield embedding_type, distance_type, distance_file, "sinkhorn_log", note
                continue

            distance_file = find_distance_file(dataset, distance_type, embedding_type)
            if distance_file is None:
                print(f"[WARNING] No {distance_type} distance file found for "
                      f"embedding={embedding_type}, skipping")
                continue
            yield embedding_type, distance_type, distance_file, None, ""
