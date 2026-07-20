#!/usr/bin/env python
"""
Build a LaTeX table comparing domain-pairwise distances (OT / MMD / FID) between
text embeddings (BERT) and location embeddings (GeoCLIP, SatCLIP).

Each distance type's N x N domain-distance matrix (rows/cols = domain indices,
produced by src/distances/{ot_distance,mmd_distance,fid_distance}.py or
build_ot_distance_matrix.py) is summarized as the mean +/- std over its
off-diagonal (cross-domain) entries.

Only GeoYFCC-Text has precomputed matrices for all three distance types right now;
pass --datasets to add GeoYFCC-Image / FMoW / GeoDE once their matrices exist under
data/<dataset>/distances/{ot_distance,mmd_distance,fid_distance}/.

Example:
  python plot/build_distance_comparison_table.py
  python plot/build_distance_comparison_table.py --datasets geoyfcc_text fmow
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from latex_table_common import DATASET_LABELS, EMBEDDING_LABELS, DISTANCE_LABELS, \
    build_dataset_embedding_table, write_and_print

DATA_ROOT = Path("data")

# Filename template for each distance type's precomputed N x N matrix, keyed by embedding
# type. These match the files currently produced by experiments/06,09,10_compute_*.sh.
DISTANCE_FILE_TEMPLATES = {
    "ot": ("ot_distance", "ot_distance_{emb}_max_per_domain.csv"),
    "mmd": ("mmd_distance", "mmd_{emb}_kmultiscale_meuclidean_nmax_per_domain_and_normalized_after.csv"),
    "fid": ("fid_distance", "fid_{emb}_meuclidean_nnone.csv"),
}

# The OT pipeline computed satclip at two resolutions (L10/L40); "satclip" everywhere else
# in the pipeline resolves to the L40 embeddings (see NPZ_EMBEDDING_FILES in src/distances/utils.py).
EMBEDDING_FILE_OVERRIDES = {"ot": {"satclip": "satclip_L40"}}


def summarize_matrix(dataset, embedding_type, distance_type):
    """Mean +/- std over the off-diagonal (cross-domain) entries of an N x N distance matrix."""
    subdir, template = DISTANCE_FILE_TEMPLATES[distance_type]
    emb = EMBEDDING_FILE_OVERRIDES.get(distance_type, {}).get(embedding_type, embedding_type)
    path = DATA_ROOT / dataset / "distances" / subdir / template.format(emb=emb)
    if not path.exists():
        print(f"[WARNING] Missing {path}, leaving {distance_type}/{embedding_type}/{dataset} blank")
        return None

    matrix = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
    np.fill_diagonal(matrix, np.nan)
    values = matrix[~np.isnan(matrix)]
    return values.mean(), values.std()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["geoyfcc_text"], choices=list(DATASET_LABELS))
    parser.add_argument("--embedding-types", nargs="+", default=["bert", "geoclip", "satclip"],
                         choices=list(EMBEDDING_LABELS))
    parser.add_argument("--distance-types", nargs="+", default=["ot", "mmd", "fid"], choices=list(DISTANCE_LABELS))
    parser.add_argument("--caption", default="Mean $\\pm$ std cross-domain distance under each metric, for "
                                              "text (BERT) vs.\\ location (GeoCLIP, SatCLIP) embeddings.")
    parser.add_argument("--label", default="tab:distance_comparison")
    parser.add_argument("--out", default="plot/plots/tables/distance_comparison.tex")
    args = parser.parse_args()

    latex = build_dataset_embedding_table(
        args.datasets, args.embedding_types, args.distance_types, DISTANCE_LABELS,
        summarize_matrix, args.caption, args.label)
    write_and_print(latex, args.out)


if __name__ == "__main__":
    main()
