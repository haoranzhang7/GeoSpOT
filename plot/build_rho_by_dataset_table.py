#!/usr/bin/env python
"""
Build a LaTeX table of |Spearman's rho| between OT domain distance and downstream
transfer accuracy, one row per method (content embedding vs. GeoSpOT's location
embeddings) and one column per dataset.

Reads the per-domain trend CSVs produced by plot_trends_by_domain.py (one row per fixed
domain, with a 'rho' column) -- the same files summarize_by_domain_trends.py aggregates.
Each cell is the mean +/- std of |rho| across domains, computed only over the "fix source
domain, vary target" regressions (--direction src).

Only GeoYFCC-Text has these trend CSVs computed right now (under plot/plots/<embedding>/
by_domain/); the other three datasets are left blank until their trend CSVs exist under
plot/plots/<dataset>/<embedding>/by_domain/.

Example:
  python plot/build_rho_by_dataset_table.py
  python plot/build_rho_by_dataset_table.py --direction both
"""

import argparse

from latex_table_common import DATASET_LABELS, format_mean_std_cell, to_latex, write_and_print
from trend_summary_common import plots_root_for, summarize_abs_rho

DATASETS = ["geoyfcc_image", "geoyfcc_text", "fmow", "geode"]

# Each row is a method: a fixed embedding type, or (for the content-embedding baseline)
# one that depends on the dataset's modality -- BERT for the text dataset, ResNet50 for
# the image datasets.
CONTENT_EMBEDDING_BY_DATASET = {
    "geoyfcc_image": "resnet50",
    "geoyfcc_text": "bert",
    "fmow": "resnet50",
    "geode": "resnet50",
}
METHODS = [
    ("OT (Image/BERT)", CONTENT_EMBEDDING_BY_DATASET),
    ("GeoSpOT (GeoCLIP)", {d: "geoclip" for d in DATASETS}),
    ("GeoSpOT (SatCLIP)", {d: "satclip" for d in DATASETS}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", default="src", choices=["src", "tgt", "both"],
                         help="Which fixed-domain regressions to pool rho values from: 'src' (fix source "
                              "domain, vary target; default), 'tgt' (fix target, vary source), or 'both'.")
    parser.add_argument("--caption", default="Mean $\\pm$ std of $|\\rho|$ (Spearman correlation between "
                                              "OT domain distance and downstream transfer accuracy, across "
                                              "fixed-source-domain regressions) for content embeddings "
                                              "(BERT/ResNet50) vs. GeoSpOT's location embeddings.")
    parser.add_argument("--label", default="tab:rho_by_dataset")
    parser.add_argument("--out", default="plot/plots/tables/rho_by_dataset.tex")
    args = parser.parse_args()

    directions = ("src", "tgt") if args.direction == "both" else (args.direction,)

    header = ["Method"] + [DATASET_LABELS[d] for d in DATASETS]
    rows = []
    for method_label, embedding_by_dataset in METHODS:
        cells = [method_label]
        for dataset in DATASETS:
            value = summarize_abs_rho(plots_root_for(dataset), embedding_by_dataset[dataset], "ot", directions)
            cells.append(format_mean_std_cell(value))
        rows.append(cells)

    latex = to_latex(header, rows, args.caption, args.label)
    write_and_print(latex, args.out)


if __name__ == "__main__":
    main()
