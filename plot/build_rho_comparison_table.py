#!/usr/bin/env python
"""
Build a LaTeX table comparing |Spearman's rho| between domain distance (OT / MMD / FID)
and downstream transfer accuracy, for text embeddings (BERT) vs. location embeddings
(GeoCLIP, SatCLIP).

Reads the per-domain trend CSVs produced by plot_trends_by_domain.py (one row per fixed
src or tgt domain, with a 'rho' column), the same files summarize_by_domain_trends.py
aggregates into by_domain_trend_summary.csv. Each table cell is the mean +/- std of
|rho| across domains. By default rho values from both the "fix src domain" and "fix tgt
domain" regressions are pooled together (--direction src/tgt/both).

Only GeoYFCC-Text has these trend CSVs computed right now (under plot/plots/<embedding>/
by_domain/); pass --datasets to add GeoYFCC-Image / FMoW / GeoDE once their trend CSVs
exist under plot/plots/<dataset>/<embedding>/by_domain/.

Example:
  python plot/build_rho_comparison_table.py
  python plot/build_rho_comparison_table.py --direction tgt
  python plot/build_rho_comparison_table.py --datasets geoyfcc_text fmow
"""

import argparse

from latex_table_common import DATASET_LABELS, EMBEDDING_LABELS, DISTANCE_LABELS, \
    build_dataset_embedding_table, write_and_print
from trend_summary_common import plots_root_for, summarize_abs_rho


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["geoyfcc_text"], choices=list(DATASET_LABELS))
    parser.add_argument("--embedding-types", nargs="+", default=["bert", "geoclip", "satclip"],
                         choices=list(EMBEDDING_LABELS))
    parser.add_argument("--distance-types", nargs="+", default=["ot", "mmd", "fid"], choices=list(DISTANCE_LABELS))
    parser.add_argument("--direction", default="both", choices=["src", "tgt", "both"],
                         help="Which fixed-domain regressions to pool rho values from: 'src' (fix source "
                              "domain, vary target), 'tgt' (fix target, vary source), or 'both' (pool both "
                              "sets of per-domain rho values together).")
    parser.add_argument("--caption", default="Mean $\\pm$ std of $|\\rho|$ (Spearman correlation between "
                                              "domain distance and downstream transfer accuracy, across "
                                              "fixed-domain regressions) for text (BERT) vs.\\ location "
                                              "(GeoCLIP, SatCLIP) embeddings.")
    parser.add_argument("--label", default="tab:rho_comparison")
    parser.add_argument("--out", default="plot/plots/tables/rho_comparison.tex")
    args = parser.parse_args()

    directions = ("src", "tgt") if args.direction == "both" else (args.direction,)

    def value_fn(dataset, embedding_type, distance_type):
        return summarize_abs_rho(plots_root_for(dataset), embedding_type, distance_type, directions)

    latex = build_dataset_embedding_table(
        args.datasets, args.embedding_types, args.distance_types, DISTANCE_LABELS,
        value_fn, args.caption, args.label)
    write_and_print(latex, args.out)


if __name__ == "__main__":
    main()
