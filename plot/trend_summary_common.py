"""Shared helpers for locating/summarizing the by_domain trend CSVs produced by
plot_trends_by_domain.py (one row per fixed domain, with a 'rho' column)."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

PLOTS_ROOT = Path("plot/plots")


def plots_root_for(dataset):
    """geoyfcc_text's trend CSVs live directly under plot/plots/<embedding>/by_domain/ (no
    dataset subfolder, since it's the only dataset processed so far); other datasets are
    expected under plot/plots/<dataset>/<embedding>/by_domain/ once they're computed."""
    return PLOTS_ROOT if dataset == "geoyfcc_text" else PLOTS_ROOT / dataset


def find_trend_csv(plots_root, embedding_type, distance_type, direction):
    pattern = Path(plots_root) / embedding_type / "by_domain" / \
        f"trend_{distance_type}_avg_test_acc_*_by_{direction}_domain.csv"
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple trend CSVs matched {pattern}: {matches}")
    return matches[0]


def load_rho_values(plots_root, embedding_type, distance_type, directions):
    """Concatenate the 'rho' column across one or more by_{direction}_domain trend CSVs."""
    values = []
    for direction in directions:
        csv_path = find_trend_csv(plots_root, embedding_type, distance_type, direction)
        if csv_path is None:
            print(f"[WARNING] No by_{direction}_domain trend CSV found for "
                  f"embedding={embedding_type}, distance_type={distance_type}, skipping")
            continue
        values.extend(pd.read_csv(csv_path)['rho'].dropna().tolist())
    return np.array(values)


def summarize_abs_rho(plots_root, embedding_type, distance_type, directions):
    """Mean +/- std of |rho| pooled across the given directions, or None if nothing found."""
    values = load_rho_values(plots_root, embedding_type, distance_type, directions)
    if len(values) == 0:
        return None
    abs_values = np.abs(values)
    return abs_values.mean(), abs_values.std()
