#!/usr/bin/env python
"""
Build a CSV of subset-selection method performance (mean/std across seeds) for
plot_method_vs_baselines.py, from the per-run summary CSVs written by
zeroshot_test_eval_subset(_grid).py under
results/subset/test_results/2_zeroshot_eval_subset/<model>/summary/*.csv.

Each summary row is one (seed, budget, K, method) evaluation. This groups across
seeds and writes one row per (domain, budget, K, method, metric) with the mean
("value_est") and std ("std_est") of that metric, for every metric in --metrics.

Example:
  python plot/build_subset_selection_csv.py
  python plot/build_subset_selection_csv.py --model bert_singlelabel --metrics acc top3_acc top5_acc
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

METRIC_LABELS = {'acc': 'test_acc', 'top3_acc': 'test_top3_acc', 'top5_acc': 'test_top5_acc'}


def load_summary(results_root, model):
    pattern = str(Path(results_root) / '2_zeroshot_eval_subset' / model / 'summary' / '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No summary CSVs found matching {pattern}')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_root', default='results/subset/test_results')
    parser.add_argument('--model', default='bert_singlelabel')
    parser.add_argument('--metrics', nargs='+', default=['acc', 'top3_acc', 'top5_acc'])
    parser.add_argument('--out', default='plot/plots/subset_selection_results.csv')
    args = parser.parse_args()

    df = load_summary(args.results_root, args.model)
    df['domain'] = df['tgt_country'].str.replace('^Domain ', '', regex=True)
    df['method'] = df['select_by'].where(df['select_by'] != 'ot', df['ot_embedding_type'])

    rows = []
    group_cols = ['domain', 'budget', 'num_select', 'method']
    for (domain, budget, k, method), group in df.groupby(group_cols):
        for metric in args.metrics:
            rows.append({
                'domain': domain, 'budget': budget, 'K': k, 'method': method,
                'metric': METRIC_LABELS.get(metric, metric),
                'value_est': group[metric].mean(), 'std_est': group[metric].std(),
                'n_seeds': len(group),
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f'Saved {len(out_df)} rows to {out_path}')


if __name__ == '__main__':
    main()
