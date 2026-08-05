import argparse
from pathlib import Path

import pandas as pd


def load_domain_names(metadata_file):
    """Map domain_idx (== country_id) to its country name from the raw metadata file."""
    names = pd.read_csv(metadata_file, usecols=['country_id', 'country']).drop_duplicates()
    return dict(zip(names['country_id'], names['country']))


def main(args):
    matrix = pd.read_csv(args.distance_file, index_col=0)
    matrix.index, matrix.columns = matrix.index.astype(int), matrix.columns.astype(int)
    domain_names = load_domain_names(args.metadata_file)

    pairs = matrix.stack(future_stack=True).rename('mmd_distance') \
        .rename_axis(['src_domain_idx', 'tgt_domain_idx']).reset_index()
    pairs = pairs[pairs['src_domain_idx'] != pairs['tgt_domain_idx']]
    pairs = pairs.dropna(subset=['mmd_distance'])

    pairs['src_domain_name'] = pairs['src_domain_idx'].map(domain_names)
    pairs['tgt_domain_name'] = pairs['tgt_domain_idx'].map(domain_names)
    pairs = pairs[['src_domain_idx', 'src_domain_name', 'tgt_domain_idx', 'tgt_domain_name', 'mmd_distance']]
    pairs = pairs.sort_values('mmd_distance', ascending=False).reset_index(drop=True)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, index=False)
    print(f"Saved {len(pairs)} domain pairs to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_file', type=str, default='data/geoyfcc_text/distances/mmd_distance/mmd_bert_kmultiscale.csv', help="Path to the square N x N MMD distance matrix CSV.")
    parser.add_argument('--metadata_file', type=str, default='data/geoyfcc_text/geoyfcc_all_metadata_before_cleaning.csv', help="Path to raw metadata CSV with 'country_id'/'country' columns, used to name domains.")
    parser.add_argument('--output_file', type=str, default='plot/plots/mmd_domain_distance_pairs.csv', help="Where to save the sorted domain-pair distance CSV.")
    args = parser.parse_args()

    main(args)
