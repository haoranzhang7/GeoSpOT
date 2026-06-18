#!/usr/bin/env python3
"""
Script to process embeddings and compute pairwise distances for multiple models.
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.getcwd(), '../..'))

from compute_distances.core.utils import compute_pairwise_distances


def main():
    parser = argparse.ArgumentParser(description="Compute pairwise distances from embeddings")
    parser.add_argument('--embedding_dir', type=str, default='./data/embeddings', help="Directory containing embeddings")
    parser.add_argument('--result_dir', type=str, default='./data/distances', help="Directory to save distance results")
    parser.add_argument('--model_type', type=str, default="bert", help="Model type")

    args = parser.parse_args()

    data_dir = args.embedding_dir
    result_dir = args.result_dir
    dataset_name = 'geoyfcc_text'
    devices = ['cuda:0']  # Adjust based on available GPUs

    model_type = args.model_type

    if model_type == "bert":
        full_model = "bert_singlelabel"
    else:
        full_model = model_type
    embedding_path = f'{data_dir}/{dataset_name}/{model_type}/{dataset_name}_train_{full_model}.npz'

    if os.path.exists(embedding_path):
        print(f"Processing {model_type}...")
        compute_pairwise_distances(
            embedding_file=embedding_path,
            model_type=model_type,
            dataset_name=dataset_name,
            result_dir=result_dir,
            devices=devices
        )
        print(f"Completed {model_type}")
    else:
        print(f"File {embedding_path} not found!")

if __name__ == "__main__":
    main()