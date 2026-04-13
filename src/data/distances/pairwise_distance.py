#!/usr/bin/env python3
"""
Script to process embeddings and compute pairwise distances for multiple models.
"""

import sys
import os
import argparse
import yaml

sys.path.append(os.path.join(os.getcwd(), '../..'))

from compute_distances.core.utils import compute_pairwise_distances

def load_config(path="configs/experiments/pretrain_geoyfcc.yaml"):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from pretrained models")
    parser.add_argument('--config', type=str, default="configs/experiments/pretrain_geoyfcc.yaml", help="Path to config file")
    parser.add_argument('--model_type', type=str, default="bert", help="Model type")

    args = parser.parse_args()
    

    config = load_config(args.config)

    # Configuration
    data_dir = config["EMBEDDING_DIR"]
    result_dir = config["RESULT_DIR"]
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