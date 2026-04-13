"""
Zero-shot evaluation script specifically for subset-trained models.
This script evaluates models that were trained using the subset selection experiments.
"""

import argparse
import csv
import logging
import time
import datetime
import os
import sys
import functools
from pathlib import Path
from tqdm import tqdm
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict

# Set deterministic training
cudnn.deterministic = True
cudnn.benchmark = False

# Add parent directory to path
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add src to path (this script is in src/evaluation, so src is two levels up)
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, src_dir)

# Local imports
from src.core.models import get_model
from src.data.load_datasets import (
    load_dataset, get_domain_split_mask, get_domain_dataloader, seed_worker
)
from src.training import get_train_eval_functions
from src.core.utils import (
    setup_logging, setup_directories, setup_seeds,
    format_metrics_vision, format_metrics_text
)

import yaml
from src.core.config_utils import load_config


def setup_test_dataloader(dataset_name, dataset, target_domain_idx, eval_batch_size, 
                         generator, model_data_seed):
    """Setup test dataloader for target domain."""
    
    test_mask = get_domain_split_mask(dataset_name, dataset, target_domain_idx, split='test')
    
    # Create worker init function with model_data_seed
    worker_init_fn = functools.partial(seed_worker, model_seed=model_data_seed)
    
    test_dataloader = get_domain_dataloader(
        dataset_name, dataset, test_mask, batch_size=eval_batch_size, 
        shuffle=False, num_workers=16, pin_memory=True, 
        worker_init_fn=worker_init_fn, generator=generator
    )
    return test_dataloader


def eval(model, test_dataloader, device, model_name):
    """Validate for one epoch."""
    _, eval_fn = get_train_eval_functions(model_name)
    
    if model_name == 'bert_multilabel' or model_name=='bert_singlelabel':
        test_preds, test_acc, test_top3_correct, test_top5_correct = eval_fn(model, test_dataloader, device)
        test_top3_acc = test_top3_correct / test_preds if test_preds > 0 else 0
        test_top5_acc = test_top5_correct / test_preds if test_preds > 0 else 0
        return test_preds, test_acc, test_top3_acc, test_top5_acc
    else:
        test_examples, test_correct, test_top3_correct, test_top5_correct = eval_fn(model, test_dataloader, device)
        test_acc = test_correct / test_examples if test_examples > 0 else 0
        test_top3_acc = test_top3_correct / test_examples if test_examples > 0 else 0
        test_top5_acc = test_top5_correct / test_examples if test_examples > 0 else 0
        return test_examples, test_acc, test_top3_acc, test_top5_acc


def load_pretrained_model(model, checkpoint_path, device):
    """Load pretrained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # If keys are prefixed with "module.", strip them
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "", 1)] = v
        state_dict = new_state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    return model


SUMMARY_COLUMNS = [
    "timestamp",
    "dataset_name",
    "tgt_domain_idx",
    "tgt_country",
    "select_by",
    "num_select",
    "budget",
    "train_model",
    "model_seed",
    "spatial_splits",
    "ot_embedding_type",
    "ot_method",
    "ot_reg",
    "ot_iter",
    "ot_metric",
    "ot_norm",
    "acc",
    "num_data",
    "num_correct",
    "num_top3_correct",
    "num_top5_correct",
    "top3_acc",
    "top5_acc",
]


def get_target_domain_name(dataset, dataset_name, domain_idx):
    """Best-effort lookup of human-readable target domain label."""
    try:
        if dataset_name == "geoyfcc_text" and hasattr(dataset, "df"):
            if "country_id" in dataset.df.columns and "country_name" in dataset.df.columns:
                matches = dataset.df.loc[dataset.df["country_id"] == domain_idx, "country_name"]
                if not matches.empty:
                    return matches.iloc[0]
        if hasattr(dataset, "domain_names"):
            domain_names = dataset.domain_names
            if isinstance(domain_names, dict) and domain_idx in domain_names:
                return domain_names[domain_idx]
        if hasattr(dataset, "metadata") and "region_name" in getattr(dataset, "metadata", {}):
            try:
                region_names = dataset.metadata["region_name"]
                if len(region_names) > domain_idx:
                    return str(region_names[domain_idx])
            except Exception:
                pass
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to get readable domain name for %s (idx=%s): %s",
            dataset_name,
            domain_idx,
            exc,
        )
    return f"Domain {domain_idx}"


def resolve_summary_csv_path(config, summary_csv_arg, results_dir, spatial_split_types, model_name):
    """Determine where zeroshot summary rows should be appended."""
    if summary_csv_arg:
        return summary_csv_arg
    config_path = config.get("ZEROSHOT_SUMMARY_CSV")
    if config_path:
        return config_path
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    filename = f"zeroshot_eval_{spatial_split_types}_{model_name}_summary.csv"
    return os.path.join(summary_dir, filename)


def append_summary_row(summary_csv_path, row):
    """Append a single summary row to CSV, writing header if needed."""
    if not summary_csv_path:
        return
    csv_path = Path(summary_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_summary_row(dataset_name, target_domain_idx, target_domain_label, subset_params,
                      model_name, spatial_split_types, model_seed, stats):
    """Create dictionary ready for CSV writing."""
    timestamp = datetime.datetime.utcnow().isoformat()
    select_by = subset_params.get('domain_selection_method') or 'unknown'
    row = {
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "tgt_domain_idx": target_domain_idx,
        "tgt_country": target_domain_label,
        "select_by": select_by,
        "num_select": subset_params.get('num_domains'),
        "budget": subset_params.get('subset_size'),
        "train_model": model_name,
        "model_seed": model_seed,
        "spatial_splits": spatial_split_types,
        "ot_embedding_type": subset_params.get('ot_embedding_type'),
        "ot_method": subset_params.get('ot_method'),
        "ot_reg": subset_params.get('ot_reg'),
        "ot_iter": subset_params.get('ot_iter'),
        "ot_metric": subset_params.get('ot_metric'),
        "ot_norm": subset_params.get('ot_norm'),
        "acc": stats.get("acc"),
        "num_data": stats.get("num_data"),
        "num_correct": stats.get("num_correct"),
        "num_top3_correct": stats.get("num_top3_correct"),
        "num_top5_correct": stats.get("num_top5_correct"),
        "top3_acc": stats.get("top3_acc"),
        "top5_acc": stats.get("top5_acc"),
    }
    return row


def save_results(results_all_seed, results_all_seed_with_preds, target_domain_idx, 
                task_name, spatial_split_types, model_name, subset_params,
                CSV_DIR, JSON_DIR, PKL_DIR, logger):
    """Save results to CSV, JSON, and pickle files with subset-specific naming."""
    
    # Create subset suffix for filenames
    subset_suffix = ""
    if subset_params.get('subset_size'):
        subset_suffix += f"_B{subset_params['subset_size']}"
    if subset_params.get('num_domains'):
        subset_suffix += f"_K{subset_params['num_domains']}_{subset_params['domain_selection_method']}"
    if subset_params.get('ot_embedding_type'):
        subset_suffix += f"_{subset_params['ot_embedding_type']}"
    
    # Save CSV
    csv_filename = f"{task_name}_{spatial_split_types}_{model_name}_subset{subset_suffix}_tgt{target_domain_idx}.csv"
    csv_filepath = os.path.join(CSV_DIR, csv_filename)
    
    # Create DataFrame without predictions for CSV
    results_for_csv = []
    for result in results_all_seed:
        csv_result = {k: v for k, v in result.items() if k not in ['test_labels', 'test_preds']}
        results_for_csv.append(csv_result)
    
    results_df = pd.DataFrame(results_for_csv)
    results_df.to_csv(csv_filepath, index=False)

    # Save JSON
    results_all_seed_dict = {result['model_data_seed']: {k: v for k, v in result.items() if k not in ['test_labels', 'test_preds']} for result in results_all_seed}
    json_filename = f"{task_name}_{spatial_split_types}_{model_name}_subset{subset_suffix}_tgt{target_domain_idx}.json"
    json_filepath = os.path.join(JSON_DIR, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(results_all_seed_dict, f)
    logger.info(f"Finish saving json to {json_filepath}")
    
    # Save pickle with predictions
    results_all_seed_with_preds_dict = {result['model_data_seed']: result for result in results_all_seed_with_preds}
    pkl_filename = f"{task_name}_{spatial_split_types}_{model_name}_subset{subset_suffix}_tgt{target_domain_idx}.pkl"
    pkl_filepath = os.path.join(PKL_DIR, pkl_filename)
    with open(pkl_filepath, 'wb') as file:
        pickle.dump(results_all_seed_with_preds_dict, file)
    logger.info(f"Finish saving pickle to {pkl_filepath}")


def main(config, target_domain_idxs, subset_params, summary_csv_override=None):
    """Main evaluation function for subset models."""
    
    task_name = "zeroshot_eval_subset"
    task_directory_name = "2_zeroshot_eval_subset"
    training_task_name = "pretrain"
    training_task_directory_name = "1_pretrain"

    # Setup device (single-GPU: cuda:0 if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name = config["DATASET_NAME"]
    spatial_split_types = config["DOMAIN_TYPE"]
    model_name = config["MODEL_NAME"]

    # Setup directories with subset-specific naming
    subset_suffix = ""
    if subset_params.get('subset_size'):
        subset_suffix += f"_subset{subset_params['subset_size']}"
    if subset_params.get('num_domains'):
        subset_suffix += f"_K{subset_params['num_domains']}_{subset_params['domain_selection_method']}"
    if subset_params.get('ot_norm'):
        norm_type = f"{subset_params['ot_norm']}"
    
    CHECKPOINT_DIR = os.path.join(config["CHECKPOINT_ROOT"], f"{training_task_directory_name}{subset_suffix}", model_name, norm_type)
    LOG_DIR = os.path.join(config["LOG_ROOT"], task_directory_name, model_name)
    
    RESULTS_DIR = os.path.join(config.get("RESULTS_ROOT", "1_training_results/test_results"), task_directory_name, model_name)
    CSV_DIR = os.path.join(RESULTS_DIR, "csv")
    JSON_DIR = os.path.join(RESULTS_DIR, "json")
    PKL_DIR = os.path.join(RESULTS_DIR, "pickle")
    
    setup_directories([LOG_DIR, RESULTS_DIR, CSV_DIR, JSON_DIR, PKL_DIR])

    summary_csv_path = resolve_summary_csv_path(
        config=config,
        summary_csv_arg=summary_csv_override,
        results_dir=RESULTS_DIR,
        spatial_split_types=spatial_split_types,
        model_name=model_name,
    )

    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"{task_name}_{spatial_split_types}_{model_name}_subset{subset_suffix}_{timestamp}.log"
    setup_logging(os.path.join(LOG_DIR, log_filename))
    logger = logging.getLogger(__name__)
    
    print(f"Log stored at file logs/{log_filename}")
    logger.info(f"Log stored at file logs/{log_filename}")

    # Load dataset
    print("Loading Dataset...")
    logger.info("Loading Dataset...")
    dataset = load_dataset(dataset_name, root_dir=config["DATA_DIR"])

    eval_batch_size = config["EVAL_BATCH_SIZE"]

    # If tgt_domain is specified in subset_params, only evaluate on that domain
    # Otherwise, evaluate on all target_domain_idxs
    if subset_params.get('tgt_domain') is not None:
        # Only evaluate on the specific target domain from checkpoint
        target_domain_idxs = [subset_params['tgt_domain']]
        print(f"[INFO] tgt_domain specified in checkpoint: {subset_params['tgt_domain']}")
        print(f"[INFO] Only evaluating on domain {subset_params['tgt_domain']}")
        logger.info(f"tgt_domain specified in checkpoint: {subset_params['tgt_domain']}, only evaluating on this domain")
    else:
        print(f"[INFO] No tgt_domain specified in checkpoint, evaluating on all domains: {target_domain_idxs}")
        logger.info(f"No tgt_domain specified in checkpoint, evaluating on all domains: {target_domain_idxs}")
    
    for target_domain_idx in target_domain_idxs:
        # Create subset-specific filename suffix
        subset_filename_suffix = ""
        if subset_params.get('subset_size'):
            subset_filename_suffix += f"_B{subset_params['subset_size']}"
        if subset_params.get('num_domains'):
            subset_filename_suffix += f"_K{subset_params['num_domains']}_{subset_params['domain_selection_method']}"
        if subset_params.get('ot_embedding_type'):
            subset_filename_suffix += f"_{subset_params['ot_embedding_type']}"
        
        csv_filename = f"{task_name}_{spatial_split_types}_{model_name}_subset{subset_filename_suffix}_tgt{target_domain_idx}.csv"
        csv_filepath = os.path.join(CSV_DIR, csv_filename)

        existing_seeds = set()
        if os.path.exists(csv_filepath):
            try:
                existing_df = pd.read_csv(csv_filepath)
                existing_seeds = set(existing_df['model_data_seed'].tolist())
                print(f"Found existing CSV with seeds: {existing_seeds}")
                logger.info(f"Found existing CSV with seeds: {existing_seeds}")
                if len(existing_seeds) == len(config["MODEL_DATA_SEEDS"]):
                    print(f"All seeds already exist in CSV")
                    logger.info(f"All seeds already exist in CSV")
                    continue
                
                # Load existing results into results_all_seed
                results_all_seed = existing_df.to_dict('records')
                
                # Try to load existing pickle file for results with predictions
                pkl_filename = f"{task_name}_{spatial_split_types}_{model_name}_subset{subset_filename_suffix}_tgt{target_domain_idx}.pkl"
                pkl_filepath = os.path.join(PKL_DIR, pkl_filename)
                if os.path.exists(pkl_filepath):
                    with open(pkl_filepath, 'rb') as file:
                        existing_pkl_data = pickle.load(file)
                        results_all_seed_with_preds = [existing_pkl_data[seed] for seed in existing_seeds if seed in existing_pkl_data]
                        
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
                logger.error(f"Error reading existing CSV: {e}")
                existing_seeds = set()
                results_all_seed = []
                results_all_seed_with_preds = []
        else:
            results_all_seed = []
            results_all_seed_with_preds = []

        print(f"Start Evaluation on domain {target_domain_idx}")
        logger.info(f"Start Evaluation on domain {target_domain_idx}")

        target_domain_label = get_target_domain_name(dataset, dataset_name, target_domain_idx)

        model_data_seeds = config["MODEL_DATA_SEEDS"]
        
        for model_data_seed in model_data_seeds:
            # Setup random seeds
            g = setup_seeds(model_data_seed)

            if model_data_seed in existing_seeds:
                print(f"Skipping model/data seed {model_data_seed} as it already exists in CSV")
                logger.info(f"Skipping model/data seed {model_data_seed} as it already exists in CSV")
                continue
            
            print(f"Loading model {model_name} with seed {model_data_seed}...")
            logger.info(f"Loading model {model_name} with seed {model_data_seed}...")
            
            model = get_model(model_name, num_classes=dataset.num_classes, device=device)
            model = model.to(device)

            # Load pretrained checkpoint with subset-specific naming
            checkpoint_parts = [training_task_name, spatial_split_types, model_name]
            
            if subset_params.get('subset_size'):
                checkpoint_parts.append(f"subset{subset_params['subset_size']}")
            if subset_params.get('num_domains'):
                checkpoint_parts.append(f"K{subset_params['num_domains']}")
                if subset_params.get('domain_selection_method') == 'ot':
                    checkpoint_parts.append(f"OT_{subset_params['ot_embedding_type']}_{subset_params.get('ot_method', 'sinkhorn')}_{subset_params.get('ot_reg', '0.01')}_{subset_params.get('ot_iter', '1000')}_{subset_params.get('ot_metric', 'cosine')}_{subset_params.get('ot_norm', 'max')}")
                else:
                    checkpoint_parts.append(subset_params['domain_selection_method'])
            if subset_params.get('val_subset_size'):
                checkpoint_parts.append(f"V{subset_params['val_subset_size']}")
            if subset_params.get('tgt_domain'):
                checkpoint_parts.append(f"tgt{subset_params['tgt_domain']}")


            checkpoint_filename = "_".join(checkpoint_parts) + f"_seed{model_data_seed}_best.pth"
            checkpoint_savepath = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
            
            if not os.path.exists(checkpoint_savepath):
                print(f"Checkpoint not found: {checkpoint_savepath}")
                logger.error(f"Checkpoint not found: {checkpoint_savepath}")
                continue
                
            print(f"Loading model {model_name} with seed {model_data_seed} from {checkpoint_savepath}...")
            model = load_pretrained_model(model, checkpoint_savepath, device)

            print("Loading Test Dataloader for target domain...")
            logger.info("Loading Test Dataloader for target domain...")
            
            test_dataloader = setup_test_dataloader(
                dataset_name, dataset, target_domain_idx, eval_batch_size, g, model_data_seed
            )

            test_preds, test_acc, test_top3_acc, test_top5_acc = eval(model, test_dataloader, device, model_name)

            result_with_preds = {
                'model_data_seed': model_data_seed,
                'test_acc': test_acc,
                'test_top3_acc': test_top3_acc,
                'test_top5_acc': test_top5_acc,
                'test_labels': getattr(test_dataloader.dataset, 'labels', None),  # optional
                'test_preds': test_preds  # optional; modify eval() if needed
            }

            result_without_preds = {k: v for k, v in result_with_preds.items() if k not in ['test_labels', 'test_preds']}

            results_all_seed.append(result_without_preds)
            results_all_seed_with_preds.append(result_with_preds)

            print(f"\tTest")
            print(f"\t\t test accuracy: {test_acc:.6f}")
            print(f"\t\t test top3 accuracy: {test_top3_acc:.6f}")
            print(f"\t\t test top5 accuracy: {test_top5_acc:.6f}")
            logger.info(f"\tTest")
            logger.info(f"\t\t test accuracy: {test_acc:.6f}")
            logger.info(f"\t\t test top3 accuracy: {test_top3_acc:.6f}")
            logger.info(f"\t\t test top5 accuracy: {test_top5_acc:.6f}")

            num_eval_examples = test_preds
            num_correct = int(round(test_acc * num_eval_examples)) if num_eval_examples else 0
            num_top3_correct = int(round(test_top3_acc * num_eval_examples)) if num_eval_examples else 0
            num_top5_correct = int(round(test_top5_acc * num_eval_examples)) if num_eval_examples else 0

            summary_stats = {
                "acc": test_acc,
                "num_data": num_eval_examples,
                "num_correct": num_correct,
                "num_top3_correct": num_top3_correct,
                "num_top5_correct": num_top5_correct,
                "top3_acc": test_top3_acc,
                "top5_acc": test_top5_acc,
            }
            summary_row = build_summary_row(
                dataset_name=dataset_name,
                target_domain_idx=target_domain_idx,
                target_domain_label=target_domain_label,
                subset_params=subset_params,
                model_name=model_name,
                spatial_split_types=spatial_split_types,
                model_seed=model_data_seed,
                stats=summary_stats,
            )
            append_summary_row(summary_csv_path, summary_row)

        # Save results
        if len(results_all_seed) > len(existing_seeds):
            save_results(
                results_all_seed,
                results_all_seed_with_preds,
                target_domain_idx,
                task_name,
                spatial_split_types,
                model_name,
                subset_params,
                CSV_DIR,
                JSON_DIR,
                PKL_DIR,
                logger
            )
        else:
            print(f"No new results to save for target domain {target_domain_idx}")
            logger.info(f"No new results to save for target domain {target_domain_idx}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_domains', type=int, nargs="+", required=True, help='Domains to test model')
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--summary_csv', type=str, help='Optional path to summary CSV file')
    
    # Subset model evaluation parameters
    parser.add_argument('--subset_size', type=int, help='Subset size B used for training')
    parser.add_argument('--num_domains', type=int, help='Number of domains K used for training')
    parser.add_argument('--domain_selection_method', type=str, choices=['random', 'ot', 'specific_domain', 'in_distribution', 'global'], help='Domain selection method used')
    parser.add_argument('--ot_embedding_type', type=str, help='OT embedding type (if OT method used)')
    parser.add_argument('--ot_method', type=str, help='OT method (if OT method used)')
    parser.add_argument('--ot_reg', type=str, help='OT regularization (if OT method used)')
    parser.add_argument('--ot_iter', type=str, help='OT iterations (if OT method used)')
    parser.add_argument('--ot_metric', type=str, help='OT metric (if OT method used)')
    parser.add_argument('--ot_norm', type=str, help='OT normalization (if OT method used)')
    parser.add_argument('--val_subset_size', type=int, help='Validation subset size V used for training')
    parser.add_argument('--tgt_domain', type=int, help='Target domain used for training')
    
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Create subset parameters dictionary
    subset_params = {
        'subset_size': args.subset_size,
        'num_domains': args.num_domains,
        'domain_selection_method': args.domain_selection_method,
        'ot_embedding_type': args.ot_embedding_type,
        'ot_method': args.ot_method,
        'ot_reg': args.ot_reg,
        'ot_iter': args.ot_iter,
        'ot_metric': args.ot_metric,
        'ot_norm': args.ot_norm,
        'val_subset_size': args.val_subset_size,
        'tgt_domain': args.tgt_domain
    }
    
    main(config, args.target_domains, subset_params, summary_csv_override=args.summary_csv)
