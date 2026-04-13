"""
Zero-shot evaluation script.
Supports both vision models (ResNet, DenseNet) on iNaturalist/FMOW
and text models (BERT) on GeoYFCCText.
"""

import argparse
import logging
import time
import datetime
import os
import sys
import functools
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
                         generator, model_seed):
    """Setup test dataloader for target domain."""
    
    test_mask = get_domain_split_mask(dataset_name, dataset, target_domain_idx, split='val')
    
    # Create worker init function with model_seed
    worker_init_fn = functools.partial(seed_worker, model_seed=model_seed)
    
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
        val_preds, val_acc, val_top3_correct, val_top5_correct = eval_fn(model, test_dataloader, device)
        val_top3_acc = val_top3_correct / val_preds if val_preds > 0 else 0
        val_top5_acc = val_top5_correct / val_preds if val_preds > 0 else 0
        return val_preds, val_acc, val_top3_acc, val_top5_acc
    else:
        val_examples, val_correct, val_top3_correct, val_top5_correct = eval_fn(model, test_dataloader, device)
        val_acc = val_correct / val_examples if val_examples > 0 else 0
        val_top3_acc = val_top3_correct / val_examples if val_examples > 0 else 0
        val_top5_acc = val_top5_correct / val_examples if val_examples > 0 else 0
        return val_examples, val_acc, val_top3_acc, val_top5_acc
 
def log_epoch_metrics(epoch, num_epochs, train_total_loss, train_avg_loss, train_acc, 
                     val_acc, model_name, logger):
    """Log metrics for the current epoch."""
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    logger.info(f"Epoch {epoch+1}/{num_epochs}")
    
    if model_name == 'bert_multilabel':
        train_msg, val_msg = format_metrics_text(train_total_loss, train_avg_loss, train_acc, val_acc)
    else:
        train_msg, val_msg = format_metrics_vision(train_total_loss, train_avg_loss, train_acc, val_acc)
    
    print(train_msg)
    print(val_msg)
    logger.info(train_msg)
    logger.info(val_msg)

def top_k_accuracy(outputs, labels, k=5):
    """Calculate top-k accuracy for vision models."""
    _, top_k_preds = torch.topk(outputs, k=k, dim=1)
    top_k_correct = torch.sum(top_k_preds.t() == labels.view(1, -1)).item()
    return top_k_correct

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



def save_results(results_all_seed, results_all_seed_with_preds, pretrain_domain_idx, 
                target_domain_idx, task_name, spatial_split_types, model_name, 
                CSV_DIR, JSON_DIR, PKL_DIR, logger):
    """Save results to CSV, JSON, and pickle files."""
    # Save CSV
    csv_filename = f"{task_name}_{spatial_split_types}_{model_name}_domains_src{pretrain_domain_idx}_tgt{target_domain_idx}.csv"
    csv_filepath = os.path.join(CSV_DIR, csv_filename)
    
    # Create DataFrame without predictions for CSV
    results_for_csv = []
    for result in results_all_seed:
        csv_result = {k: v for k, v in result.items() if k not in ['test_labels', 'test_preds']}
        results_for_csv.append(csv_result)
    
    results_df = pd.DataFrame(results_for_csv)
    results_df.to_csv(csv_filepath, index=False)

    # Save JSON
    results_all_seed_dict = {result['model_seed']: {k: v for k, v in result.items() if k not in ['test_labels', 'test_preds']} for result in results_all_seed}
    json_filename = f"{task_name}_{spatial_split_types}_{model_name}_domains_src{pretrain_domain_idx}_tgt{target_domain_idx}.json"
    json_filepath = os.path.join(JSON_DIR, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(results_all_seed_dict, f)
    logger.info(f"Finish saving json to {json_filepath}")
    
    # Save pickle with predictions
    results_all_seed_with_preds_dict = {result['model_seed']: result for result in results_all_seed_with_preds}
    pkl_filename = f"{task_name}_{spatial_split_types}_{model_name}_domains_src{pretrain_domain_idx}_tgt{target_domain_idx}.pkl"
    pkl_filepath = os.path.join(PKL_DIR, pkl_filename)
    with open(pkl_filepath, 'wb') as file:
        pickle.dump(results_all_seed_with_preds_dict, file)
    logger.info(f"Finish saving pickle to {pkl_filepath}")


def main(config, pretrain_domain_idx, target_domain_idxs, subset_model=False, subset_size=None, num_domains=None, 
         domain_selection_method=None, ot_embedding_type=None, ot_method=None, ot_reg=None, ot_iter=None, 
         ot_metric=None, ot_norm=None, val_subset_size=None, tgt_domain=None):
    """Main evaluation function."""
    
    task_name = "zeroshot_eval"
    task_directory_name = "2_zeroshot_eval"
    training_task_name = "pretrain"
    training_task_directory_name = "1_pretrain"

    # Setup device (single-GPU: cuda:0 if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name = config["DATASET_NAME"]
    spatial_split_types = config["DOMAIN_TYPE"]
    model_name = config["MODEL_NAME"]

    # Setup directories
    if subset_model:
        # For subset models, use the subset-specific directory structure
        subset_suffix = ""
        if subset_size is not None:
            subset_suffix = f"_subset{subset_size}"
        if num_domains is not None:
            subset_suffix += f"_K{num_domains}_{domain_selection_method}"
        
        CHECKPOINT_DIR = os.path.join(config["CHECKPOINT_ROOT"], f"{training_task_directory_name}{subset_suffix}", model_name)
    else:
        # For regular models, use the standard directory structure
        CHECKPOINT_DIR = os.path.join(config["CHECKPOINT_ROOT"], training_task_directory_name, model_name)
    LOG_DIR = os.path.join(config["LOG_ROOT"], task_directory_name, model_name)
    
    RESULTS_DIR = os.path.join(config.get("RESULTS_ROOT", "1_training_results/test_results"), task_directory_name, model_name)
    CSV_DIR = os.path.join(RESULTS_DIR, "csv")
    JSON_DIR = os.path.join(RESULTS_DIR, "json")
    PKL_DIR = os.path.join(RESULTS_DIR, "pickle")
    
    setup_directories([LOG_DIR, RESULTS_DIR, CSV_DIR, JSON_DIR, PKL_DIR])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"{task_name}_{spatial_split_types}_{model_name}_domain{pretrain_domain_idx}_{timestamp}.log"
    setup_logging(os.path.join(LOG_DIR, log_filename))
    logger = logging.getLogger(__name__)
    
    print(f"Log stored at file logs/{log_filename}")
    logger.info(f"Log stored at file logs/{log_filename}")

    print("Loading Dataset...")
    logger.info("Loading Dataset...")
    dataset = load_dataset(dataset_name, root_dir=config["DATA_DIR"])

    eval_batch_size = config["EVAL_BATCH_SIZE"]

    for target_domain_idx in target_domain_idxs:
        print('Target domain indices:', target_domain_idxs)
        csv_filename = f"{task_name}_{spatial_split_types}_{model_name}_domains_src{pretrain_domain_idx}_tgt{target_domain_idx}.csv"
        csv_filepath = os.path.join(CSV_DIR, csv_filename)

        existing_seeds = set()
        if os.path.exists(csv_filepath):
            try:
                print(csv_filepath)
                existing_df = pd.read_csv(csv_filepath)
                existing_seeds = set(existing_df['model_seed'].tolist())
                print(f"Found existing CSV with seeds: {existing_seeds}")
                logger.info(f"Found existing CSV with seeds: {existing_seeds}")
                if len(existing_seeds) == len(config["MODEL_SEEDS"]):
                    print(f"All seeds already exist in CSV")
                    logger.info(f"All seeds already exist in CSV")
                    continue
                
                # Load existing results into results_all_seed
                results_all_seed = existing_df.to_dict('records')
                
                # Try to load existing pickle file for results with predictions
                pkl_filename = f"{task_name}_{spatial_split_types}_{model_name}_domains_src{pretrain_domain_idx}_tgt{target_domain_idx}.pkl"
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

        model_seeds = config["MODEL_SEEDS"]
        
        for model_seed in model_seeds:
            # Setup random seeds
            g = setup_seeds(model_seed)

            if model_seed in existing_seeds:
                print(f"Skipping model seed {model_seed} as it already exists in CSV")
                logger.info(f"Skipping model seed {model_seed} as it already exists in CSV")
                continue
            
            print(f"Loading model {model_name} with seed {model_seed}...")
            logger.info(f"Loading model {model_name} with seed {model_seed}...")
            
            model = get_model(model_name, num_classes=dataset.num_classes, device=device)
            model = model.to(device)

            # Load pretrained checkpoint
            if subset_model:
                # Build subset-specific checkpoint filename
                checkpoint_parts = [training_task_name, spatial_split_types, model_name]
                
                if subset_size is not None:
                    checkpoint_parts.append(f"subset{subset_size}")
                if num_domains is not None:
                    checkpoint_parts.append(f"K{num_domains}")
                    if domain_selection_method == 'ot':
                        checkpoint_parts.append(f"OT_{ot_embedding_type}_{ot_method}_{ot_reg}_{ot_iter}_{ot_metric}_{ot_norm}")
                    else:
                        checkpoint_parts.append(domain_selection_method)
                if val_subset_size is not None:
                    checkpoint_parts.append(f"V{val_subset_size}")
                if tgt_domain is not None:
                    checkpoint_parts.append(f"tgt{tgt_domain}")
                
                checkpoint_filename = "_".join(checkpoint_parts) + f"_seed{model_seed}_best.pth"
            else:
                # Standard checkpoint filename
                checkpoint_filename = f"{training_task_name}_{spatial_split_types}_{model_name}_domain{pretrain_domain_idx}_seed{model_seed}_best.pth"
            
            checkpoint_savepath = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
            
            if not os.path.exists(checkpoint_savepath):
                print(f"Checkpoint not found: {checkpoint_savepath}")
                logger.error(f"Checkpoint not found: {checkpoint_savepath}")
                continue
                
            print(f"Loading model {model_name} with seed {model_seed} from {checkpoint_savepath}...")
            model = load_pretrained_model(model, checkpoint_savepath, device)

            print("Loading Test Dataloader for target domain...")
            logger.info("Loading Test Dataloader for target domain...")
            
            test_dataloader = setup_test_dataloader(
                dataset_name, dataset, target_domain_idx, eval_batch_size, g, model_seed
            )

            test_preds, test_acc, test_top3_acc, test_top5_acc = eval(model, test_dataloader, device, model_name)

            result_with_preds = {
                'model_seed': model_seed,
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

        # Save results
        if len(results_all_seed) > len(existing_seeds):
            save_results(
                results_all_seed,
                results_all_seed_with_preds,
                pretrain_domain_idx,
                target_domain_idx,
                task_name,
                spatial_split_types,
                model_name,
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
    parser.add_argument('-d', '--pretrain_domain', type=int, required=True, help='Domain to pretrain model')
    parser.add_argument('-t', '--target_domains', type=int, nargs="+", required=True, help='Domain to test model')
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    # Subset model evaluation parameters
    parser.add_argument('--subset_model', action='store_true', help='Use subset-trained model')
    parser.add_argument('--subset_size', type=int, help='Subset size B used for training')
    parser.add_argument('--num_domains', type=int, help='Number of domains K used for training')
    parser.add_argument('--domain_selection_method', type=str, choices=['random', 'ot'], help='Domain selection method used')
    parser.add_argument('--ot_embedding_type', type=str, help='OT embedding type (if OT method used)')
    parser.add_argument('--ot_method', type=str, help='OT method (if OT method used)')
    parser.add_argument('--ot_reg', type=str, help='OT regularization (if OT method used)')
    parser.add_argument('--ot_iter', type=str, help='OT iterations (if OT method used)')
    parser.add_argument('--ot_metric', type=str, help='OT metric (if OT method used)')
    parser.add_argument('--ot_norm', type=str, help='OT normalization (if OT method used)')
    parser.add_argument('--val_subset_size', type=int, help='Validation subset size V used for training')
    parser.add_argument('--tgt_domain', type=int, help='Target domain used for training')
    args = parser.parse_args()

    # Handle relative config paths
    config_path = args.config
    if not os.path.isabs(config_path):
        # If not absolute, assume it's relative to project root (one level up from src_dir)
        project_root = os.path.dirname(src_dir)
        config_path = os.path.abspath(os.path.join(project_root, 'SatOT', config_path))
    config = load_config(config_path)
    
    main(config, args.pretrain_domain, args.target_domains,
         subset_model=args.subset_model, subset_size=args.subset_size, num_domains=args.num_domains,
         domain_selection_method=args.domain_selection_method, ot_embedding_type=args.ot_embedding_type,
         ot_method=args.ot_method, ot_reg=args.ot_reg, ot_iter=args.ot_iter, ot_metric=args.ot_metric,
         ot_norm=args.ot_norm, val_subset_size=args.val_subset_size, tgt_domain=args.tgt_domain)