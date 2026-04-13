"""
Domain adaptation pretraining script with subset selection.
Supports both vision models (ResNet, DenseNet) on iNaturalist/FMOW 
and text models (BERT) on GeoYFCCText.
"""

import argparse
import functools
import sys
import os
import time
import logging

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add src to path (this script is in src/training, so src is two levels up)
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, src_dir)

from base_trainer import BasePretrainTrainer, load_config
from src.data.load_datasets import get_domain_split_mask, get_domain_dataloader, seed_worker
from src.core.subset_selection import choose_candidate_domains, uniformly_sample_across_domains
from src.core.utils import setup_logging, setup_directories, get_criterion, EarlyStopping, load_checkpoint_if_exists
import torch.optim as optim


class PretrainSubsetTrainer(BasePretrainTrainer):
    """Pretraining trainer with subset selection functionality."""
    
    def __init__(self, config, model_data_seed, 
                 subset_size=None, val_subset_size=None,
                 num_domains=None, domain_selection_method='random',
                 exclude_domains=None, ot_params=None, tgt_domain=None):
        """Initialize trainer with subset selection parameters."""
        self.subset_size = subset_size
        self.val_subset_size = val_subset_size
        self.num_domains = num_domains
        self.domain_selection_method = domain_selection_method
        self.candidate_domains = None
        self.ot_params = ot_params
        self.tgt_domain = tgt_domain
        self.model_data_seed = model_data_seed
        super().__init__(config, pretrain_domain_idx=None, model_seed=model_data_seed, exclude_domains=exclude_domains)
        # Store as model_data_seed for clarity (base class stores it as self.model_seed)
        
    def _setup_directories(self):
        """Setup checkpoint and log directories with subset information."""
        # Create subset suffix for directory names
        subset_suffix = ""
        if self.subset_size is not None:
            subset_suffix = f"_subset{self.subset_size}"
        if self.num_domains is not None:
            subset_suffix += f"_K{self.num_domains}_{self.domain_selection_method}"
        
        # Determine normalization type for subfolder
        # For OT method, use the normalization from ot_params; for others, use 'none'
        if self.domain_selection_method == 'ot' and self.ot_params is not None:
            norm_type = self.ot_params.get('norm', 'none')
        else:
            norm_type = 'none'
        
        self.checkpoint_dir = os.path.join(
            self.config["CHECKPOINT_ROOT"], 
            f"{self.task_directory_name}{subset_suffix}", 
            self.model_name,
            norm_type  # Add normalization type as subfolder
        )
        self.log_dir = os.path.join(
            self.config["LOG_ROOT"], 
            f"{self.task_directory_name}{subset_suffix}", 
            self.model_name
        )
        setup_directories([self.checkpoint_dir, self.log_dir])
    
    def _setup_logging(self):
        """Setup logging configuration with subset information."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Use _run_suffix() which already includes all subset information
        log_filename = f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{self._run_suffix()}_seed{self.model_data_seed}_{timestamp}.log"
        setup_logging(os.path.join(self.log_dir, log_filename))
        self.logger = logging.getLogger(__name__)
        
        print(f"Log stored at file logs/{log_filename}")
        self.logger.info(f"Log stored at file logs/{log_filename}")
        print(f"[CFG] subset_size(B)={self.subset_size}, num_domains(K)={self.num_domains}, domain_method={self.domain_selection_method}, val_subset_size(V)={self.val_subset_size}")
    
    def _setup_training_components(self):
        """Setup training components with subset-specific model path."""
        self.criterion = get_criterion(self.model_name)
        
        # Training hyperparameters
        model_config = self.config["MODELS"].get(self.model_name.upper(), {})
        lr = float(model_config.get("LEARNING_RATE", 5e-5))
        self.num_epochs = int(model_config.get("NUM_EPOCHS", 50))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Create subset suffix for model path
        subset_suffix = ""
        if self.subset_size is not None:
            subset_suffix = f"_subset{self.subset_size}"
        if self.num_domains is not None:
            subset_suffix += f"_K{self.num_domains}_{self.domain_selection_method}"
        
        model_path = os.path.join(
            self.checkpoint_dir,
            f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{self._run_suffix()}_seed{self.model_data_seed}_best.pth"
        )
        
        self.early_stopper = EarlyStopping(
            model_path=model_path,
            patience=self.config["PATIENCE"],
            mode='max',
            start_from_epoch=self.config["START_FROM_EPOCH"]
        )
        
        self.start_epoch = load_checkpoint_if_exists(model_path, self.model, self.optimizer)
        print(f"[CHECKPOINT] best_model_path={model_path}")

    def _run_suffix(self) -> str:
        parts = []
        if self.subset_size is not None:
            parts.append(f"_subset{self.subset_size}")
        if self.num_domains is not None:
            parts.append(f"_K{self.num_domains}")
            if self.domain_selection_method == 'ot':
                parts.append(f"_OT_{self.ot_params['embedding_type']}_{self.ot_params['method']}_{self.ot_params['reg']}_{self.ot_params['iter']}_{self.ot_params['metric']}_{self.ot_params['norm']}")
            else:
                parts.append(f"_{self.domain_selection_method}")
        if self.val_subset_size is not None:
            parts.append(f"_V{self.val_subset_size}")
        if self.tgt_domain is not None:
            parts.append(f"_tgt{self.tgt_domain}")
        return "".join(parts)
    
    def _is_training_completed(self) -> bool:
        """Check if training is already completed by looking for checkpoint and completed log."""
        # Check if checkpoint exists
        model_path = os.path.join(
            self.checkpoint_dir,
            f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{self._run_suffix()}_seed{self.model_data_seed}_best.pth"
        )
        
        if not os.path.exists(model_path):
            return False
        
        # Check if there's a log file with "Training completed in"
        log_pattern = f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{self._run_suffix()}_seed{self.model_data_seed}_*.log"
        import glob
        log_files = glob.glob(os.path.join(self.log_dir, log_pattern))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "Training completed in" in content:
                        print(f"Training already completed: {os.path.basename(log_file)}")
                        return True
            except Exception as e:
                print(f"Warning: Could not read log file {log_file}: {e}")
                continue
        
        return False
    
    def setup_dataloaders(self, dataset_name, dataset, train_batch_size, pretrain_domain_idx,
                         eval_batch_size, generator, model_data_seed, exclude_domains=None):
        """Setup train and validation dataloaders with multi-domain subset selection."""

        # 1) Choose candidate domains
        all_domains = self.config.get('PRETRAIN_DOMAINS', [])
        tokenizer_type = self.config["MODELS"].get(self.model_name.upper(), {}).get("TOKENIZER_TYPE", None)
        specific_domain = None
        if self.domain_selection_method == 'in_distribution':
            if self.tgt_domain is None:
                raise ValueError("tgt_domain must be provided when domain_selection_method='in_distribution'")
            specific_domain = self.tgt_domain

        self.candidate_domains = choose_candidate_domains(
            all_domains=all_domains,
            num_domains=self.num_domains,
            method=self.domain_selection_method,
            seed=model_data_seed,
            exclude_domains=exclude_domains,
            ot_params=self.ot_params,
            specific_domain=specific_domain,
        )
        print(f"[DOMAINS] K={len(self.candidate_domains)} -> {self.candidate_domains}")
        
        # 2) Build training mask with uniform sampling under budget across candidate domains
        if self.subset_size is None:
            raise ValueError("subset_size must be provided for multi-domain subset training.")
        train_mask = uniformly_sample_across_domains(
            dataset_name=dataset_name,
            dataset=dataset,
            candidate_domains=self.candidate_domains,
            split='train',
            budget=self.subset_size,
            seed=model_data_seed,
        )
        try:
            print(f"[MASK] Train selected={int(train_mask.sum())} / budget={self.subset_size}")
        except Exception:
            pass
        if self.val_subset_size is None:
            # Default: use full union if no val budget specified
            val_mask = None
            for d in self.candidate_domains:
                d_val = get_domain_split_mask(dataset_name, dataset, d, split='val')
                val_mask = d_val if val_mask is None else (val_mask | d_val)
        else:
            val_mask = uniformly_sample_across_domains(
                dataset_name=dataset_name,
                dataset=dataset,
                candidate_domains=self.candidate_domains,
                split='val',
                budget=self.val_subset_size,
                seed=model_data_seed,
            )
        try:
            print(f"[MASK] Val selected={int(val_mask.sum())}{' (union full)' if self.val_subset_size is None else f' / budget={self.val_subset_size}'}")
        except Exception:
            pass


        # Log chosen domains
        self.logger.info(f"Candidate domains (K={len(self.candidate_domains)}): {self.candidate_domains}")

        # Create worker init function with model_data_seed
        worker_init_fn = functools.partial(seed_worker, model_seed=model_data_seed)
        
        train_dataloader = get_domain_dataloader(
            dataset_name, dataset, tokenizer_type, train_mask, batch_size=train_batch_size, 
            shuffle=True, num_workers=16, pin_memory=True, 
            worker_init_fn=worker_init_fn, generator=generator
        )
        val_dataloader = get_domain_dataloader(
            dataset_name, dataset, tokenizer_type, val_mask, batch_size=eval_batch_size, 
            shuffle=False, num_workers=16, pin_memory=True, 
            worker_init_fn=worker_init_fn, generator=generator
        )
        return train_dataloader, val_dataloader


def _get_subset_defaults_from_config(config):
    """Read subset and OT defaults from config so CLI can omit them."""
    defaults = {}
    # Model/data seed
    if config.get("MODEL_DATA_SEEDS"):
        defaults["model_data_seed"] = config["MODEL_DATA_SEEDS"][0]
    elif config.get("MODEL_SEEDS"):
        defaults["model_data_seed"] = config["MODEL_SEEDS"][0]
    # Subset size / num_domains / method
    if config.get("BUDGET_VALUES"):
        defaults["subset_size"] = config["BUDGET_VALUES"][0]
    if config.get("K_VALUES"):
        defaults["num_domains"] = config["K_VALUES"][0]
    if config.get("DOMAIN_METHODS"):
        defaults["domain_selection_method"] = config["DOMAIN_METHODS"][0]
    if config.get("VAL_SIZE_MAPPING") and defaults.get("subset_size") is not None:
        defaults["val_subset_size"] = config["VAL_SIZE_MAPPING"].get(defaults["subset_size"])
    # OT: target domain and embedding type
    if config.get("TARGET_DOMAINS"):
        defaults["tgt_domain"] = config["TARGET_DOMAINS"][0]
    elif "TGT_DOMAIN" in config:
        defaults["tgt_domain"] = config["TGT_DOMAIN"]
    if config.get("OT_EMBEDDING_TYPE"):
        defaults["ot_embedding_type"] = config["OT_EMBEDDING_TYPE"]
    elif config.get("OT_EMBEDDING_TYPES"):
        defaults["ot_embedding_type"] = config["OT_EMBEDDING_TYPES"][0]
    # OT params (YAML uses OT_REG_E, OT_MAX_ITER, OT_NORMALIZE_COST)
    if "OT_METHOD" in config:
        defaults["ot_method"] = config["OT_METHOD"]
    if "OT_REG_E" in config:
        defaults["ot_reg"] = str(config["OT_REG_E"])
    if "OT_MAX_ITER" in config:
        defaults["ot_iter"] = str(config["OT_MAX_ITER"])
    if "OT_METRIC" in config:
        defaults["ot_metric"] = config["OT_METRIC"]
    if "OT_NORMALIZE_COST" in config:
        defaults["ot_norm"] = config["OT_NORMALIZE_COST"]
    return defaults


def main(config, model_data_seed, subset_size=None, val_subset_size=None, num_domains=None, domain_selection_method='random', exclude_domains=None,
         tgt_domain=None, ot_embedding_type=None, ot_method=None, ot_reg=None, ot_iter=None, ot_metric=None, ot_norm=None):
    """Main training function with subset selection."""
    # Fill from config when not provided
    cfg_defaults = _get_subset_defaults_from_config(config)
    if model_data_seed is None and cfg_defaults.get("model_data_seed") is not None:
        model_data_seed = cfg_defaults["model_data_seed"]
    if model_data_seed is None:
        raise ValueError("model_data_seed is required: provide -s/--model_data_seed or set MODEL_DATA_SEEDS/MODEL_SEEDS in config")
    if subset_size is None and cfg_defaults.get("subset_size") is not None:
        subset_size = cfg_defaults["subset_size"]
    if val_subset_size is None and cfg_defaults.get("val_subset_size") is not None:
        val_subset_size = cfg_defaults["val_subset_size"]
    if num_domains is None and cfg_defaults.get("num_domains") is not None:
        num_domains = cfg_defaults["num_domains"]
    if domain_selection_method == 'random' and cfg_defaults.get("domain_selection_method") is not None:
        domain_selection_method = cfg_defaults["domain_selection_method"]
    if domain_selection_method == 'ot':
        if tgt_domain is None and cfg_defaults.get("tgt_domain") is not None:
            tgt_domain = cfg_defaults["tgt_domain"]
        if ot_embedding_type is None and cfg_defaults.get("ot_embedding_type") is not None:
            ot_embedding_type = cfg_defaults["ot_embedding_type"]
        if ot_method is None and cfg_defaults.get("ot_method") is not None:
            ot_method = cfg_defaults["ot_method"]
        if ot_reg is None and cfg_defaults.get("ot_reg") is not None:
            ot_reg = cfg_defaults["ot_reg"]
        if ot_iter is None and cfg_defaults.get("ot_iter") is not None:
            ot_iter = cfg_defaults["ot_iter"]
        if ot_metric is None and cfg_defaults.get("ot_metric") is not None:
            ot_metric = cfg_defaults["ot_metric"]
        if ot_norm is None and cfg_defaults.get("ot_norm") is not None:
            ot_norm = cfg_defaults["ot_norm"]

    ot_params = None
    if domain_selection_method == 'ot':
        if tgt_domain is None or ot_embedding_type is None:
            raise ValueError(
                "For domain_selection_method='ot', provide --tgt_domain and --ot_embedding_type, "
                "or set TARGET_DOMAINS/TGT_DOMAIN and OT_EMBEDDING_TYPE/OT_EMBEDDING_TYPES in config."
            )
        ot_params = {
            'ot_distance_dir': config.get('OT_DISTANCE_DIR', None),
            'source_domain_idx': tgt_domain,
            'embedding_type': ot_embedding_type,
            'method': ot_method or 'sinkhorn',
            'reg': ot_reg or '0.01',
            'iter': ot_iter or '1000',
            'metric': ot_metric or 'cosine',
            'norm': ot_norm or 'max',
        }
    
    trainer = PretrainSubsetTrainer(
        config, model_data_seed, 
        subset_size=subset_size, val_subset_size=val_subset_size,
        num_domains=num_domains, domain_selection_method=domain_selection_method,
        exclude_domains=exclude_domains, ot_params=ot_params, tgt_domain=tgt_domain
    )
    
    # Check if training is already completed
    if trainer._is_training_completed():
        print("⏭️  Skipping training - already completed")
        return
    
    trainer.train()


if __name__ == '__main__':
    print("[INFO] Starting pretrain_by_domain_subset.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--model_data_seed', type=int, default=None,
                        help='Model/data seed (default: first of MODEL_DATA_SEEDS or MODEL_SEEDS in config)')
    parser.add_argument('--subset_size', type=int, default=None, 
                    help='Size B of subset to use from pretrain domain (None = use all data)')
    parser.add_argument('--val_subset_size', type=int, default=None, 
                    help='Size B of subset to use from val domain (None = use all data)')
    parser.add_argument('--num_domains', '-k', type=int, default=None,
                    help='K: number of candidate domains to choose (default: all)')
    parser.add_argument('--domain_selection_method', type=str, default='random',
                    choices=['random','ot','in_distribution','global'],
                    help='Method for choosing candidate domains')
    parser.add_argument('--exclude_domains', type=int, default=None,
                    help='Domains to exclude from candidate domains')
    # OT selection parameters
    parser.add_argument('--tgt_domain', type=int, default=None, help='Target domain for evaluation (e.g., OT source domain)')
    parser.add_argument('--ot_embedding_type', type=str, default=None, help='OT embedding type (e.g., geoclip)')
    parser.add_argument('--ot_method', type=str, default=None, help='OT method (e.g., sinkhorn)')
    parser.add_argument('--ot_reg', type=str, default=None, help='OT regularization (e.g., 0.01)')
    parser.add_argument('--ot_iter', type=str, default=None, help='OT iterations (e.g., 1000)')
    parser.add_argument('--ot_metric', type=str, default=None, help='OT metric (e.g., cosine)')
    parser.add_argument('--ot_norm', type=str, default=None, help='OT normalization (e.g., max)')
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    main(CONFIG, args.model_data_seed,
         subset_size=args.subset_size,
         val_subset_size=args.val_subset_size,
         num_domains=args.num_domains, domain_selection_method=args.domain_selection_method,
         exclude_domains=args.exclude_domains,
         tgt_domain=args.tgt_domain, ot_embedding_type=args.ot_embedding_type, ot_method=args.ot_method,
         ot_reg=args.ot_reg, ot_iter=args.ot_iter, ot_metric=args.ot_metric, ot_norm=args.ot_norm)