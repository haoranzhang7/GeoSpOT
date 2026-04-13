"""
Few-shot transfer training script.
Loads a pretrained checkpoint and fine-tunes on a target domain with few-shot samples.
"""

import argparse
import logging
import time
import os
import sys
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Set deterministic training
cudnn.deterministic = True
cudnn.benchmark = False

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# Local imports
from src.training.base_trainer import BasePretrainTrainer
from src.core.config_utils import load_config
from src.data.load_datasets import (
    get_domain_split_mask, get_domain_dataloader, seed_worker, get_finetune_train_mask
)
from src.core.utils import (
    EarlyStopping, setup_logging, setup_directories, load_checkpoint_if_exists, get_criterion
)

import torch.nn as nn


def find_last_linear(module: nn.Module) -> nn.Linear | None:
    """
    Find the last nn.Linear layer in the model without recursion.
    """
    last_linear = None
    for m in module.modules():  # includes the top-level module and all submodules
        if isinstance(m, nn.Linear):
            last_linear = m
    return last_linear

class FewshotTransferTrainer(BasePretrainTrainer):
    """Few-shot transfer trainer that loads a pretrained checkpoint and fine-tunes on a target domain."""
    
    def __init__(self, config, pretrain_domain_idx, target_domain_idx, finetune_size, 
                 model_seed, finetune_data_seed, finetune_mode='last', budget=None, train_val_ratio=0.5):
        """Initialize the few-shot transfer trainer."""
        self.target_domain_idx = target_domain_idx
        self.finetune_size = int(finetune_size)
        self.finetune_data_seed = finetune_data_seed
        self.finetune_mode = finetune_mode
        self.budget = budget
        self.train_val_ratio = float(train_val_ratio)  # val_size = train_size * ratio
        super().__init__(config, pretrain_domain_idx=pretrain_domain_idx, model_seed=model_seed)
    
    def _run_suffix(self) -> str:
        """Generate run suffix for checkpoint and log naming."""
        return f"_ft{self.finetune_size}shot_domains_src{self.pretrain_domain_idx}_tgt{self.target_domain_idx}_mseed{self.model_seed}_dseed{self.finetune_data_seed}"
    
    def _setup_directories(self):
        """Setup checkpoint and log directories."""
        self.checkpoint_dir = os.path.join(
            self.config["CHECKPOINT_ROOT"], 
            self.task_directory_name, 
            self.model_name, 
            f"ft{self.finetune_size}shot"
        )
        self.log_dir = os.path.join(
            self.config["LOG_ROOT"], 
            self.task_directory_name, 
            self.model_name
        )
        print(f"[DIR] checkpoint_dir={self.checkpoint_dir}")
        print(f"[DIR] log_dir={self.log_dir}")
        setup_directories([self.checkpoint_dir, self.log_dir])
    
    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{self._run_suffix()}_{timestamp}.log"
        setup_logging(os.path.join(self.log_dir, log_filename))
        self.logger = logging.getLogger(__name__)
        
        print(f"Log stored at file logs/{log_filename}")
        self.logger.info(f"Log stored at file logs/{log_filename}")
    
    def _setup_model(self):
        """Setup the model and load pretrained checkpoint."""
        # Build base model using parent class
        super()._setup_model()
        
        # Load pretrain checkpoint
        pretrain_dir = os.path.join(self.config["CHECKPOINT_ROOT"], "1_pretrain", self.model_name)
        ckpt_name = f"pretrain_{self.spatial_split_types}_{self.model_name}_domain{self.pretrain_domain_idx}_seed{self.model_seed}_best.pth"
        checkpoint_savepath = os.path.join(pretrain_dir, ckpt_name)
        
        if not os.path.exists(checkpoint_savepath):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_savepath}")
        
        print(f"[CHECKPOINT] Loading from {checkpoint_savepath}")
        ckpt = torch.load(checkpoint_savepath, map_location=self.device)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        
        # Handle DataParallel prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")
        
        # Freeze if last layer only
        if self.finetune_mode == 'last':
            print("[FREEZE] Freezing all layers except classifier head")
            for p in self.model.parameters():
                p.requires_grad = False
            
            # Unfreeze classifier head based on architecture
            last_linear = find_last_linear(self.model)
            if last_linear is not None:
                for p in last_linear.parameters():
                    p.requires_grad = True
                print(f"[FREEZE] Unfrozen classifier head: {last_linear}")
            else:
                raise ValueError("Could not find classifier head to unfreeze")
    
    def _setup_training_components(self):
        """Setup training components (criterion, optimizer, early stopper)."""
        self.criterion = get_criterion(self.model_name)
        
        # Training hyperparameters from config
        model_config = self.config["MODELS"].get(self.model_name.upper(), {})
        lr = float(model_config.get("LEARNING_RATE", 1e-5))
        self.num_epochs = int(model_config.get("NUM_EPOCHS", 50))
        
        # Only optimize trainable parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model_path = os.path.join(
            self.checkpoint_dir,
            f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{self._run_suffix()}_best.pth"
        )
        
        self.early_stopper = EarlyStopping(
            model_path=self.model_path,
            patience=self.config.get("PATIENCE", 10),
            mode='max',
            start_from_epoch=self.config.get("START_FROM_EPOCH", 20)
        )
        
        self.start_epoch = load_checkpoint_if_exists(self.model_path, self.model, self.optimizer)
        print(f"[TRAIN] lr={lr}, epochs={self.num_epochs}, batch_size={self.config['TRAIN_BATCH_SIZE']}, finetune_mode={self.finetune_mode}")
    
    def _subset_val_mask(self, dataset_name, dataset, full_val_mask, target_size, seed):
        """
        Subset validation mask to target size using random sampling.
        
        Args:
            dataset_name: Name of the dataset
            dataset: Dataset object
            full_val_mask: Full validation mask (boolean array)
            target_size: Target size for subset
            seed: Random seed for reproducibility
        
        Returns:
            Boolean mask subset to target size
        """
        candidate_indices = np.where(full_val_mask)[0]
        if len(candidate_indices) == 0 or target_size <= 0:
            return full_val_mask
        
        # If target size is larger than available, return full mask
        if target_size >= len(candidate_indices):
            return full_val_mask
        
        # Simple random sampling without class balance
        rng = np.random.default_rng(seed=seed)
        selected_indices = rng.choice(candidate_indices, size=target_size, replace=False)
        
        subset_val_mask = np.full_like(full_val_mask, False, dtype=bool)
        subset_val_mask[selected_indices] = True
        
        return subset_val_mask
    
    def setup_dataloaders(self, dataset_name, dataset, train_batch_size, eval_batch_size, 
                          generator, model_seed, exclude_domains=None):
        """Setup train and validation dataloaders with few-shot sampling."""
        # Create few-shot train mask with balanced per-class sampling
        train_mask = get_finetune_train_mask(
            dataset_name, 
            dataset, 
            self.target_domain_idx,
            avg_examples_per_class=self.finetune_size,
            seed=self.finetune_data_seed,
            budget=self.budget,
            domain_type=self.spatial_split_types
        )
        
        # Get full validation mask from target domain
        full_val_mask = get_domain_split_mask(
            dataset_name, 
            dataset, 
            self.target_domain_idx, 
            split='val',
            domain_type=self.spatial_split_types
        )
        
        # Subset validation mask based on train_val_ratio
        # val_size = train_size * train_val_ratio
        train_size = int(train_mask.sum())
        target_val_size = int(train_size * self.train_val_ratio)
        
        # Subset validation mask while preserving class balance
        val_mask = self._subset_val_mask(
            dataset_name, 
            dataset, 
            full_val_mask, 
            target_val_size, 
            seed=self.finetune_data_seed
        )
        
        try:
            print(f"[MASK] Finetune selected={int(train_mask.sum())}, Val selected={int(val_mask.sum())} (target={target_val_size}, ratio={self.train_val_ratio})")
        except Exception:
            pass
        
        # Create worker init function with model_seed
        worker_init_fn = functools.partial(seed_worker, model_seed=model_seed)
        
        train_loader = get_domain_dataloader(
            dataset_name, dataset, train_mask, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=16, 
            pin_memory=True, 
            worker_init_fn=worker_init_fn, 
            generator=generator
        )
        
        val_loader = get_domain_dataloader(
            dataset_name, dataset, val_mask, 
            batch_size=eval_batch_size, 
            shuffle=False, 
            num_workers=16, 
            pin_memory=True, 
            worker_init_fn=worker_init_fn, 
            generator=generator
        )
        
        return train_loader, val_loader

    def _is_training_completed(self) -> bool:
        """Check if training is already completed by looking for checkpoint and completed log."""
        # Check if checkpoint exists
        if not os.path.exists(self.model_path):
            return False
        # Check logs for completion marker
        log_pattern = f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{self._run_suffix()}_*.log"
        import glob
        log_files = glob.glob(os.path.join(self.log_dir, log_pattern))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "Training completed in" in content:
                        print(f"✅ Few-shot finetune already completed: {os.path.basename(log_file)}")
                        return True
            except Exception:
                continue
        return False


def main(config, pretrain_domain_idx, target_domain_idx, finetune_size, model_seed, 
         finetune_data_seed, finetune_mode='all', budget=None, train_val_ratio=0.5):
    """Main training function."""
    assert pretrain_domain_idx != target_domain_idx, \
        "WARNING: Attempt to finetune on the same domain where model is pretrained on"
    
    trainer = FewshotTransferTrainer(
        config,
        pretrain_domain_idx=pretrain_domain_idx,
        target_domain_idx=target_domain_idx,
        finetune_size=finetune_size,
        model_seed=model_seed,
        finetune_data_seed=finetune_data_seed,
        finetune_mode=finetune_mode,
        budget=budget,
        train_val_ratio=train_val_ratio,
    )
    # Skip if already completed
    if trainer._is_training_completed():
        print("⏭️  Skipping few-shot transfer - already completed")
        return
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot transfer training')
    parser.add_argument('-d', '--pretrain_domain', type=int, required=True, 
                       help='Domain used in pretraining')
    parser.add_argument('-f', '--finetune_domain', type=int, required=True, 
                       help='Target domain for finetuning')
    parser.add_argument('-n', '--finetune_size', type=int, required=True, 
                       help='n samples per class for finetune')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to config file')
    parser.add_argument('--finetune_mode', type=str, choices=['all', 'last'], 
                       default='all', help='Finetune entire model or only last layer')
    parser.add_argument('--budget', type=int, default=None, 
                       help='Optional global budget after per-class sampling')
    parser.add_argument('--train_val_ratio', type=float, default=0.5,
                       help='Validation set size as ratio of train set size (default: 0.5, meaning val is half of train)')
    args = parser.parse_args()
    
    # Handle relative config paths
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        config_path = os.path.abspath(os.path.join(project_root, 'SatOT', args.config))
    else:
        config_path = args.config
    
    CONFIG = load_config(config_path)
    
    # Set defaults if not present
    CONFIG.setdefault("CHECKPOINT_ROOT", os.path.join("1_training_results", "checkpoints"))
    CONFIG.setdefault("LOG_ROOT", os.path.join("1_training_results", "logs"))
    
    # Get seeds from config (one-to-one: use the same seed for pretrain and finetune)
    model_seeds = CONFIG.get("MODEL_SEEDS", [48329])
    
    # Run training using identical seed for model and finetune data
    for model_seed in model_seeds:
        main(
            CONFIG, 
            args.pretrain_domain, 
            args.finetune_domain, 
            args.finetune_size, 
            model_seed, 
            model_seed,  # finetune_data_seed equals model_seed
            finetune_mode=args.finetune_mode, 
            budget=args.budget,
            train_val_ratio=args.train_val_ratio
        )
