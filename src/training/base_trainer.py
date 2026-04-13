"""
Base trainer class for domain adaptation pretraining.
Contains all common functionality shared between different pretraining variants.
"""

import argparse
import logging
import time
import datetime
import os
import sys
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.models import get_model
from src.data.load_datasets import (
    load_dataset, get_domain_split_mask, get_domain_dataloader, seed_worker
)
from src.training import get_train_eval_functions, get_criterion
from src.core.utils import (
    EarlyStopping, setup_logging, setup_directories, setup_seeds,
    load_checkpoint_if_exists, format_metrics_vision, format_metrics_text,
)

import yaml
from src.core.config_utils import load_config


class BasePretrainTrainer:
    """Base class for domain adaptation pretraining."""
    
    def __init__(self, config, pretrain_domain_idx, model_seed, exclude_domains=None):
        """Initialize the trainer with configuration and parameters."""
        self.config = config
        self.pretrain_domain_idx = pretrain_domain_idx
        self.exclude_domains = exclude_domains
        self.model_seed = model_seed
        
        # Setup random seeds
        self.g = setup_seeds(model_seed)
        
        # Basic setup
        self.task_name = "pretrain"
        self.task_directory_name = "1_pretrain"
        self.start_time = time.time()
        
        # Setup device (single-GPU: cuda:0 if available)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dataset_name = config["DATASET_NAME"]
        self.spatial_split_types = config["DOMAIN_TYPE"]
        self.model_name = config["MODEL_NAME"]
        
        self.dataset = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.early_stopper = None
        self.logger = None
        
        self._setup_directories()
        self._setup_logging()
        
        # Load dataset
        self._load_dataset()
        
        # Setup model and training components
        self._setup_model()
        self._setup_training_components()
    
    def _setup_directories(self):
        self.checkpoint_dir = os.path.join(self.config["CHECKPOINT_ROOT"], self.task_directory_name, self.model_name)
        self.log_dir = os.path.join(self.config["LOG_ROOT"], self.task_directory_name, self.model_name)
        print(f"[DIR] checkpoint_dir={self.checkpoint_dir}")
        print(f"[DIR] log_dir={self.log_dir}")
        setup_directories([self.checkpoint_dir, self.log_dir])
    
    def _setup_logging(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        domain_tag = f"_domain{self.pretrain_domain_idx}" if self.pretrain_domain_idx is not None else ""
        log_filename = f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{domain_tag}{self._run_suffix()}_seed{self.model_seed}_{timestamp}.log"
        setup_logging(os.path.join(self.log_dir, log_filename))
        self.logger = logging.getLogger(__name__)
        
        print(f"Log stored at file logs/{log_filename}")
        self.logger.info(f"Log stored at file logs/{log_filename}")
    
    def _load_dataset(self):
        print("Loading Dataset...")
        self.logger.info("Loading Dataset...")
        self.dataset = load_dataset(self.dataset_name, root_dir=self.config["DATA_DIR"])
        try:
            print(f"[DATA] Loaded {self.dataset_name} with {len(self.dataset)} samples")
        except Exception:
            pass
    
    def _setup_model(self):
        print(f"Loading model {self.model_name}...")
        self.logger.info(f"Loading model {self.model_name}...")
        self.model = get_model(self.model_name, num_classes=self.dataset.num_classes, device=self.device)
        self.model = self.model.to(self.device)

        model_config = self.config["MODELS"].get(self.model_name.upper(), {})
        
        if model_config.get("FREEZE_ENCODER", False):
            print("Freezing encoder parameters...")
            if hasattr(self.model, 'bert'):
                if hasattr(self.model.bert, 'bert'):  # BertForSingleLabel has nested structure
                    for param in self.model.bert.bert.parameters():
                        param.requires_grad = False
                else:
                    raise(ValueError)
            
            # Print trainable vs frozen parameter counts
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
        
        print(f"[MODEL] {self.model_name} on device {self.device}")
    
    def _setup_training_components(self):
        self.criterion = get_criterion(self.model_name)
        
        # Training hyperparameters
        model_config = self.config["MODELS"].get(self.model_name.upper(), {})
        
        # Get learning rate based on whether encoder is frozen
        if model_config.get("FREEZE_ENCODER", False):
            # Use higher learning rate for classification head only
            lr = float(model_config.get("CLASSIFIER_LEARNING_RATE", 1e-3))
        else:
            # Use lower learning rate for full model finetuning
            lr = float(model_config.get("DEFAULT_LEARNING_RATE", 2e-5))

        print(f"Using learning rate: {lr}")
        print(f"Encoder frozen: {model_config.get('FREEZE_ENCODER', False)}")

        self.num_epochs = int(model_config.get("NUM_EPOCHS", 50))
        
        print("MAKE SURE TO LOAD MODEL FIRST")
        # Get optimizer type from config
        optimizer_type = model_config.get("DEFAULT_OPTIMIZER", "AdamW").lower()

        # Get weight decay
        weight_decay = model_config.get("DEFAULT_WEIGHT_DECAY", 0.01)

        # Create optimizer based on type
        if optimizer_type == "adamw":
            self.optimizer = optim.AdamW(                                                                                                                                                                                         
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
            print(f"Using AdamW optimizer with lr={lr}, weight_decay={weight_decay}")
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
            print(f"Using Adam optimizer with lr={lr}, weight_decay={weight_decay}")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'Adam' or 'AdamW'")
        
        domain_tag = f"_domain{self.pretrain_domain_idx}" if self.pretrain_domain_idx is not None else ""
        model_path = os.path.join(
            self.checkpoint_dir,
            f"{self.task_name}_{self.spatial_split_types}_{self.model_name}{domain_tag}{self._run_suffix()}_seed{self.model_seed}_best.pth"
        )
        
        self.early_stopper = EarlyStopping(
            model_path=model_path,
            patience=self.config["PATIENCE"],
            mode='max',
            start_from_epoch=self.config["START_FROM_EPOCH"]
        )
        
        self.start_epoch = load_checkpoint_if_exists(model_path, self.model, self.optimizer)
        print(f"[TRAIN] lr={lr}, epochs={self.num_epochs}, batch_size={self.config['TRAIN_BATCH_SIZE']}")

    def _run_suffix(self) -> str:
        # Default: no suffix; subclasses can override
        return ""
    
    def setup_dataloaders(self, dataset_name, dataset, pretrain_domain_idx, train_batch_size, 
                         eval_batch_size, generator, model_seed):
        train_mask = get_domain_split_mask(dataset_name, dataset, pretrain_domain_idx, split='train')
        val_mask = get_domain_split_mask(dataset_name, dataset, pretrain_domain_idx, split='val')
        try:
            train_count = int(train_mask.sum())
            val_count = int(val_mask.sum())
            print(f"[MASK] Single-domain: train_count={train_count}, val_count={val_count}")
        except Exception:
            pass
        
        # Create worker init function with model_seed
        worker_init_fn = functools.partial(seed_worker, model_seed=model_seed)

        model_config = self.config["MODELS"].get(self.model_name.upper(), {})
        tokenizer_type = model_config.get("TOKENIZER_TYPE", None)
        
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
    
    def train_epoch(self, model, train_dataloader, criterion, optimizer, device, model_name, logger):
        model_config = self.config["MODELS"].get(self.model_name.upper(), {})
        grad_accumulation = model_config.get("GRAD_ACCUMULATION", False)
        k_val = model_config.get("K_VAL", None)
        train_fn, _ = get_train_eval_functions(model_name, grad_accumulation=grad_accumulation)
        
        if model_name in ('bert_multilabel', 'bert_singlelabel'):
            if grad_accumulation:
                train_total_loss, train_avg_loss, train_examples, train_acc, train_top3_correct, train_top5_correct = train_fn(
                    model, train_dataloader, k_val, criterion, optimizer, device
                )
            else:
                train_total_loss, train_avg_loss, train_examples, train_acc, train_top3_correct, train_top5_correct = train_fn(
                    model, train_dataloader, criterion, optimizer, device
                )
            train_top3_acc = train_top3_correct / train_examples if train_examples > 0 else 0
            train_top5_acc = train_top5_correct / train_examples if train_examples > 0 else 0
            return train_total_loss, train_avg_loss, train_examples, train_acc, train_top3_acc, train_top5_acc
        else:
            train_total_loss, train_avg_loss, train_examples, train_correct, train_top3_correct, train_top5_correct = train_fn(
                model, train_dataloader, criterion, optimizer, device
            )
            train_acc = train_correct / train_examples if train_examples > 0 else 0
            train_top3_acc = train_top3_correct / train_examples if train_examples > 0 else 0
            train_top5_acc = train_top5_correct / train_examples if train_examples > 0 else 0
            return train_total_loss, train_avg_loss, train_examples, train_acc, train_top3_acc, train_top5_acc
    
    def validate_epoch(self, model, val_dataloader, device, model_name):
        _, eval_fn = get_train_eval_functions(model_name)
        
        if model_name in ('bert_multilabel', 'bert_singlelabel'):
            val_examples, val_acc, val_top3_correct, val_top5_correct = eval_fn(model, val_dataloader, device)
            val_top3_acc = val_top3_correct / val_examples if val_examples > 0 else 0
            val_top5_acc = val_top5_correct / val_examples if val_examples > 0 else 0
            return val_examples, val_acc, val_top3_acc, val_top5_acc
        else:
            val_examples, val_correct, val_top3_correct, val_top5_correct = eval_fn(model, val_dataloader, device)
            val_acc = val_correct / val_examples if val_examples > 0 else 0
            val_top3_acc = val_top3_correct / val_examples if val_examples > 0 else 0
            val_top5_acc = val_top5_correct / val_examples if val_examples > 0 else 0
            return val_examples, val_acc, val_top3_acc, val_top5_acc
    
    def log_epoch_metrics(self, epoch, num_epochs, train_total_loss, train_avg_loss, train_acc, 
                         val_acc, train_top3_acc, train_top5_acc, val_top3_acc, val_top5_acc, model_name, logger):
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
        
        # Log top-k accuracies
        print(f"\tTrain Top-3 Accuracy: {train_top3_acc:.6f}, Top-5 Accuracy: {train_top5_acc:.6f}")
        print(f"\tVal Top-3 Accuracy: {val_top3_acc:.6f}, Top-5 Accuracy: {val_top5_acc:.6f}")
        logger.info(f"\tTrain Top-3 Accuracy: {train_top3_acc:.6f}, Top-5 Accuracy: {train_top5_acc:.6f}")
        logger.info(f"\tVal Top-3 Accuracy: {val_top3_acc:.6f}, Top-5 Accuracy: {val_top5_acc:.6f}")
    
    def train(self):
        """Main training loop."""
        print("Loading Dataloaders...")
        self.logger.info("Loading Dataloaders...")
        train_batch_size = self.config["TRAIN_BATCH_SIZE"]
        eval_batch_size = self.config["EVAL_BATCH_SIZE"]
        
        train_dataloader, val_dataloader = self.setup_dataloaders(
            self.dataset_name, self.dataset, self.pretrain_domain_idx, 
            train_batch_size, eval_batch_size, self.g, self.model_seed
        )
        
        print("Start Training")
        self.logger.info("Start Training")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")

            train_total_loss, train_avg_loss, _, train_acc, train_top3_acc, train_top5_acc = self.train_epoch(
                self.model, train_dataloader, self.criterion, self.optimizer, 
                device=self.device, model_name=self.model_name, logger=self.logger
            )

            print(f"\tTrain:")
            print(f"\t\t total_loss: {train_total_loss:.6f}, avg_loss: {train_avg_loss:.6f}, train accuracy: {train_acc:.6f}")
            print(f"\t\t train top-3 accuracy: {train_top3_acc:.6f}, train top-5 accuracy: {train_top5_acc:.6f}")
            self.logger.info(f"\tTrain:")
            self.logger.info(f"\t\t total_loss: {train_total_loss:.6f}, avg_loss: {train_avg_loss:.6f}, train accuracy: {train_acc:.6f}")
            self.logger.info(f"\t\t train top-3 accuracy: {train_top3_acc:.6f}, train top-5 accuracy: {train_top5_acc:.6f}")

            _, val_acc, val_top3_acc, val_top5_acc = self.validate_epoch(
                self.model, val_dataloader, self.device, self.model_name
            )

            print(f"\tValidation")
            print(f"\t\t validation accuracy: {val_acc:.6f}")
            print(f"\t\t validation top-3 accuracy: {val_top3_acc:.6f}, validation top-5 accuracy: {val_top5_acc:.6f}")
            self.logger.info(f"\tValidation")
            self.logger.info(f"\t\t validation accuracy: {val_acc:.6f}")
            self.logger.info(f"\t\t validation top-3 accuracy: {val_top3_acc:.6f}, validation top-5 accuracy: {val_top5_acc:.6f}")

            if self.early_stopper(val_acc, epoch+1, self.model, self.optimizer):
                print(f"Improvement is not observed for the past {self.early_stopper.patience} epochs. Early Stopping...")
                print(f"Loading Best Model Checkpoint from epoch {self.early_stopper.best_epoch} with validation loss {self.early_stopper.best_metric:.4f}")
                self.logger.info(f"Improvement is not observed for the past {self.early_stopper.patience} epochs. Early Stopping...")
                self.logger.info(f"Loading Best Model Checkpoint from epoch {self.early_stopper.best_epoch} with validation accuracy {-self.early_stopper.best_metric:.4f}")
                break

        time_taken = str(datetime.timedelta(seconds=int(time.time() - self.start_time)))
        print(f"Training completed in {time_taken}")
        self.logger.info(f"Training completed in {time_taken}")
