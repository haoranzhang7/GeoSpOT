"""
Utility functions for logging, early stopping, and other common tasks.
"""

import logging
import os
import torch


class EarlyStopping:
    """Early stopping utility with model checkpointing."""
    
    def __init__(self, model_path, patience=10, min_delta=0, mode='max', start_from_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.start_from_epoch = start_from_epoch
        
        self.model_path = model_path
        self.counter = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
    
    def __call__(self, updated_value, epoch, model, optimizer):
        if self.mode == 'max':
            updated_value = -updated_value
        if updated_value < self.best_metric:
            self.best_metric = updated_value
            self.best_epoch = epoch
            self.counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, self.model_path)
        elif updated_value >= (self.best_metric - self.min_delta):
            if epoch >= self.start_from_epoch:
                self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def setup_logging(log_path):
    """Setup logging configuration."""
    logging.basicConfig(
        filename=log_path,
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    return log_path


def setup_directories(base_dirs):
    """Create directories if they don't exist."""
    for dir_path in base_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


def setup_seeds(model_seed):
    """Setup random seeds for reproducibility."""
    import numpy as np
    
    g = torch.Generator()
    g.manual_seed(model_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    np.random.seed(model_seed)
    
    return g


def load_checkpoint_if_exists(model_path, model, optimizer):
    """Load model checkpoint if it exists."""
    if os.path.exists(model_path):
        try:
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            return start_epoch
        except Exception as e:
            print(f"Error loading checkpoint from {model_path}: {e}")
            return 0
    else:
        return 0


def format_metrics_vision(train_total_loss, train_avg_loss, train_acc, val_acc):
    """Format metrics for vision models."""
    train_msg = f"\tTrain:\n\t\t total_loss: {train_total_loss:.6f}, avg_loss: {train_avg_loss:.6f}, train accuracy: {train_acc:.6f}"
    val_msg = f"\tValidation\n\t\t validation accuracy: {val_acc:.6f}"
    return train_msg, val_msg


def format_metrics_text(train_total_loss, train_avg_loss, train_acc, val_acc):
    """Format metrics for text models."""
    train_msg = f"\tTrain:\n\t\t total_loss: {train_total_loss:.6f}, avg_loss: {train_avg_loss:.6f}, exact match accuracy: {train_acc:.6f}"
    val_msg = f"\tValidation\n\t\t exact match accuracy: {val_acc:.6f}"
    return train_msg, val_msg


def get_criterion(model_name):
    """Get criterion based on model name."""
    if model_name == 'bert_multilabel':
        return torch.nn.BCEWithLogitsLoss()
    else:
        return torch.nn.CrossEntropyLoss()