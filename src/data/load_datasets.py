"""
Dataset loading and processing utilities.
Handles both vision datasets (iNaturalist, FMOW) and text datasets (GeoYFCCText).
"""

from typing import Tuple, List
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer

import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_dataset(dataset_name: str, root_dir: str = "./data"):
    """Load dataset by name."""
    if dataset_name == "geoyfcc_text":
        dataset = load_geoyfcc_text_dataset(root_dir=os.path.join(root_dir, "geoyfcc"))
    elif dataset_name == "fmow":
        dataset = load_fmow_dataset(root_dir=os.path.join(root_dir, "fmow"))
    elif dataset_name == "geoyfcc_image":
        dataset = load_geoyfcc_image_dataset(root_dir=os.path.join(root_dir, "geoyfcc_image"))
    else:
        raise ValueError(f"Dataset name '{dataset_name}' has not been implemented.")
    return dataset


def load_geoyfcc_text_dataset(root_dir: str = "./data/geoyfcc"):
    """Load GeoYFCCText dataset."""
    from datasets.geoyfcc.geoyfcc import GeoYFCCText
    dataset = GeoYFCCText(root=root_dir, split=None)
    return dataset


def load_fmow_dataset(root_dir: str = "./data/fmow", top_k_countries: int = 25, download: bool = False):
    """Load FMoW dataset with top-k country filtering."""
    from datasets.fmow.fmow import FMoW
    dataset = FMoW(root=root_dir, split=None, domain_idx=None, top_k_countries=top_k_countries, download=download)
    return dataset


def load_geoyfcc_image_dataset(root_dir: str = "./data/geoyfcc_image", metadata_file: str = None):
    """Load GeoYFCCImage dataset (all 62 countries)."""
    from datasets.geoyfcc_image.geoyfcc_image import GeoYFCCImage
    dataset = GeoYFCCImage(root=root_dir, metadata_file=metadata_file, split=None, domain_idx=None)
    return dataset

def get_split_mask(dataset_name: str, dataset, split: str):
    """
    Get boolean mask for split only (without domain filtering).

    Args:
        dataset_name: Name of the dataset ("fmow", "geoyfcc_text", or "geoyfcc_image")
        dataset: Dataset object
        split: "train", "val", or "test"

    Returns:
        Boolean mask over dataset samples indicating selected split
    """
    if dataset_name == "geoyfcc_text":
        from datasets.geoyfcc.geoyfcc import get_split_indices
        split_indices = get_split_indices(dataset, split)
        split_mask = np.zeros(len(dataset), dtype=bool)
        split_mask[split_indices] = True
        return split_mask
    
    elif dataset_name == "fmow":
        # FMoW dataset has get_split_mask() method
        return dataset.get_split_mask(split)
    
    elif dataset_name == "geoyfcc_image":
        # GeoYFCCImage dataset has get_split_mask() method
        return dataset.get_split_mask(split)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_domain_split_mask(dataset_name: str, dataset, domain_idx: int, split: str, domain_type: str = "continent"):
    """
    Get boolean mask for domain and split combination.

    For FMoW:
        - Filters dataset by country (domain_idx) and split (train/val/test)
        - Automatically applies temporal filtering (2016-2017 only)
    For GeoYFCCText:
        - Filters dataset by country_id and split (train/val/test)
    For GeoYFCCImage:
        - Filters dataset by country_id and split (train/val/test)

    Args:
        dataset_name: Name of the dataset ("fmow", "geoyfcc_text", or "geoyfcc_image")
        dataset: Dataset object
        domain_idx: Domain index (country index for FMoW/GeoYFCC)
        split: "train", "val", or "test"
        domain_type: Reserved for future use (not currently used)

    Returns:
        Boolean mask over dataset samples indicating selected domain + split
    """
    if dataset_name == "geoyfcc_text":
        from datasets.geoyfcc.geoyfcc import get_domain_indices_from_geoyfcc, get_split_indices

        # Domain mask
        domain_indices, _ = get_domain_indices_from_geoyfcc(dataset, domain_idx)
        domain_mask = np.zeros(len(dataset), dtype=bool)
        domain_mask[domain_indices] = True

        # Split mask
        split_indices = get_split_indices(dataset, split)

        split_mask = np.zeros(len(dataset), dtype=bool)
        split_mask[split_indices] = True

        return domain_mask & split_mask
    
    elif dataset_name == "fmow":
        # FMoW dataset has get_domain_split_mask() method
        # Note: Temporal filtering (2016-2017) is already applied in FMoW.__init__
        return dataset.get_domain_split_mask(domain_idx, split)
    
    elif dataset_name == "geoyfcc_image":
        # GeoYFCCImage dataset has get_domain_split_mask() method
        return dataset.get_domain_split_mask(domain_idx, split)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_domain_subset(dataset_name, dataset, mask):
    """Get subset of dataset based on domain mask."""
    split_indices = np.where(mask)[0]
    
    if dataset_name in ["geoyfcc_text", "fmow", "geoyfcc_image"]:
        # Subset using torch Subset wrapper
        from torch.utils.data import Subset
        domain_subset = Subset(dataset, split_indices)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return domain_subset


def get_domain_dataloader(dataset_name, dataset, tokenizer_type, mask, batch_size, shuffle=False, 
                         num_workers=1, pin_memory=False, worker_init_fn=None, generator=None):
    domain_subset = get_domain_subset(dataset_name, dataset, mask)

    if dataset_name == "geoyfcc_text":
        num_labels = 1261
        collate_fn = lambda b: collate_fn_geoyfcc_text(b, tokenizer_type, num_labels)
    elif dataset_name in ["fmow", "geoyfcc_image"]:
        # Vision datasets: use default collate_fn (handles image, label, metadata tuples)
        collate_fn = None
    else:
        collate_fn = None


    domain_dataloader = DataLoader(
        domain_subset, 
        shuffle=shuffle, 
        sampler=None, 
        collate_fn=collate_fn, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        worker_init_fn=worker_init_fn, 
        generator=generator
    )
    return domain_dataloader

def collate_fn_geoyfcc_text(batch, tokenizer_type, num_labels: int):
    # Ensure batch is always a list (DataLoader may pass a single dict in some worker setups)
    if isinstance(batch, dict):
        batch = [batch]
    text = [item["combined_text"] for item in batch]
    
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        label = item["label"]
        if 0 <= label < num_labels:
            labels[i] = label
        else:
            labels[i] = 0

    if tokenizer_type == 'bert-base-uncased' or tokenizer_type is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encodings = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return encodings, labels

    else:
        raise ValueError(f'Specify a tokenizer_type; right now, we support only bert-base-uncased. Got {tokenizer_type}')


def seed_worker(worker_id, model_seed):
    """Worker initialization function for reproducible data loading."""
    worker_seed = model_seed + worker_id
    np.random.seed(worker_seed)


def get_finetune_train_mask(dataset_name: str, dataset, finetune_domain_idx: int, 
                             avg_examples_per_class: int = 1, seed: int = None, 
                             budget: int = None, domain_type: str = "biome"):
    """
    Create a boolean mask for few-shot finetuning with balanced per-class sampling.
    
    Strategy:
    - Select up to n=avg_examples_per_class samples per class from the target domain train split.
    - If budget is provided and the balanced set size exceeds budget, randomly subsample to the budget size.
    
    Args:
        dataset_name: Name of the dataset ("fmow", "geoyfcc_text", or "geoyfcc_image")
        dataset: Dataset object
        finetune_domain_idx: Domain index for the target domain
        avg_examples_per_class: Number of samples per class to select (default: 1)
        seed: Random seed for reproducibility
        budget: Optional global budget cap after per-class sampling
        domain_type: Reserved for future use
    
    Returns:
        Boolean mask over dataset samples indicating selected samples for few-shot finetuning
    """
    finetune_domain_mask = get_domain_split_mask(
        dataset_name, dataset, finetune_domain_idx, split='train', domain_type=domain_type
    )
    
    # Gather candidate indices in the target domain train split
    candidate_indices = np.where(finetune_domain_mask)[0]
    if len(candidate_indices) == 0:
        return finetune_domain_mask  # empty mask
    
    rng = np.random.default_rng(seed=seed)
    
    # Build per-class buckets
    per_class_indices = [[] for _ in range(dataset.num_classes)]
    for idx in candidate_indices:
        try:
            # Get label based on dataset type
            if dataset_name == "fmow":
                # FMoW: get label from df (y column contains class labels)
                class_id = dataset.df.loc[idx, "y"]
            elif dataset_name in ["geoyfcc_text", "geoyfcc_image"]:
                # GeoYFCC: get label_id from df
                class_id = dataset.df.loc[idx, "label_id"]
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
        except Exception as e:
            raise ValueError(f"Could not get label for index {idx}: {e}")

        if 0 <= class_id < dataset.num_classes:
            per_class_indices[class_id].append(idx)
    
    # Sample up to n per class
    selected_indices = []
    for class_list in per_class_indices:
        if not class_list:
            continue
        if len(class_list) <= avg_examples_per_class:
            selected_indices.extend(class_list)
        else:
            selected_indices.extend(rng.choice(class_list, size=avg_examples_per_class, replace=False))
    
    selected_indices = np.array(selected_indices, dtype=int)
    
    # Optional global budget cap
    if budget is not None and len(selected_indices) > budget:
        selected_indices = rng.choice(selected_indices, size=budget, replace=False)
    
    finetune_train_mask = np.full_like(finetune_domain_mask, False, dtype=bool)
    finetune_train_mask[selected_indices] = True
    return finetune_train_mask