import argparse
import logging
import time, datetime
import os
import sys
import functools
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn
cudnn.deterministic = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is at src/data/embeddings/embeddings.py -> project root is 3 levels up
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.models import get_model, BertForSingleLabel
from src.data.load_datasets import (
    load_dataset, get_domain_split_mask, get_split_mask, get_domain_dataloader, seed_worker
)
from src.core.utils import setup_seeds, setup_directories

from location_embeddings.utils import generate_geoclip_embeddings, generate_satclip_embeddings

import yaml

def load_config(path="configs/datasets/geoyfcc.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    # Derive ROOT_DIR for load_dataset: parent of dataset dir (e.g. ./data from ./data/geoyfcc)
    if "ROOT_DIR" not in cfg:
        paths = cfg.get("PATHS") or {}
        data_dir = paths.get("data_dir", "./data/geoyfcc")
        cfg["ROOT_DIR"] = os.path.dirname(data_dir) or "./data"
    return cfg

def setup_dataloaders(dataset_name, dataset, batch_size, split, model_seed, tokenizer_type=None):
    mask = get_split_mask(dataset_name, dataset, split=split)
    generator = setup_seeds(model_seed)
    worker_init_fn = functools.partial(seed_worker, model_seed=model_seed)
    dataloader = get_domain_dataloader(
        dataset_name, dataset, tokenizer_type, mask, batch_size=batch_size,
        shuffle=False, num_workers=16, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=generator
    )
    return dataloader, mask

def get_pretrained_model(model_type, num_classes=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = get_model(model_type, num_classes=num_classes, device=device)
        # Remove classification head to get features
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = torch.nn.Identity()
    except Exception as e:
        print(e)
    
    model.eval()
    model.to(device)
    return model

def get_domain_labels(dataset_name, dataset, indices):
    """Get domain (country) id for each index. GeoYFCC only."""
    domain_labels = []
    for idx in indices:
        if hasattr(dataset, "get_domain"):
            domain = dataset.get_domain(idx)
        else:
            domain = 0
        domain_labels.append(domain)
    return np.array(domain_labels)

def extract_text_embeddings(
    dataset_name,
    output_dir,
    cfg,
    model_type="bert_singlelabel",
    model_path=None,
    split="train",
    batch_size=128,
    device=None,
    return_embeddings=False,
    model_seed=42,
):
    if dataset_name != "geoyfcc_text":
        raise ValueError("Only geoyfcc_text dataset is supported for embedding extraction.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, root_dir=cfg["ROOT_DIR"])

    dataloader, mask = setup_dataloaders(
        dataset_name, dataset, batch_size, split, model_seed, tokenizer_type=None
    )
    
    # Load model
    print(f"Loading {model_type} model...")
    model = get_pretrained_model(model_type, num_classes=dataset.num_classes, device=device)
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    print(f"Extracting embeddings ({split})...")
    with torch.no_grad():
        for encodings, labels in tqdm(dataloader, desc=f"({split})"):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            # Use base encoder, skip classifier
            outputs = model.bert.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # CLS token embedding
            features = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
            
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indices = np.where(mask)[0]
    
    domain_labels = get_domain_labels(dataset_name, dataset, indices)
    
    if return_embeddings:
        return embeddings, labels, indices, domain_labels
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{dataset_name}_{split}_{model_type}.npz")
    np.savez_compressed(
        save_path,
        embeddings=embeddings,
        labels=labels,
        indices=indices,
        domains=domain_labels,
    )
    print(f"Saved embeddings to {save_path}")
    # Save .pt for OT distance script (expects data/geoyfcc/embeddings/bert.pt)
    if model_type == "bert_singlelabel":
        pt_path = os.path.join(output_dir, "bert.pt")
        torch.save(torch.from_numpy(embeddings).float(), pt_path)
        print(f"Saved OT format to {pt_path}")


def extract_geoclip_embeddings(
    dataset_name,
    output_dir,
    cfg,
    split="train",
    batch_size=128,
    device=None,
    return_embeddings=False,
    model_seed=42,
):
    if dataset_name != "geoyfcc_text":
        raise ValueError("Only geoyfcc_text dataset is supported.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name, root_dir=cfg["ROOT_DIR"])
    mask = get_split_mask(dataset_name, dataset, split)
    indices = np.where(mask)[0]

    data_root = cfg.get("ROOT_DIR", "./data")
    cache_dir = os.path.join(data_root, "geoyfcc")
    coords_path = os.path.join(cache_dir, "coordinates.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")

    if os.path.exists(coords_path) and os.path.exists(labels_path):
        print(f"Loading cached coordinates and labels from {coords_path} and {labels_path}...")
        coords = np.load(coords_path)
        labels = np.load(labels_path)
    else:
        print(f"Extracting coordinates ({split})...")
        coords = []
        labels = []
        for idx in tqdm(indices, desc="Loading coordinates"):
            sample = dataset[idx]
            lat = sample["latitude"]
            lon = sample["longitude"]
            coords.append((lat, lon))
            labels.append(sample["label"])
        coords = np.array(coords, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(coords_path, coords)
        np.save(labels_path, labels)

    print(f"Coords shape: {coords.shape}, Labels shape: {labels.shape}")
    domain_labels = get_domain_labels(dataset_name, dataset, indices)
    print(f"Extracting GeoClip embeddings ({split})...")
    all_embeddings = []
    for i in tqdm(range(0, len(coords), batch_size), desc="Extracting location embeddings"):
        batch_coords = coords[i:i+batch_size]
        batch_embeddings = generate_geoclip_embeddings(batch_coords)
        all_embeddings.append(batch_embeddings.detach().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    coords_array = np.array(coords, dtype=np.float32)
    
    if return_embeddings:
        return embeddings, labels, coords_array, indices, domain_labels
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{dataset_name}_{split}_geoclip.npz")
    np.savez_compressed(save_path, 
                       embeddings=embeddings, 
                       labels=labels, 
                       coords=coords_array, 
                       indices=indices,
                       domains=domain_labels)
    print(f"Saved GeoClip location embeddings to {save_path}")

def extract_satclip_embeddings(
    dataset_name,
    output_dir,
    cfg,
    legendre_polys=10,
    split="train",
    batch_size=128,
    device=None,
    return_embeddings=False,
    model_seed=42,
):
    if dataset_name != "geoyfcc_text":
        raise ValueError("Only geoyfcc_text dataset is supported.")
    if legendre_polys == 10:
        satclip_type = "l10"
    elif legendre_polys == 40:
        satclip_type = "l40"
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name, root_dir=cfg["ROOT_DIR"])
    mask = get_split_mask(dataset_name, dataset, split)
    indices = np.where(mask)[0]

    data_root = cfg.get("ROOT_DIR", "./data")
    cache_dir = os.path.join(data_root, "geoyfcc")
    coords_path = os.path.join(cache_dir, "coordinates.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")

    if os.path.exists(coords_path) and os.path.exists(labels_path):
        print(f"Loading cached coordinates and labels from {coords_path} and {labels_path}...")
        coords = np.load(coords_path)
        labels = np.load(labels_path)
    else:
        print(f"Extracting coordinates ({split})...")
        coords = []
        labels = []
        for idx in tqdm(indices, desc="Loading coordinates"):
            sample = dataset[idx]
            lat = sample["latitude"]
            lon = sample["longitude"]
            coords.append((lat, lon))
            labels.append(sample["label"])
        coords = np.array(coords, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(coords_path, coords)
        np.save(labels_path, labels)

    print(f"Coords shape: {coords.shape}, Labels shape: {labels.shape}")
    domain_labels = get_domain_labels(dataset_name, dataset, indices)
    print(f"Extracting SatCLIP embeddings ({split})...")
    all_embeddings = []
    device = torch.device('cuda:0')
    for i in tqdm(range(0, len(coords), batch_size), desc="Extracting location embeddings"):
        batch_coords = torch.tensor(coords[i:i+batch_size], dtype=torch.float32, device=device)
        batch_embeddings = generate_satclip_embeddings(batch_coords, satclip_type=satclip_type, device=device)
        all_embeddings.append(batch_embeddings.detach().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    
    coords_array = np.array(coords, dtype=np.float32)
    
    if return_embeddings:
        return embeddings, labels, coords_array, indices, domain_labels
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{dataset_name}_{split}_satclip_L{legendre_polys}.npz")
    np.savez_compressed(save_path, 
                       embeddings=embeddings, 
                       labels=labels, 
                       coords=coords_array, 
                       indices=indices,
                       domains=domain_labels)
    print(f"Saved SatCLIP location-only embeddings to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for GeoYFCC (BERT, GeoCLIP, SatCLIP)")
    parser.add_argument("--config", type=str, default="configs/datasets/geoyfcc.yaml", help="Path to config file (e.g. configs/datasets/geoyfcc.yaml)")
    parser.add_argument("--dataset", type=str, default="geoyfcc_text", choices=["geoyfcc_text"], help="Dataset (only geoyfcc_text supported)")
    parser.add_argument("--domain", type=str, default="all", help="Domain index or 'all' (kept for CLI compatibility)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Data split")
    parser.add_argument("--model_type", type=str, default="bert_singlelabel", choices=["bert_singlelabel"], help="Model type (BERT for text)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to custom model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--satclip", action="store_true", help="Extract SatCLIP embeddings")
    parser.add_argument("--geoclip", action="store_true", help="Extract GeoClip embeddings")
    parser.add_argument("--legendre_polys", type=int, default=10, choices=[10, 40], help="Legendre polynomials for SatCLIP")
    parser.add_argument("--model_seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_directories([args.output_dir])
    print(f"Using device: {device}")

    if args.geoclip:
        extract_geoclip_embeddings(
            args.dataset,
            args.output_dir,
            config,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
            return_embeddings=False,
            model_seed=args.model_seed,
        )
    elif args.satclip:
        extract_satclip_embeddings(
            args.dataset,
            args.output_dir,
            config,
            args.legendre_polys,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
            return_embeddings=False,
            model_seed=args.model_seed,
        )
    else:
        extract_text_embeddings(
            args.dataset,
            args.output_dir,
            config,
            model_type=args.model_type,
            model_path=args.model_path,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
            return_embeddings=False,
            model_seed=args.model_seed,
        )

if __name__ == '__main__':
    main()