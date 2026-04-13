"""
ResNet Embeddings Extraction for Vision Datasets

This module provides generic ResNet50 feature extraction for vision datasets (FMoW, GeoYFCCImage).
Follows the same pattern as bert_embeddings.py for consistency.

Author: Migration from backup code
Date: March 17, 2026
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Union
import os


def get_resnet50_feature_extractor(device: Optional[str] = None) -> nn.Module:
    """
    Load ResNet50 model as a feature extractor (without final classification layer).
    
    Args:
        device (str, optional): 'cuda' or 'cpu'; auto-detect if None
        
    Returns:
        nn.Module: ResNet50 feature extractor outputting 2048-dimensional features
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load pre-trained ResNet50
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Remove final classification layer to get features
    modules = list(resnet50.children())[:-1]  # Remove avgpool and fc
    feature_extractor = nn.Sequential(*modules)
    
    feature_extractor.eval()
    feature_extractor.to(device)
    
    return feature_extractor


def get_default_transforms() -> transforms.Compose:
    """
    Get default ImageNet preprocessing transforms.
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def generate_resnet_embeddings(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 512,
    device: Optional[str] = None,
    model: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Generate ResNet50 embeddings for a dataset.
    
    Args:
        dataset: PyTorch Dataset object (should return (image, label, metadata) or (image, label))
        batch_size: Batch size for DataLoader
        device: 'cuda' or 'cpu'; auto-detect if None
        model: Pre-loaded model; if None, will load ResNet50 feature extractor
        
    Returns:
        embeddings: torch.Tensor of shape (num_samples, 2048)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model if not provided
    if model is None:
        model = get_resnet50_feature_extractor(device)
    
    # DataLoader for batching
    def collate_fn(batch):
        """Handle both 2-tuple and 3-tuple dataset formats."""
        if len(batch[0]) == 3:
            # (image, label, metadata) format
            images = torch.stack([item[0] for item in batch])
            return images
        else:
            # (image, label) format
            images = torch.stack([item[0] for item in batch])
            return images
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    all_embeddings = []
    with torch.no_grad():
        for images in tqdm(loader, desc="Generating ResNet embeddings"):
            images = images.to(device)
            
            # Extract features
            features = model(images)
            
            # Flatten spatial dimensions (from [B, 2048, 1, 1] to [B, 2048])
            features = features.view(features.size(0), -1)
            
            all_embeddings.append(features.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"[INFO] Generated embeddings shape: {embeddings.shape}")
    return embeddings


def extract_embeddings_by_domain(
    dataset_class: type,
    root_dir: str,
    dataset_name: str,
    num_domains: int,
    output_dir: str,
    batch_size: int = 512,
    device: Optional[str] = None
):
    """
    Extract ResNet50 embeddings for all domains and splits.
    
    Saves embeddings in PyTorch format (.pt) organized by:
        output_dir/
            train/
                embeddings_train_domain0.pt
                embeddings_train_domain1.pt
                ...
            val/
                ...
            test/
                ...
    
    Args:
        dataset_class: Dataset class (FMoW or GeoYFCCImage)
        root_dir: Root directory containing dataset
        dataset_name: Name of dataset for logging
        num_domains: Number of domains to process
        output_dir: Output directory for embeddings
        batch_size: Batch size for processing
        device: 'cuda' or 'cpu'
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load feature extractor once
    model = get_resnet50_feature_extractor(device)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    
    # Process each domain and split
    for split in ['train', 'val', 'test']:
        print(f"\n[INFO] Processing {split} split...")
        
        for domain_idx in range(num_domains):
            print(f"[INFO] Domain {domain_idx}/{num_domains-1}...")
            
            # Load dataset for this domain and split
            dataset = dataset_class(root=root_dir, split=split, domain_idx=domain_idx)
            
            if len(dataset) == 0:
                print(f"[WARNING] No samples for domain {domain_idx}, split {split}. Skipping.")
                continue
            
            # Generate embeddings
            embeddings = generate_resnet_embeddings(
                dataset,
                batch_size=batch_size,
                device=device,
                model=model
            )
            
            # Save embeddings
            output_path = os.path.join(
                output_dir,
                split,
                f"embeddings_{split}_domain{domain_idx}.pt"
            )
            torch.save(embeddings, output_path)
            print(f"[INFO] Saved embeddings to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ResNet50 embeddings for vision datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["fmow", "geoyfcc_image"],
                       help="Dataset name")
    parser.add_argument("--root_dir", type=str, required=True,
                       help="Root directory containing dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for embeddings")
    parser.add_argument("--num_domains", type=int, default=25,
                       help="Number of domains (25 for FMoW, 62 for GeoYFCCImage)")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default=None,
                       help="Device ('cuda' or 'cpu'); auto-detect if not specified")
    
    args = parser.parse_args()
    
    # Import dataset class
    if args.dataset == "fmow":
        from datasets.fmow.fmow import FMoW as DatasetClass
    elif args.dataset == "geoyfcc_image":
        from datasets.geoyfcc_image.geoyfcc_image import GeoYFCCImage as DatasetClass
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Extract embeddings
    extract_embeddings_by_domain(
        dataset_class=DatasetClass,
        root_dir=args.root_dir,
        dataset_name=args.dataset,
        num_domains=args.num_domains,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"\n[INFO] Embedding extraction complete!")
    print(f"[INFO] Embeddings saved to {args.output_dir}")
    
    # Example usage:
    # python src/data/embeddings/resnet_embeddings.py \
    #     --dataset fmow \
    #     --root_dir ./data/fmow \
    #     --output_dir ./data/fmow/embeddings/resnet50 \
    #     --num_domains 25 \
    #     --batch_size 512
