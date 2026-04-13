"""
FMoW (Functional Map of the World) Dataset Module

This module provides a PyTorch Dataset wrapper for the FMoW dataset from the WILDS benchmark.
Supports country-level domain splits with top-k country filtering and temporal filtering.

Author: Migration from backup code
Date: March 17, 2026
"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Optional, Tuple
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset


class FMoW(Dataset):
    """
    FMoW dataset wrapper with country-level domain support.
    
    Features:
    - 62-class satellite image classification
    - Top-k country domain filtering
    - Temporal filtering (2016-2017 data)
    - WILDS benchmark integration
    
    Args:
        root (str): Root directory containing FMoW data
        split (str, optional): Data split ('train', 'val', 'test', or None for all)
        domain_idx (int, optional): Country domain index (0 to num_countries-1)
        top_k_countries (int): Number of top countries by training samples (default: 25)
        download (bool): Whether to download the dataset if not present
    """
    
    def __init__(
        self,
        root: str = './data/fmow',
        split: Optional[str] = None,
        domain_idx: Optional[int] = None,
        top_k_countries: int = 25,
        download: bool = False
    ):
        self.root = root
        self.split = split
        self.domain_idx = domain_idx
        self.top_k_countries = top_k_countries
        self.num_classes = 62
        
        # Load WILDS FMoW dataset
        self.wilds_dataset = get_dataset(dataset="fmow", root_dir=root, download=download)
        
        # Load and preprocess metadata
        metadata_path = os.path.join(root, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.df = pd.read_csv(metadata_path, index_col=0)
        
        # Filter sequestered images (not used in WILDS)
        self.df = self.df[self.df['split'] != 'seq'].reset_index(drop=True)
        
        # Preprocess to add country domain indices
        self._preprocess_country_domains()
        
        # Apply split filter if specified
        if split is not None:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        # Apply domain filter if specified
        if domain_idx is not None:
            self.df = self.df[self.df['domain_idx'] == domain_idx].reset_index(drop=True)
        
        # Apply temporal filter (2016-2017 only)
        self.df = self.df[(self.df['year'] == 14.0) | (self.df['year'] == 15.0)].reset_index(drop=True)
        
        # Default transforms (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _preprocess_country_domains(self):
        """
        Preprocess metadata to assign country domain indices.
        
        Selects top-k countries by number of training samples and assigns
        domain_idx from 0 to k-1.
        """
        # Filter for training split to count samples
        train_df = self.df[self.df['split'] == 'train']
        
        # Count samples per country_code
        country_counts = train_df['country_code'].value_counts()
        
        # Select top-k countries
        top_countries = country_counts.head(self.top_k_countries).index.tolist()
        
        # Filter to only top countries
        self.df = self.df[self.df['country_code'].isin(top_countries)].reset_index(drop=True)
        
        # Create mapping from country_code to domain_idx
        self.country_to_domain = {country: idx for idx, country in enumerate(top_countries)}
        self.domain_to_country = {idx: country for country, idx in self.country_to_domain.items()}
        
        # Add domain_idx column
        self.df['domain_idx'] = self.df['country_code'].map(self.country_to_domain)
        
        # Store number of domains
        self.num_domains = len(top_countries)
        
        print(f"[FMoW] Preprocessed {len(self.df)} samples across {self.num_domains} countries")
        print(f"[FMoW] Top countries: {top_countries}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple of (image, label, metadata)
            - image: torch.Tensor of shape (3, 224, 224)
            - label: int (class label 0-61)
            - metadata: dict with keys 'country_code', 'domain_idx', 'year', etc.
        """
        row = self.df.iloc[idx]
        
        # Get image from WILDS dataset
        wilds_idx = row.name  # Original index in WILDS dataset
        image, label, metadata = self.wilds_dataset[wilds_idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Construct metadata dict
        metadata_dict = {
            'country_code': row['country_code'],
            'domain_idx': row['domain_idx'],
            'year': row['year'],
            'region': row.get('region', None),
            'split': row['split']
        }
        
        return image, label, metadata_dict
    
    def get_domain_split_mask(self, domain_idx: int, split: str) -> np.ndarray:
        """
        Get boolean mask for samples in a specific domain and split.
        
        Args:
            domain_idx (int): Country domain index (0 to num_domains-1)
            split (str): Data split ('train', 'val', or 'test')
            
        Returns:
            np.ndarray: Boolean mask of shape (len(df),)
        """
        # Temporal filter (2016-2017)
        time_mask = (self.df['year'] == 14.0) | (self.df['year'] == 15.0)
        
        # Domain filter
        domain_mask = self.df['domain_idx'] == domain_idx
        
        # Split filter
        split_mask = self.df['split'] == split
        
        return (domain_mask & time_mask & split_mask).values
    
    def get_split_mask(self, split: str) -> np.ndarray:
        """
        Get boolean mask for samples in a specific split (across all domains).
        
        Args:
            split (str): Data split ('train', 'val', or 'test')
            
        Returns:
            np.ndarray: Boolean mask of shape (len(df),)
        """
        # Temporal filter (2016-2017)
        time_mask = (self.df['year'] == 14.0) | (self.df['year'] == 15.0)
        
        # Split filter
        split_mask = self.df['split'] == split
        
        return (time_mask & split_mask).values
    
    def get_domain_info(self) -> dict:
        """
        Get information about country domains.
        
        Returns:
            dict: Mapping of domain_idx to country_code and sample counts
        """
        domain_info = {}
        for domain_idx in range(self.num_domains):
            country = self.domain_to_country[domain_idx]
            count = (self.df['domain_idx'] == domain_idx).sum()
            domain_info[domain_idx] = {
                'country_code': country,
                'num_samples': count
            }
        return domain_info


def get_fmow_dataloader(
    dataset: FMoW,
    split_mask: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for a subset of the FMoW dataset.
    
    Args:
        dataset (FMoW): FMoW dataset instance
        split_mask (np.ndarray): Boolean mask for subset selection
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader instance
    """
    from torch.utils.data import DataLoader, Subset
    
    # Get indices where mask is True
    indices = np.where(split_mask)[0].tolist()
    
    # Create subset
    subset = Subset(dataset, indices)
    
    # Create DataLoader
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    print("Loading FMoW dataset...")
    dataset = FMoW(root="../../data/fmow", split=None, top_k_countries=25)
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of country domains: {dataset.num_domains}")
    
    print(f"\nDomain information:")
    domain_info = dataset.get_domain_info()
    for idx, info in domain_info.items():
        print(f"  Domain {idx}: {info['country_code']} ({info['num_samples']} samples)")
    
    print(f"\nSample data:")
    image, label, metadata = dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")
    print(f"  Metadata: {metadata}")
