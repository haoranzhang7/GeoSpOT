"""
GeoYFCCImage Dataset Module

This module provides a PyTorch Dataset wrapper for the GeoYFCC dataset (image-based version).
Distinct from the text-based geoyfcc module (used by collaborator for BERT experiments).

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
from PIL import Image


class GeoYFCCImage(Dataset):
    """
    GeoYFCC image dataset with country-level domain support.
    
    Features:
    - 1261-class image classification
    - 62 country domain support (all countries)
    - Image-based (distinct from text-based geoyfcc dataset)
    - Organized by class folders
    
    Args:
        root (str): Root directory containing GeoYFCC image data
        metadata_file (str, optional): Path to metadata CSV/Parquet file
        split (str, optional): Data split ('train', 'val', 'test', or None for all)
        domain_idx (int, optional): Country domain index (0 to 61)
    """
    
    def __init__(
        self,
        root: str = './data/geoyfcc_image',
        metadata_file: Optional[str] = None,
        split: Optional[str] = None,
        domain_idx: Optional[int] = None
    ):
        self.root = root
        self.split = split
        self.domain_idx = domain_idx
        self.num_classes = 1261
        self.num_domains = 62  # All 62 countries
        
        # Load metadata
        if metadata_file is None:
            # Default metadata paths
            csv_path = os.path.join(root, "geoyfcc_all_metadata_before_cleaning.csv")
            parquet_path = os.path.join(root, "geoyfcc_all_metadata_before_cleaning.csv.parquet")
            
            if os.path.exists(parquet_path):
                metadata_file = parquet_path
            elif os.path.exists(csv_path):
                metadata_file = csv_path
            else:
                raise FileNotFoundError(f"No metadata file found in {root}")
        
        # Load metadata
        if metadata_file.endswith(".parquet"):
            self.df = pd.read_parquet(metadata_file)
        else:
            self.df = pd.read_csv(metadata_file)
        
        # Convert numeric columns
        numeric_cols = ["photo_id", "lat", "lon", "country_id", "label_id", "imagenet5k_id"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Preprocess country domains
        self._preprocess_country_domains()
        
        # Apply split filter if specified
        if split is not None:
            if 'split' in self.df.columns:
                self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            else:
                print(f"[Warning] 'split' column not found in metadata. Skipping split filter.")
        
        # Apply domain filter if specified
        if domain_idx is not None:
            self.df = self.df[self.df['domain_idx'] == domain_idx].reset_index(drop=True)
        
        # Build image path to sample mapping
        self._build_sample_list()
        
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
        
        Uses all 62 countries and assigns domain_idx from 0 to 61.
        """
        if 'country_id' not in self.df.columns:
            raise ValueError("Metadata must contain 'country_id' column for country domains")
        
        # Get unique countries
        unique_countries = sorted(self.df['country_id'].dropna().unique())
        
        # Create mapping from country_id to domain_idx
        self.country_to_domain = {country: idx for idx, country in enumerate(unique_countries)}
        self.domain_to_country = {idx: country for country, idx in self.country_to_domain.items()}
        
        # Add domain_idx column
        self.df['domain_idx'] = self.df['country_id'].map(self.country_to_domain)
        
        # Store number of domains
        self.num_domains = len(unique_countries)
        
        print(f"[GeoYFCCImage] Preprocessed {len(self.df)} samples across {self.num_domains} countries")
    
    def _build_sample_list(self):
        """
        Build list of valid image samples.
        
        Assumes image directory structure:
            root/
                class_0/
                    photo_id_1.jpg
                    photo_id_2.jpg
                class_1/
                    ...
        """
        self.samples = []
        
        ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".ppm", ".pgm", ".webp"}
        
        # Iterate through metadata
        for idx, row in self.df.iterrows():
            label_id = row.get('label_id', None)
            photo_id = row.get('photo_id', None)
            
            if pd.isna(label_id) or pd.isna(photo_id):
                continue
            
            # Construct expected image path
            class_dir = f"class_{int(label_id)}"
            # Try different extensions
            for ext in ALLOWED_EXTENSIONS:
                img_path = os.path.join(self.root, class_dir, f"{int(photo_id)}{ext}")
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(label_id), idx))
                    break
        
        print(f"[GeoYFCCImage] Found {len(self.samples)} valid image files")
        
        if len(self.samples) == 0:
            print(f"[Warning] No images found in {self.root}. Expected structure: root/class_N/*.jpg")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple of (image, label, metadata)
            - image: torch.Tensor of shape (3, 224, 224)
            - label: int (class label 0-1260)
            - metadata: dict with keys 'photo_id', 'country_id', 'domain_idx', 'lat', 'lon', etc.
        """
        img_path, label, metadata_idx = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to load image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get metadata from dataframe
        row = self.df.iloc[metadata_idx]
        metadata_dict = {
            'photo_id': row.get('photo_id', None),
            'country_id': row.get('country_id', None),
            'domain_idx': row.get('domain_idx', None),
            'lat': row.get('lat', None),
            'lon': row.get('lon', None),
            'split': row.get('split', None)
        }
        
        return image, label, metadata_dict
    
    def get_domain_split_mask(self, domain_idx: int, split: str) -> np.ndarray:
        """
        Get boolean mask for samples in a specific domain and split.
        
        Args:
            domain_idx (int): Country domain index (0 to 61)
            split (str): Data split ('train', 'val', or 'test')
            
        Returns:
            np.ndarray: Boolean mask of shape (len(df),)
        """
        domain_mask = self.df['domain_idx'] == domain_idx
        
        if 'split' in self.df.columns:
            split_mask = self.df['split'] == split
        else:
            print(f"[Warning] 'split' column not found. Returning domain mask only.")
            return domain_mask.values
        
        return (domain_mask & split_mask).values
    
    def get_split_mask(self, split: str) -> np.ndarray:
        """
        Get boolean mask for samples in a specific split (across all domains).
        
        Args:
            split (str): Data split ('train', 'val', or 'test')
            
        Returns:
            np.ndarray: Boolean mask of shape (len(df),)
        """
        if 'split' in self.df.columns:
            return (self.df['split'] == split).values
        else:
            print(f"[Warning] 'split' column not found. Returning all samples.")
            return np.ones(len(self.df), dtype=bool)
    
    def get_domain_info(self) -> dict:
        """
        Get information about country domains.
        
        Returns:
            dict: Mapping of domain_idx to country_id and sample counts
        """
        domain_info = {}
        for domain_idx in range(self.num_domains):
            country_id = self.domain_to_country.get(domain_idx, None)
            count = (self.df['domain_idx'] == domain_idx).sum()
            domain_info[domain_idx] = {
                'country_id': country_id,
                'num_samples': count
            }
        return domain_info


def get_geoyfcc_image_dataloader(
    dataset: GeoYFCCImage,
    split_mask: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for a subset of the GeoYFCCImage dataset.
    
    Args:
        dataset (GeoYFCCImage): GeoYFCCImage dataset instance
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
    print("Loading GeoYFCCImage dataset...")
    dataset = GeoYFCCImage(root="../../data/geoyfcc_image", split=None)
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of country domains: {dataset.num_domains}")
    
    print(f"\nDomain information (first 10):")
    domain_info = dataset.get_domain_info()
    for idx in range(min(10, len(domain_info))):
        info = domain_info[idx]
        print(f"  Domain {idx}: Country {info['country_id']} ({info['num_samples']} samples)")
    
    if len(dataset) > 0:
        print(f"\nSample data:")
        image, label, metadata = dataset[0]
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {label}")
        print(f"  Metadata: {metadata}")
