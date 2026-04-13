import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Union, Optional

import sys
import os

try:
    from geoclip import LocationEncoder
    GEOCLIP_AVAILABLE = True
except ImportError:
    GEOCLIP_AVAILABLE = False
    LocationEncoder = None

try:
    from huggingface_hub import hf_hub_download
    from satclip.load import get_satclip
    SATCLIP_AVAILABLE = True
except ImportError:
    SATCLIP_AVAILABLE = False
    print("Warning: SatCLIP not available. SatCLIP embeddings will be disabled.")

SUPPORTED_EMBEDDING_TYPES = ["geoclip", "satclip_l10", "satclip_l40"]

def generate_geoclip_embeddings(lat_lon_list: List[Tuple[float, float]]) -> torch.Tensor:
    """
    Generate GeoClip embeddings for a list of latitude, longitude coordinates.
    
    Args:
        lat_lon_list: List of tuples containing (latitude, longitude) pairs
        encoder: Optional pre-initialized LocationEncoder. If None, creates a new one.
    
    Returns:
        torch.Tensor: Embeddings with shape (n_locations, embedding_dim)
    """
    if not GEOCLIP_AVAILABLE:
        raise ImportError("GeoClip is not available. Please install it: pip install geoclip")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = LocationEncoder().to(device)
    encoder.eval()  # Set to evaluation mode
    
    # Convert list to tensor and move to device
    gps_data = torch.tensor(lat_lon_list, dtype=torch.float32).to(device)
    
    # Generate embeddings
    with torch.no_grad():  # Disable gradient computation for inference
        embeddings = encoder(gps_data)
    
    return embeddings

def generate_satclip_embeddings(lat_lon_list: List[Tuple[float, float]], satclip_type='l10', device='cuda:0') -> torch.Tensor:
    """
    Generate SatClip embeddings for a list of latitude, longitude coordinates.
    
    Args:
        lat_lon_list: List of tuples containing (latitude, longitude) pairs
        encoder: Optional pre-initialized SatClipEncoder. If None, creates a new one.
    
    Returns:
        torch.Tensor: Embeddings with shape (n_locations, embedding_dim)
    """
    if not SATCLIP_AVAILABLE:
        raise ImportError("SatClip is not available. Please install it.")
    
    if satclip_type == 'l10':
        encoder = get_satclip(
            hf_hub_download("microsoft/SatCLIP-ResNet50-L10", "satclip-resnet50-l10.ckpt"),
            device=device,
        )
    elif satclip_type == 'l40':
        encoder = get_satclip(
            hf_hub_download("microsoft/SatCLIP-ResNet50-L40", "satclip-resnet50-l40.ckpt"),
            device=device,
        )
    
    encoder.eval()  # Set to evaluation mode
    
    # lat_lon_list is [(lat, lon), ...] → we need [(lon, lat), ...]
    lon_lat_list = [(lon, lat) for lat, lon in lat_lon_list]

    gps_data = torch.tensor(lon_lat_list, dtype=torch.float64, device=device)

    with torch.no_grad():  # Disable gradient computation for inference
        embeddings = encoder(gps_data)
    
    return embeddings


def generate_location_embeddings(lat_lon_list: List[Tuple[float, float]], 
                                embedding_type: str) -> torch.Tensor:
    """
    Generate location embeddings using the specified embedding type.
    
    Args:
        lat_lon_list: List of tuples containing (latitude, longitude) pairs
        embedding_type: Type of embeddings to generate ('geoclip' or 'satclip')
        encoder: Optional pre-initialized encoder. If None, creates a new one.
    
    Returns:
        torch.Tensor: Embeddings with shape (n_locations, embedding_dim)
    """
    
    if embedding_type.lower() not in SUPPORTED_EMBEDDING_TYPES:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. "
                        f"Supported types: {SUPPORTED_EMBEDDING_TYPES}")
    
    if embedding_type == "geoclip":
        return generate_geoclip_embeddings(lat_lon_list)
    elif embedding_type == "satclip_L10":
        return generate_satclip_embeddings(lat_lon_list, satclip_type='l10')
    elif embedding_type == "satclip_L40":
        return generate_satclip_embeddings(lat_lon_list, satclip_type='l40')
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

def generate_and_save_embeddings(lat_lon_list: List[Tuple[float, float]], 
                                save_path: Union[str, Path],
                                embedding_type: str = "geoclip",
                                format: str = 'torch') -> torch.Tensor:
    """
    Generate location embeddings and save them to the specified path.
    
    Args:
        lat_lon_list: List of tuples containing (latitude, longitude) pairs
        save_path: Path where to save the embeddings
        embedding_type: Type of embeddings to generate ('geoclip' or 'satclip')
        encoder: Optional pre-initialized encoder. If None, creates a new one.
        format: Save format - 'torch' (.pt), 'numpy' (.npy), or 'pickle' (.pkl)
    
    Returns:
        torch.Tensor: The generated embeddings
    """
    embeddings = generate_location_embeddings(lat_lon_list, embedding_type)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'torch':
        if not save_path.suffix:
            save_path = save_path.with_suffix('.pt')
        torch.save(embeddings, save_path)
        
    elif format.lower() == 'numpy':
        if not save_path.suffix:
            save_path = save_path.with_suffix('.npy')
        np.save(save_path, embeddings.numpy())
        
    elif format.lower() == 'pickle':
        if not save_path.suffix:
            save_path = save_path.with_suffix('.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'torch', 'numpy', or 'pickle'")
    
    print("Saved embeddings!")
    
    print(f"Embeddings ({embedding_type}) saved to: {save_path}")
    print(f"Shape: {embeddings.shape}")
    
    return embeddings

def get_available_embedding_types() -> List[str]:
    """
    Get a list of available embedding types based on installed packages.
    
    Returns:
        List[str]: List of available embedding types
    """
    available = []
    if GEOCLIP_AVAILABLE:
        available.append("geoclip")
    if SATCLIP_AVAILABLE:
        available.append("satclip")
    return available