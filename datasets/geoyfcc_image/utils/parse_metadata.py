"""
Metadata parsing utilities for GeoYFCCImage dataset.

These utilities help process and clean the GeoYFCC metadata CSV files.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def parse_metadata_csv(
    csv_path: str,
    clean_invalid: bool = True
) -> pd.DataFrame:
    """
    Parse GeoYFCC metadata CSV file with data cleaning.
    
    Args:
        csv_path (str): Path to metadata CSV file
        clean_invalid (bool): Whether to remove invalid entries
        
    Returns:
        pd.DataFrame: Cleaned metadata dataframe
    """
    print(f"[parse_metadata] Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[parse_metadata] Initial samples: {len(df)}")
    
    if clean_invalid:
        # Remove entries with missing critical fields
        initial_count = len(df)
        
        df = df.dropna(subset=['photo_id', 'label_id', 'country_id'])
        print(f"[parse_metadata] Removed {initial_count - len(df)} samples with missing critical fields")
        
        # Ensure valid coordinates
        df = df[(df['lat'] >= -90) & (df['lat'] <= 90)]
        df = df[(df['lon'] >= -180) & (df['lon'] <= 180)]
        print(f"[parse_metadata] After coordinate validation: {len(df)} samples")
    
    # Convert types
    df['photo_id'] = pd.to_numeric(df['photo_id'], errors='coerce')
    df['label_id'] = pd.to_numeric(df['label_id'], errors='coerce')
    df['country_id'] = pd.to_numeric(df['country_id'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    print(f"[parse_metadata] Final samples: {len(df)}")
    
    return df


def filter_by_countries(
    df: pd.DataFrame,
    country_ids: List[int]
) -> pd.DataFrame:
    """
    Filter metadata to specific countries.
    
    Args:
        df (pd.DataFrame): Metadata dataframe
        country_ids (List[int]): List of country IDs to keep
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    initial_count = len(df)
    df = df[df['country_id'].isin(country_ids)].reset_index(drop=True)
    print(f"[filter_by_countries] Filtered to {len(country_ids)} countries: {len(df)} samples (from {initial_count})")
    return df


def add_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Add train/val/test split column to metadata.
    
    Performs stratified split by country to ensure each country has samples in all splits.
    
    Args:
        df (pd.DataFrame): Metadata dataframe
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Dataframe with 'split' column added
    """
    from sklearn.model_selection import train_test_split
    
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    df['split'] = None
    
    # Split stratified by country
    for country_id in df['country_id'].unique():
        country_mask = df['country_id'] == country_id
        country_indices = df[country_mask].index.tolist()
        
        if len(country_indices) < 3:
            # Too few samples - assign all to train
            df.loc[country_indices, 'split'] = 'train'
            continue
        
        # First split: train vs. (val + test)
        train_idx, temp_idx = train_test_split(
            country_indices,
            train_size=train_ratio,
            random_state=random_state
        )
        
        # Second split: val vs. test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=random_state
        )
        
        df.loc[train_idx, 'split'] = 'train'
        df.loc[val_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'
    
    print(f"[add_split] Train: {(df['split'] == 'train').sum()} samples")
    print(f"[add_split] Val: {(df['split'] == 'val').sum()} samples")
    print(f"[add_split] Test: {(df['split'] == 'test').sum()} samples")
    
    return df


def get_country_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics per country.
    
    Args:
        df (pd.DataFrame): Metadata dataframe
        
    Returns:
        pd.DataFrame: Statistics dataframe with columns ['country_id', 'num_samples', 'num_classes']
    """
    stats = []
    
    for country_id in sorted(df['country_id'].unique()):
        country_df = df[df['country_id'] == country_id]
        stats.append({
            'country_id': country_id,
            'num_samples': len(country_df),
            'num_classes': country_df['label_id'].nunique()
        })
    
    stats_df = pd.DataFrame(stats)
    return stats_df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parse_metadata.py <metadata_csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Parse and clean metadata
    df = parse_metadata_csv(csv_path, clean_invalid=True)
    
    # Get country statistics
    stats = get_country_statistics(df)
    print("\nCountry statistics:")
    print(stats.head(10))
    
    # Add splits
    df = add_train_val_test_split(df, random_state=42)
    
    # Save cleaned metadata
    output_path = csv_path.replace('.csv', '_cleaned.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned metadata to {output_path}")
