import os
import pickle
import numpy as np
import pandas as pd
import unicodedata
import re
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import unquote_plus
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def clean_text(text) -> str:
    if pd.isna(text) or not text:
        return ""
    text = unquote_plus(str(text)).replace('+', ' ')
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class GeoYFCCBase(Dataset):
    def __init__(self, root: str = '.', metadata_file: Optional[str] = None,
                 domain_idx: Optional[int] = None, expand_multilabel: bool = True):
        self.root = root
        self.num_classes = 1261
        if metadata_file is None:
            # Default CSV or Parquet path
            csv_path = os.path.join(root, "geoyfcc_all_metadata_before_cleaning.csv")
            parquet_path = os.path.join(root, "geoyfcc_all_metadata_before_cleaning.csv.parquet")
            if os.path.exists(csv_path):
                metadata_file = csv_path
            elif os.path.exists(parquet_path):
                metadata_file = parquet_path
            else:
                raise FileNotFoundError(f"No metadata file found in {root}")
        
        # Load metadata
        if metadata_file.endswith(".csv"):
            self.df = pd.read_csv(metadata_file)
        else:
            self.df = pd.read_parquet(metadata_file)

        # Convert numeric columns (adjust to new column names)
        numeric_cols = ["photo_id", "lat", "lon", "country_id", "label_id", "imagenet5k_id"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Filter by domain_id if specified
        if domain_idx is not None and "country_id" in self.df.columns:
            self.df = self.df[self.df["country_id"] == domain_idx].reset_index(drop=True)
        
        # Expand multi-label if requested
        if expand_multilabel:
            self._expand_to_single_label()

    def _expand_to_single_label(self):
        """Expand multi-label samples to single-label samples.

        This method handles cases where label_id might contain multiple labels.
        If your data is already single-label, this can be skipped.
        """
        print("[INFO] Expanding multi-label dataset to single-label...")
        expanded_rows = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Expanding samples"):
            # Parse label_id - handle different formats
            label_ids = row.get("label_ids", [])

            # Handle string representation of lists like "[1, 2, 3]"
            if isinstance(label_ids, str):
                try:
                    # Remove brackets and split by comma
                    if label_ids.startswith('[') and label_ids.endswith(']'):
                        label_ids = label_ids[1:-1]
                    if label_ids.strip():
                        label_ids = [int(x.strip()) for x in label_ids.split(',') if x.strip()]
                    else:
                        label_ids = []
                except (ValueError, AttributeError):
                    # If it's just a single number as string
                    try:
                        label_ids = [int(label_ids)]
                    except ValueError:
                        label_ids = []

            # Handle actual list or other iterable
            elif hasattr(label_ids, '__iter__') and not isinstance(label_ids, str):
                try:
                    label_ids = [int(x) for x in label_ids if x is not None]
                except (ValueError, TypeError):
                    label_ids = []

            # Handle single numeric value
            elif isinstance(label_ids, (int, float)) and not pd.isna(label_ids):
                label_ids = [int(label_ids)]
            else:
                label_ids = []

            # Skip samples with no labels
            if not label_ids:
                continue

            # Create one sample for each label
            for label_id in label_ids:
                new_row = row.copy()
                new_row['label_id'] = label_id  # Single label
                new_row['original_yfcc_row_id'] = row.get('yfcc_row_id', idx)
                # Create unique ID for this expanded sample
                new_row['yfcc_row_id'] = f"{row.get('yfcc_row_id', idx)}_{label_id}"
                expanded_rows.append(new_row)

        # Replace the original dataframe with expanded one
        self.df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        print(f"[INFO] Expanded to {len(self.df)} single-label samples")

    def get_coordinates(self, idx: int) -> Tuple[Optional[float], Optional[float]]:
        """Return (latitude, longitude) for a sample index, or (None, None)."""
        row = self.df.iloc[idx]
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.notna(lat) and pd.notna(lon):
            return float(lat), float(lon)
        return None, None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        lat, lon = self.get_coordinates(idx)
        
        # Extract text features (handle missing values)
        text_features = {
            "title": clean_text(row.get("title", "")),
            "description": clean_text(row.get("description", "")),
            "user_tags": clean_text(row.get("usertags", "")),
        }
        combined_text = " ".join([text_features["title"], text_features["description"]]).strip()
        
        sample = {
            "yfcc_row_id": row.get("yfcc_row_id", idx),
            "original_yfcc_row_id": row.get("original_yfcc_row_id", row.get("yfcc_row_id", idx)),
            "label": int(row.get("label_id", -1)) if pd.notna(row.get("label_id")) else -1,
            "label_name": row.get("label_name", ""),
            "country": row.get("country_name", ""),
            "country_id": int(row.get("country_id", -1)) if pd.notna(row.get("country_id")) else -1,
            "imagenet5k_id": int(row.get("imagenet5k_id", -1)) if pd.notna(row.get("imagenet5k_id")) else -1,
            "split": row.get("split", ""),
            "is_train": row.get("split", "") == "train",
            "photo_id": row.get("photo_id"),
            "image_key": row.get("image_key", ""),
            "text_features": text_features,
            "title": text_features["title"],
            "description": text_features["description"],
            "user_tags": text_features["user_tags"],
            "combined_text": combined_text,
            "latitude": lat,
            "longitude": lon,
            "has_coordinates": lat is not None and lon is not None
        }
        return sample

    def get_domain(self, idx: int) -> int:
        """Return the country_id (domain) for a sample."""
        sample = self.__getitem__(idx)
        return sample['country_id']

class GeoYFCCText(GeoYFCCBase):
    """GeoYFCC dataset with automatic text filtering and fast reload via filtered pickle."""

    def __init__(self, root: str = '.', metadata_file: Optional[str] = None,
                domain_idx: Optional[int] = None, split: Optional[str] = None,
                filter_missing_text: bool = True, save_filtered: bool = True,
                expand_multilabel: bool = True, verbose: bool = True):
        self.root = root
        self.filtered_file = os.path.join(root, "geoyfcc_text_filtered_single_label.pkl")
        
        # If filtered file exists, load it
        if os.path.exists(self.filtered_file):
            print(f"[INFO] Loading filtered single-label dataset from {self.filtered_file}")
            with open(self.filtered_file, "rb") as f:
                self.df = pickle.load(f)
            print(f"[INFO] Loaded {len(self.df)} single-label samples from filtered file")
            self.num_classes = 1261
        else:
            # Initialize parent class
            super().__init__(root, metadata_file, domain_idx, expand_multilabel=expand_multilabel)
            
            self.filter_text_enabled = filter_missing_text
            self.apply_text_filter()

            # CHANGED: Only set split if 'split' column doesn't exist
            if 'split' not in self.df.columns or self.df['split'].isna().any():
                self.set_split()
            else:
                print(f"[INFO] Using existing 'split' column from CSV")

            self.save_filtered_enabled = save_filtered
            self.save_filtered_dataset()
        
        self.split = split
        
        # Apply split filtering if specified
        if self.split is not None:
            self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)
            print(f"[INFO] {len(self.df)} samples after applying '{self.split}' split")

    def set_split(self, seed=42):
        print("[INFO] Creating train/val/test split based on is_train column and country_id...")
        self.df["split"] = self.df["is_train"].apply(lambda x: "test" if not x else "train")

        # For each country, split training data into 60% train and 20% val
        np.random.seed(seed)
        for country_id in self.df["country_id"].unique():
            train_idx = self.df.index[(self.df["country_id"] == country_id) & (self.df["split"] == "train")].to_numpy()
            if len(train_idx) == 0:
                continue
            train_idx = np.random.permutation(train_idx)
            n_train = int(0.75 * len(train_idx))
            self.df.loc[train_idx[n_train:], "split"] = "val"

        self.assess_split()

    def assess_split(self):
        for country_id in self.df["country_id"].unique():
            # Print split sizes for this country
            n_country_train = ((self.df["country_id"] == country_id) & (self.df["split"] == "train")).sum()
            n_country_val = ((self.df["country_id"] == country_id) & (self.df["split"] == "val")).sum()
            n_country_test = ((self.df["country_id"] == country_id) & (self.df["split"] == "test")).sum()
            n_country_total = n_country_train + n_country_val + n_country_test
            
            pct_train = 100 * n_country_train / n_country_total if n_country_total > 0 else 0
            pct_val = 100 * n_country_val / n_country_total if n_country_total > 0 else 0
            pct_test = 100 * n_country_test / n_country_total if n_country_total > 0 else 0
            
            print(f"[INFO] Country {country_id} → train: {n_country_train} ({pct_train:.1f}%), "
                f"val: {n_country_val} ({pct_val:.1f}%), test: {n_country_test} ({pct_test:.1f}%)")

    def apply_text_filter(self):
        """Filter out samples with empty combined text."""
        if self.filter_text_enabled:
            print("[INFO] Filtering out samples with empty combined text...")
            valid_mask = self.df.apply(
                lambda row: bool(
                    (
                        "" if pd.isna(row.get("title")) else str(row.get("title"))
                        + " " +
                        "" if pd.isna(row.get("description")) else str(row.get("description"))
                    ).strip()
                ),
                axis=1
            )
            self.df = self.df[valid_mask].reset_index(drop=True)
            print(f"[INFO] {len(self.df)} samples remaining after text filtering")

    def save_filtered_dataset(self):
        """Save filtered dataset to pickle file for faster loading."""
        if self.save_filtered_enabled:
            print(f"[INFO] Saving filtered single-label dataset to {self.filtered_file} ...")
            print(f"[INFO] Saving {len(self.df)} samples")
            with open(self.filtered_file, "wb") as f:
                pickle.dump(self.df, f)
            print("[INFO] Filtered single-label dataset saved.")

    def plot_world_map(self, sample_limit: Optional[int] = None,
                       figsize=(15, 8), alpha=0.5, color='red', save_path=None):
        """Plot sample locations on world map."""
        n_samples = len(self.df) if sample_limit is None else min(sample_limit, len(self.df))
        lats, lons = [], []
        for i in range(n_samples):
            lat, lon = self.get_coordinates(i)
            if lat is not None and lon is not None:
                lats.append(lat)
                lons.append(lon)
        if len(lats) == 0:
            print("[WARNING] No samples with valid coordinates found.")
            return
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.scatter(lons, lats, color=color, alpha=alpha, s=2, transform=ccrs.PlateCarree())
        plt.title(f"GeoYFCC Single-Label Sample Locations (n={len(lats)})")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def _filter_dataset(dataset, mask, file_suffix=None):
    """Return a copy of dataset filtered to rows where mask is True."""
    copy = type(dataset).__new__(type(dataset))
    for attr, val in dataset.__dict__.items():
        setattr(copy, attr, dataset.df[mask].reset_index(drop=True) if attr == 'df' else val)
    if file_suffix and hasattr(copy, 'filtered_file'):
        base = os.path.splitext(os.path.basename(copy.filtered_file))[0]
        copy.filtered_file = os.path.join(os.path.dirname(copy.filtered_file), f"{base}{file_suffix}.pkl")
    return copy


def get_domain_indices_from_geoyfcc(dataset, domain_idx: int) -> Tuple[List[int], 'GeoYFCCBase']:
    if not hasattr(dataset, 'df') or 'country_id' not in dataset.df.columns:
        raise ValueError("Dataset must have a 'df' attribute with 'country_id' column")
    mask = dataset.df['country_id'] == domain_idx
    return dataset.df.index[mask].tolist(), _filter_dataset(dataset, mask, f"_domain_{domain_idx}")


def get_domain_indices_by_country_name(dataset, country_name: str) -> Tuple[List[int], 'GeoYFCCBase']:
    if not hasattr(dataset, 'df') or 'country_name' not in dataset.df.columns:
        raise ValueError("Dataset must have a 'df' attribute with 'country_name' column")
    mask = dataset.df['country_name'].str.lower() == country_name.lower()
    safe = re.sub(r'[^\w\-_]', '_', country_name.lower())
    return dataset.df.index[mask].tolist(), _filter_dataset(dataset, mask, f"_country_{safe}")

def list_available_domains_from_geoyfcc(root_path: str, return_countries: bool=False, force_refresh: bool=False, split='train'):
    """List all available domains with caching for maximum speed."""
    
    # Cache file path - updated for single-label dataset
    cache_file_name = f"{split}_domains_cache_single_label.pkl" if split is not None else "domains_cache_single_label.pkl"
    cache_file = os.path.join(root_path, cache_file_name)
    
    # Try to load from cache first (unless force_refresh is True)
    if not force_refresh and os.path.exists(cache_file):
        try:
            print("[INFO] Loading domains from single-label cache...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            domain_ids = cached_data['domain_ids']
            domain_countries = cached_data['domain_countries']
            domain_counts = cached_data['domain_counts']
            
            print(f"[INFO] Loaded {len(domain_ids)} domains from cache")
            
            # Display results
            print("\n[INFO] Available domains in single-label dataset:")
            print("Domain ID | Country | Sample Count")
            print("-" * 50)
            for domain_id in sorted(domain_ids):
                country = domain_countries[domain_id]
                count = domain_counts[domain_id]
                print(f"{domain_id:9} | {country:25} | {count:6}")
            
            if return_countries:
                return domain_ids, [domain_countries[domain_id] for domain_id in domain_ids]
            return domain_ids
            
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}")
            print("[INFO] Falling back to reading single-label dataset...")
    
    # If cache doesn't exist or failed to load, use the single-label dataset
    print("[INFO] Loading single-label dataset to analyze domains...")
    
    try:
        print("[INFO] Creating temporary GeoYFCCText instance to get single-label data...")
        temp_dataset = GeoYFCCText(root=root_path, split=split)
        df = temp_dataset.df
            
    except Exception as e:
        print(f"[ERROR] Failed to load single-label dataset: {e}")
        print("[INFO] Falling back to raw metadata file...")
        
        # Find metadata file
        csv_path = os.path.join(root_path, "metadata.csv")
        parquet_path = os.path.join(root_path, "metadata.parquet")
        
        if os.path.exists(parquet_path):
            metadata_file = parquet_path
            print(f"[INFO] Using Parquet file: {metadata_file}")
            df = pd.read_parquet(metadata_file)
        elif os.path.exists(csv_path):
            metadata_file = csv_path
            print(f"[INFO] Using CSV file: {metadata_file}")
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No metadata file found in {root_path}")
    
    print(f"[INFO] Loaded {len(df)} single-label samples from dataset")
    
    # Get unique domains directly from the dataset
    if 'country_id' in df.columns and 'country_name' in df.columns:
        # Count samples per domain using pandas groupby (much faster)
        domain_counts = df['country_id'].value_counts().to_dict()
        domain_names = df.groupby('country_id')['country_name'].first().to_dict()
    else:
        # Fallback: check what columns are available
        print(f"[WARNING] Expected columns 'country_id' and 'country_name' not found.")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Dataset doesn't contain required 'country_id' and 'country_name' columns")
    
    # Sort and display
    domains = [(domain_id, domain_names[domain_id], count) 
               for domain_id, count in sorted(domain_counts.items())]
    
    print("\n[INFO] Available domains in single-label dataset:")
    print("Domain ID | Country | Sample Count")
    print("-" * 50)
    for domain_id, country, count in domains:
        print(f"{domain_id:9} | {country:25} | {count:6}")

    domain_ids = [domain_id for domain_id, _ in sorted(domain_counts.items())]
    
    # Save to cache for next time
    try:
        print("[INFO] Saving domains to single-label cache...")
        cache_data = {
            'domain_ids': domain_ids,
            'domain_countries': domain_names,
            'domain_counts': domain_counts
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[INFO] Cached domains to {cache_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save cache: {e}")

    if return_countries:
        domain_countries = [country for _, country, _ in domains]
        return domain_ids, domain_countries
    
    return domain_ids

def get_split_indices(dataset, split) -> List[int]:
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"Unknown split: {split}")
    return dataset.df.index[dataset.df['split'] == split].tolist()

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    root_path = os.path.join(repo_root, "data", "geoyfcc")
    print("=" * 60)
    print("[TEST] Loading GeoYFCCText single-label dataset")
    print("=" * 60)

    dataset_all = GeoYFCCText(root=root_path, metadata_file=None, split='train')
    print(f"[INFO] Total single-label samples: {len(dataset_all)}")
    