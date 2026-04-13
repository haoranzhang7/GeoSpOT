# GeoYFCCImage Dataset Module

This module provides PyTorch Dataset support for the GeoYFCC dataset (image-based version), distinct from the text-based geoyfcc module used by the collaborator.

## Overview

**GeoYFCCImage** is an image classification dataset with:
- **1261 classes** (Yahoo Flickr Creative Commons categories)
- **62 country domains** (all countries in the dataset)
- **Geographic diversity** with lat/lon coordinates
- **Multi-modal support** (images + location metadata)

## Distinction from geoyfcc_text

| Feature | geoyfcc_text (Collaborator) | geoyfcc_image (This module) |
|---------|----------------------------|------------------------------|
| **Task** | Text classification | Image classification |
| **Model** | BERT | ResNet50, etc. |
| **Input** | Text descriptions | Image pixels |
| **Classes** | 1261 | 1261 |
| **Domains** | 62 countries | 62 countries |

Both modules use the same underlying GeoYFCC dataset but target different modalities.

## Installation

No additional requirements beyond standard PyTorch and torchvision.

## Dataset Structure

```
data/geoyfcc_image/
├── geoyfcc_all_metadata_before_cleaning.csv  # Metadata file
├── class_0/                                   # Images organized by class
│   ├── 12345.jpg
│   ├── 12346.jpg
│   └── ...
├── class_1/
│   └── ...
├── ...
└── class_1260/
    └── ...
```

## Usage

### Basic Loading

```python
from datasets.geoyfcc_image.geoyfcc_image import GeoYFCCImage

# Load full dataset (all splits, all 62 countries)
dataset = GeoYFCCImage(root="./data/geoyfcc_image")

# Load specific split
train_dataset = GeoYFCCImage(root="./data/geoyfcc_image", split="train")

# Load specific domain (country)
domain_0_dataset = GeoYFCCImage(root="./data/geoyfcc_image", split="train", domain_idx=0)
```

### Domain Information

```python
# Get domain metadata
domain_info = dataset.get_domain_info()
for idx, info in domain_info.items():
    print(f"Domain {idx}: Country {info['country_id']} ({info['num_samples']} samples)")

# Example output:
# Domain 0: Country 1 (8234 samples)
# Domain 1: Country 2 (6921 samples)
# ...
```

### Creating DataLoaders

```python
from datasets.geoyfcc_image.geoyfcc_image import get_geoyfcc_image_dataloader

# Get domain-specific mask
train_mask = dataset.get_domain_split_mask(domain_idx=0, split="train")

# Create DataLoader
train_loader = get_geoyfcc_image_dataloader(
    dataset=dataset,
    split_mask=train_mask,
    batch_size=256,
    shuffle=True,
    num_workers=4
)

# Iterate
for images, labels, metadata in train_loader:
    # images: torch.Tensor of shape (batch_size, 3, 224, 224)
    # labels: torch.Tensor of shape (batch_size,) with values 0-1260
    # metadata: dict with keys 'photo_id', 'country_id', 'domain_idx', 'lat', 'lon', etc.
    pass
```

### Integration with git/SatOT

```python
from src.data.load_datasets import load_dataset

# Load via unified interface
dataset = load_dataset(dataset_name="geoyfcc_image", root_dir="./data")

# Use with training pipeline
from src.training.pretrain_by_domain import main as pretrain

# Train on domain 0
pretrain(config_path="configs/experiments/pretrain_geoyfcc_image.yaml")
```

## Domain Definitions

### Country-Level Domains

The dataset includes **all 62 countries** present in GeoYFCC:
- Domain indices: 0 to 61
- Each country is a separate domain
- Balanced experiments can select subsets of countries

## Dataset Statistics

| Split | Samples | Classes | Domains |
|-------|---------|---------|---------|
| Train | ~80,000 | 1261 | 62 |
| Val | ~15,000 | 1261 | 62 |
| Test | ~15,000 | 1261 | 62 |

*Note: Exact counts depend on metadata availability*

## Image Preprocessing

All images use standard ImageNet preprocessing:
- Resize to 256x256
- Center crop to 224x224
- Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Metadata Utilities

### Parsing and Cleaning

```python
from datasets.geoyfcc_image.utils import parse_metadata_csv

# Parse metadata with cleaning
df = parse_metadata_csv(
    "data/geoyfcc_image/geoyfcc_all_metadata_before_cleaning.csv",
    clean_invalid=True
)
```

### Adding Train/Val/Test Splits

```python
from datasets.geoyfcc_image.utils import add_train_val_test_split

# Add stratified splits (by country)
df = add_train_val_test_split(
    df,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)
```

### Country Statistics

```python
from datasets.geoyfcc_image.utils import get_country_statistics

# Get per-country statistics
stats = get_country_statistics(df)
print(stats.head(10))
```

## Configuration

Example configuration file (`configs/datasets/geoyfcc_image.yaml`):

```yaml
DATASET:
  name: "geoyfcc_image"
  num_classes: 1261
  num_countries: 62
  domain_type: "country"
  image_size: 224
  
DATA:
  root_dir: "./data/geoyfcc_image"
  metadata_file: "geoyfcc_all_metadata_before_cleaning.csv"
  
TRANSFORMS:
  resize: 256
  crop: 224
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

## Experiments

### Pretraining by Domain

Train a model on each country domain separately:

```bash
python src/training/pretrain_by_domain.py \
    --config configs/experiments/pretrain_geoyfcc_image.yaml \
    --pretrain_domain_idx 0
```

### Zero-Shot Evaluation

Evaluate transfer between all domain pairs:

```bash
python src/evaluation/zeroshot_test_eval.py \
    --config configs/experiments/zeroshot_geoyfcc_image.yaml
```

### OT Distance Computation

Compute optimal transport distances between domains:

```bash
python compute_distances/ot_distance.py \
    --config configs/experiments/ot_distance_geoyfcc_image.yaml
```

## Notes

- **Image Organization:** Assumes images are in `class_N/` folders with photo IDs as filenames
- **Metadata Required:** Metadata CSV must contain `photo_id`, `label_id`, `country_id`, `lat`, `lon`
- **Missing Images:** Invalid or missing images are skipped during dataset initialization
- **Compatible with git/SatOT** training and evaluation infrastructure

## Troubleshooting

**Issue:** `FileNotFoundError: No metadata file found`
- **Solution:** Ensure `geoyfcc_all_metadata_before_cleaning.csv` exists in root directory

**Issue:** `No images found` warning
- **Solution:** Check image directory structure - should be `root/class_N/*.jpg`

**Issue:** Slow loading
- **Solution:** Use `num_workers > 0` in DataLoader for parallel loading

**Issue:** Different number of domains than expected
- **Solution:** Check metadata - not all 62 countries may have samples after filtering

## See Also

- [FMoW Dataset](../fmow/README.md) - Satellite imagery dataset
- [GeoYFCC Text Dataset](../geoyfcc/) - Text-based version (collaborator's)
- [git/SatOT Training Guide](../../docs/MAIN_UPDATES.md)
