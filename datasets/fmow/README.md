# FMoW Dataset Module

This module provides PyTorch Dataset support for the FMoW (Functional Map of the World) dataset from the WILDS benchmark.

## Overview

**FMoW** is a satellite imagery dataset for land use classification with:
- **62 classes** (building types and land use categories)
- **Country-level domain splits** (top-25 countries by training samples)
- **Temporal filtering** (2016-2017 data only)
- **Geographic diversity** across multiple continents

## Installation

The FMoW dataset requires the `wilds` package:

```bash
pip install wilds>=2.0.0
```

See `additional_requirements.txt` for the exact version requirement.

## Dataset Structure

```
data/fmow/
├── metadata.csv          # Dataset metadata with country codes, years, etc.
├── images/              # Image files organized by split
│   ├── train/
│   ├── val/
│   └── test/
└── embeddings/          # (Optional) Pre-computed embeddings
    └── resnet50/
```

## Usage

### Basic Loading

```python
from datasets.fmow.fmow import FMoW

# Load full dataset (all splits, top-25 countries)
dataset = FMoW(root="./data/fmow", top_k_countries=25)

# Load specific split
train_dataset = FMoW(root="./data/fmow", split="train", top_k_countries=25)

# Load specific domain (country)
domain_0_dataset = FMoW(root="./data/fmow", split="train", domain_idx=0)
```

### Domain Information

```python
# Get domain metadata
domain_info = dataset.get_domain_info()
for idx, info in domain_info.items():
    print(f"Domain {idx}: {info['country_code']} ({info['num_samples']} samples)")

# Example output:
# Domain 0: USA (15234 samples)
# Domain 1: CHN (8921 samples)
# ...
```

### Creating DataLoaders

```python
from datasets.fmow.fmow import get_fmow_dataloader

# Get domain-specific mask
train_mask = dataset.get_domain_split_mask(domain_idx=0, split="train")

# Create DataLoader
train_loader = get_fmow_dataloader(
    dataset=dataset,
    split_mask=train_mask,
    batch_size=256,
    shuffle=True,
    num_workers=4
)

# Iterate
for images, labels, metadata in train_loader:
    # images: torch.Tensor of shape (batch_size, 3, 224, 224)
    # labels: torch.Tensor of shape (batch_size,) with values 0-61
    # metadata: dict with keys 'country_code', 'domain_idx', 'year', etc.
    pass
```

### Integration with git/SatOT

```python
from src.data.load_datasets import load_dataset

# Load via unified interface
dataset = load_dataset(dataset_name="fmow", root_dir="./data")

# Use with training pipeline
from src.training.pretrain_by_domain import main as pretrain

# Train on domain 0
pretrain(config_path="configs/experiments/pretrain_fmow.yaml")
```

## Domain Definitions

### Country-Level Domains (Default)

The dataset uses top-k countries by training sample count:
- **Default k=25** (top 25 countries)
- Countries ranked by training set size
- Domain indices: 0 to k-1

### Temporal Filtering

Only samples from **2016-2017** are included:
- `year == 14.0` (2016)
- `year == 15.0` (2017)

This filtering ensures temporal consistency across experiments.

## Dataset Statistics

| Split | Samples | Classes | Domains |
|-------|---------|---------|---------|
| Train | ~45,000 | 62 | 25 |
| Val | ~8,000 | 62 | 25 |
| Test | ~10,000 | 62 | 25 |

*Note: Exact counts depend on temporal filtering and top-k selection*

## Image Preprocessing

All images use standard ImageNet preprocessing:
- Resize to 256x256
- Center crop to 224x224
- Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Configuration

Example configuration file (`configs/datasets/fmow.yaml`):

```yaml
DATASET:
  name: "fmow"
  num_classes: 62
  num_countries: 25  # Top-25 countries
  domain_type: "country"
  temporal_filter: [14.0, 15.0]  # 2016-2017
  image_size: 224
  
DATA:
  root_dir: "./data/fmow"
  metadata_file: "metadata.csv"
  
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
    --config configs/experiments/pretrain_fmow.yaml \
    --pretrain_domain_idx 0
```

### Zero-Shot Evaluation

Evaluate transfer between all domain pairs:

```bash
python src/evaluation/zeroshot_test_eval.py \
    --config configs/experiments/zeroshot_fmow.yaml
```

### OT Distance Computation

Compute optimal transport distances between domains:

```bash
python compute_distances/ot_distance.py \
    --config configs/experiments/ot_distance_fmow.yaml
```

## Citation

If using FMoW, please cite the original dataset and WILDS benchmark:

```bibtex
@inproceedings{christie2018fmow,
  title={Functional Map of the World},
  author={Christie, Gordon and Fendley, Neil and Wilson, James and Mukherjee, Ryan},
  booktitle={CVPR},
  year={2018}
}

@article{koh2021wilds,
  title={WILDS: A Benchmark of in-the-Wild Distribution Shifts},
  author={Koh, Pang Wei and Sagawa, Shiori and Marklund, Henrik and others},
  journal={ICML},
  year={2021}
}
```

## Notes

- **Sequestered images** (`split == 'seq'`) are excluded following WILDS convention
- **Top-k country selection** ensures sufficient samples per domain
- **Temporal filtering** reduces distribution shift from different time periods
- Compatible with git/SatOT training and evaluation infrastructure

## Troubleshooting

**Issue:** `FileNotFoundError: metadata.csv not found`
- **Solution:** Ensure FMoW dataset is downloaded via WILDS: `python -c "from wilds import get_dataset; get_dataset('fmow', root_dir='./data/fmow', download=True)"`

**Issue:** `ImportError: No module named 'wilds'`
- **Solution:** Install WILDS: `pip install wilds>=2.0.0`

**Issue:** Different number of domains than expected
- **Solution:** Check `top_k_countries` parameter - default is 25

## See Also

- [GeoYFCCImage Dataset](../geoyfcc_image/README.md) - Image-based GeoYFCC dataset
- [WILDS Documentation](https://wilds.stanford.edu/)
- [git/SatOT Training Guide](../../docs/MAIN_UPDATES.md)
