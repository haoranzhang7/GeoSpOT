## GeoSpOT --

This repository is for

- Domain-wise pretraining on datasets (GeoYFCC, GeoYFCC-Text, FMoW, GeoDE)
- Zero-shot (few-shot) evaluation across domains
- Computing OT (Sinkhorn) Distances

---

## 1. Environment

Set up a conda environment and install the Python dependencies:

```bash
conda create -n geospot python=3.10
conda activate geospot
pip install -r requirements.txt
```

---

## 2. Data configuration

By default, all configs assume your data lives under `./data` in the repo root.

### Data Download

1. Follow `datasets/geoyfcc/README.md` to download the GeoYFCC metadata and place it under:

```text
./data/geoyfcc/
```

2. The main configs that reference this directory are:

- `configs/experiments/pretrain_geoyfcc.yaml` (uses `DATA_DIR: ./data`)
- `configs/experiments/zeroshot_geoyfcc.yaml` (uses `DATA_DIR: ./data`)
- `configs/datasets/geoyfcc.yaml` (uses `PATHS.data_dir: ./data/geoyfcc`)

If you prefer a different location, update those YAML entries accordingly.

---

## 3. Running core experiments

- **Pretraining (single domain)**: Example pretraining

```bash
python src/training/pretrain_by_domain.py \
  --config configs/experiments/pretrain_geoyfcc.yaml \
  --pretrain_domain 5 \
  --model_seed 48329
```

- **Subset-selection pretraining**: Example configuration (for OT selection method, need to generate ot distances first)

```bash
python src/training/pretrain_by_domain_subset.py \
  --config configs/experiments/subset_selection.yaml
```

- **Zero-shot evaluation**:

```bash
python src/evaluation/zeroshot_test_eval.py \
  --config configs/experiments/zeroshot_geoyfcc.yaml \
  --pretrain_domain 5 \
  --target_domains 0 1 2 3 4
```
---

## 4. Embeddings and OT distances

- **GeoYFCC text embeddings (BERT)** (uses `configs/datasets/geoyfcc.yaml`; output dir can match `PATHS.embeddings_dir` there):

```bash
python src/data/embeddings/embeddings.py \
  --config configs/datasets/geoyfcc.yaml \
  --output_dir ./data/geoyfcc/embeddings/ \
  --split train
```

- **OT distances between domains**: Specify source domain and will compute OT distances to all domains (need to compute embeddings first)

```bash
python src/data/distances/ot_distance.py \
  --embedding-type bert \
  --source-domain-idx 5 \
  --reg-e 0.01 \
  --max-iter 1000 \
  --metric cosine \
  --method sinkhorn \
  --normalize-cost max_per_domain \
  --k 1
```

Adjust `--source-domain-idx`, `--embedding-type`, `--k`, and the OT hyperparameters as needed for your experiments.
