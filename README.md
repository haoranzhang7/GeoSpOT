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

Pass `--data_dir`/`--data-root` (and `--dataset`/`--dataset-name`) to point at a different location. Run any script with `--help` to see its full flag list.

### Data Download

1. Follow `datasets/geoyfcc/README.md` to download the GeoYFCC metadata and place it under:

```text
./data/geoyfcc/
```

---

## 3. Running core experiments

- **Pretraining (single domain)**: Example pretraining

```bash
python src/training/pretrain_by_domain.py \
  --pretrain_domain 5 \
  --model_seed 48329 \
  --data_dir ./data
```

- **Subset-selection pretraining**: Example using OT-based domain selection (for the OT selection method, you need to generate OT distances first)

```bash
python src/training/pretrain_by_domain_subset.py \
  --model_data_seed 48329 \
  --num_domains 5 \
  --domain_selection_method ot \
  --tgt_domain 5 \
  --ot_embedding_type bert \
  --data_dir ./data
```

- **Zero-shot evaluation**:

```bash
python src/evaluation/zeroshot_test_eval.py \
  --pretrain_domain 5 \
  --target_domains 0 1 2 3 4 \
  --data_dir ./data
```
---

## 4. Embeddings and OT distances

- **GeoYFCC text embeddings (BERT)**:

```bash
python src/data/embeddings/embeddings.py \
  --output_dir ./data/geoyfcc/embeddings/ \
  --split train
```

- **OT distances between domains**: Specify source domain and will compute OT distances to all domains (need to compute embeddings first)

```bash
python src/distances/ot_distance.py \
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
