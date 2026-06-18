#!/bin/bash
set -e

DATA_ROOT="./data"
OUTPUT_DIR="./data/embeddings"

for split in train val test; do
    # BERT embeddings
    python src/data/embeddings/embeddings.py \
        --data_root $DATA_ROOT \
        --split $split \
        --model_type bert_singlelabel \
        --output_dir $OUTPUT_DIR

    # GeoCLIP embeddings
    python src/data/embeddings/embeddings.py \
        --data_root $DATA_ROOT \
        --split $split \
        --geoclip \
        --output_dir $OUTPUT_DIR

    # SatCLIP embeddings (L10)
    python src/data/embeddings/embeddings.py \
        --data_root $DATA_ROOT \
        --split $split \
        --satclip --legendre_polys 10 \
        --output_dir $OUTPUT_DIR

    # SatCLIP embeddings (L40)
    python src/data/embeddings/embeddings.py \
        --data_root $DATA_ROOT \
        --split $split \
        --satclip --legendre_polys 40 \
        --output_dir $OUTPUT_DIR
done
