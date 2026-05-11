#!/bin/bash

# This script will download and preprocess FineWebEdu-10BT.
# Expect some token loss by batched concat_chunk.


mkdir -p "${SCRATCH}/tmp"

PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="${SCRATCH}/fwedu/fwedu_sample_10B_tokenizer_Qwen2" \
  --cache_path="${SCRATCH}/tmp" \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-10BT" \
  --tokenizer="Qwen/Qwen2-0.5B" \
  --seq_length=2048 \
  --split_train_valid=True \
  --n_tokens_valid=10000000
