# Muown Optimizer

This repository contains the training code accompanying the paper on
**Muown**, a Muon-style optimizer that learns implicit per-parameter magnitudes.
It supports pretraining decoder-only Transformers on language modeling and
includes a small library of optimizer baselines (`adamw`, `muon_torch`,
`muon_bench`, `muown`, `muown_dp`, `normuon`, `lion`, `soap`).

## Installation

```bash
conda create --name muown python=3.12 -y && conda activate muown
# install the appropriate torch build for your CUDA version first, then:
pip install -r requirements.txt
```

## Data

We provide a script for downloading, tokenizing, chunking, and saving Hugging
Face datasets: `data/datasets/prepare.py`. Any HF dataset and tokenizer is
supported, with optional streaming so the full corpus is never materialized.

For the experiments in the paper we use **FineWeb-Edu (100BT sample)** tokenized
with `EleutherAI/gpt-neox-20b` at sequence length 2048. The reference command
is in `data/datasets/prepare_finewebedu_100BT.sh`; minimally:

```bash
PYTHONPATH=. python data/datasets/prepare.py \
    --out_path=<DATA_DIR>/fwedu_sample_100B_tokenizer_GPTNeoX \
    --cache_path=<CACHE_DIR> \
    --download --tokenize --chunk \
    --save_tokenized --save_tokenizer \
    --dataset_path="HuggingFaceFW/fineweb-edu" \
    --dataset_split="train" \
    --dataset_name="sample-100BT" \
    --tokenizer="EleutherAI/gpt-neox-20b" \
    --seq_length=2048 \
    --split_train_valid=True \
    --n_tokens_valid=10000000
```

This produces `train/` and `valid/` subdirectories of pre-chunked token shards
that are passed to the trainer as `trainset_path` / `validset_path`.

> **Qwen experiments.** The Qwen2-0.5B runs
> (`config/experiments/qwen.yaml`) instead use the **FineWeb-Edu sample-10BT**
> corpus tokenized with the matching `Qwen/Qwen2-0.5B` tokenizer at sequence
> length 2048. Re-run `data/datasets/prepare.py` with
> `--dataset_name="sample-10BT"` and `--tokenizer="Qwen/Qwen2-0.5B"` to
> regenerate this dataset, then point `trainset_path` / `validset_path` at the
> resulting `train/` / `valid/` directories.

## Usage

The main training entry point is `train_hydra.py`. It is a [Hydra](https://hydra.cc/)
application: pick a YAML config under `config/experiments/` with `--config-name`
and override any field with `++key=value`.

#### Single GPU

```bash
python train_hydra.py \
    --config-path 'config/experiments' \
    --config-name '160M_optimizer_sweep'
```

#### Multiple GPUs (single node, DDP)

```bash
torchrun --standalone --nproc_per_node=4 \
    train_hydra.py \
    --config-path 'config/experiments' \
    --config-name '160M_optimizer_sweep'
```

## Reproducing the 160M runs

The 160M-class optimizer sweep from the paper is driven by
[`config/experiments/160M_optimizer_sweep.yaml`](config/experiments/160M_optimizer_sweep.yaml).
It defines a 12-layer, 768-dim, 12-head GLU-MLP Transformer with tied
embeddings and RMSNorm (~125M non-embedding parameters; "160M class" in
common nomenclature), trained for 5B tokens (`steps_budget=9537`) with
WSD scheduling (2% warmup, 20% cooldown), `bfloat16`, and an
effective batch of 524K tokens/step.

A baseline run with the default optimizer (Muown, `lr=1e-3`) on 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 \
    train_hydra.py \
    --config-path 'config/experiments' \
    --config-name '160M_optimizer_sweep' \
    ++trainset_path=<DATA_DIR>/train \
    ++validset_path=<DATA_DIR>/valid \
    ++optim=muown \
    ++lr=1e-3 \
    ++out_dir=<OUT_DIR>/muown_lr1e-3 \
    ++hydra.run.dir=<OUT_DIR>/muown_lr1e-3 \
```

Replace `<DATA_DIR>` with the path produced by the data prep step, and
`<OUT_DIR>` with any writable directory.



#### Smaller-scale smoke test

A short end-to-end check on the same config is provided at
`scripts/local/160M_smoke.sh`. It auto-computes `grad_accumulation_steps` so
the *effective* batch size matches the full sweep (524K tokens/step) regardless
of how many GPUs you point it at, and it accepts `OPTIM`, `LR`, `STEPS`,
`MICRO_BS`, `SEQ_LEN`, and `CUDA_DEVICES` as environment-variable overrides.

The script assumes the FineWeb-Edu (100BT) dataset has already been prepared
via `data/datasets/prepare_finewebedu_100BT.sh`. Export `TRAINSET_PATH` and
`VALIDSET_PATH` once (pointing at the resulting `train/` and `valid/`
directories) before invoking:

```bash
export TRAINSET_PATH=<DATA_DIR>/train
export VALIDSET_PATH=<DATA_DIR>/valid
```

```bash
# Default: Muown at lr=1e-3, 300 steps on a single GPU (CUDA device 0).
bash scripts/local/160M_smoke.sh

CUDA_DEVICES=1,2 OPTIM=muown LR=1e-3 STEPS=300 \
    bash scripts/local/160M_smoke.sh

```

The script writes its output directory under `<OUT_DIR>/run`. By default
`<OUT_DIR>` is a repo-local `runs/<timestamp>/` directory; override
`OUTPUT_ROOT` (parent dir) or `OUTPUT_DIR` (full path) to point it at a
different writable location on your machine.

## Repository layout

```
.
├── config/experiments/  # YAML configs for all paper experiments
├── data/                # Data prep + dataloaders + stateful samplers
│   └── datasets/        # Tokenization / chunking scripts
├── engine/              # Training engine (forward/backward/step, eval)
├── models/              # Decoder-only Transformer + parameter grouping
├── optim/               # Muown and baseline optimizers + LR schedulers
│   ├── muown.py         # Muown (this paper) and MuownDP (DP variant)
│   ├── muon.py          # Reference Muon used for runtime benchmarks
│   ├── lion.py, soap.py # Additional baselines
│   ├── lr_schedule.py   # WSD, cosine, linear-cooldown schedulers
│   └── init_optim.py    # Optimizer factory dispatched by `cfg.optim`
├── scripts/             # Launch scripts (smoke tests + cluster scripts)
├── checkpoint_utils.py  # Save / resume utilities
├── torch_utils.py       # DDP, seeding, TF32 setup
├── logging_utils.py     # Step-time / memory / weight-norm logging
├── train_hydra.py       # Main training entry point
├── modded_nanogpt_2024-10-10.py  # Standalone reference baseline:
│                        # modified modded-nanogpt Muon GPT-2 training
│                        # script (FineWeb / nanoGPT data layout), kept 
│                        # mostly verbatim. 
└── utils.py             # Misc helpers (wandb init, logging, ...)
```

## Credits

The training scaffolding builds on two openly available codebases, which we
gratefully acknowledge:

- [plainLM](https://github.com/Niccolo-Ajroldi/plainLM) 
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/993c1bd071b98909ed27ea3bc5f4da3b932b8c41/records/track_1_short/2024-10-10_Muon/train_gpt2.py)

