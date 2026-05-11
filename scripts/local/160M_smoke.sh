#!/bin/bash
# Smoke test for the ~160M custom transformer optimizer sweep.
#
# Assumes the FineWeb-Edu (100BT) dataset has been prepared via
# `data/datasets/prepare_finewebedu_100BT.sh`. Point TRAINSET_PATH /
# VALIDSET_PATH at the resulting `train/` and `valid/` directories
# (export them once, then invoke the script as below):
#
#   export TRAINSET_PATH=/path/to/.../ctx_2048/train
#   export VALIDSET_PATH=/path/to/.../ctx_2048/valid
#
# Usage:
#   bash scripts/local/160M_smoke.sh                        # default: 1 GPU (0)
#   CUDA_DEVICES=0,1,2,3 bash scripts/local/160M_smoke.sh   # match sweep (4 GPUs)
#   OPTIM=adamw LR=3e-4 bash scripts/local/160M_smoke.sh    # try a different optimizer
#   MICRO_BS=16 bash scripts/local/160M_smoke.sh            # halve micro-bs if OOM
#   CUDA_DEVICES=0,1 STEPS=20 bash scripts/local/160M_smoke.sh
set -euo pipefail

if [[ -z "${TRAINSET_PATH:-}" || -z "${VALIDSET_PATH:-}" ]]; then
    echo "ERROR: TRAINSET_PATH and VALIDSET_PATH must be set (point them at the" >&2
    echo "       train/ and valid/ directories produced by" >&2
    echo "       data/datasets/prepare_finewebedu_100BT.sh)." >&2
    exit 1
fi

DEVICES="${CUDA_DEVICES:-0}"
NPROC=$(awk -F, '{print NF}' <<< "${DEVICES}")

STEPS=${STEPS:-300}
MICRO_BS=${MICRO_BS:-32}
SEQ_LEN=${SEQ_LEN:-1024}
LR=${LR:-1e-3}
OPTIM=${OPTIM:-"muown"}

# Target effective batch: matches config/experiments/160M_optimizer_sweep.yaml
# (4 GPUs * micro_bs=32 * grad_accum=4 = 512 samples = 524K tokens/step at seq=1024).
# Auto-compute GRAD_ACCUM so MICRO_BS * GRAD_ACCUM * NPROC = TARGET_SAMPLES.
TARGET_SAMPLES=${TARGET_SAMPLES:-512}
PER_GPU=$(( TARGET_SAMPLES / NPROC ))
if (( TARGET_SAMPLES % NPROC != 0 )); then
    echo "ERROR: TARGET_SAMPLES=${TARGET_SAMPLES} not divisible by NPROC=${NPROC}" >&2
    exit 1
fi
if (( PER_GPU % MICRO_BS != 0 )); then
    echo "ERROR: per-GPU samples ${PER_GPU} (= ${TARGET_SAMPLES}/${NPROC}) not divisible by MICRO_BS=${MICRO_BS}" >&2
    exit 1
fi
GRAD_ACCUM=${GRAD_ACCUM:-$(( PER_GPU / MICRO_BS ))}
EFFECTIVE_BS=$(( MICRO_BS * GRAD_ACCUM * NPROC ))
TOKENS_PER_STEP=$(( EFFECTIVE_BS * SEQ_LEN ))

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_NAME="160M_smoke_$(date +%Y%m%d-%H%M%S)"
# Override OUTPUT_DIR (or its parent OUTPUT_ROOT) to point at a writable
# location on your machine; defaults to a repo-local `runs/` directory.
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/runs}"
OUTPUT_BASE="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"
mkdir -p "${OUTPUT_BASE}"

echo "========================================"
echo "160M optimizer-sweep smoke test"
echo "  Model:        transformer d_model=768 n_layers=12 n_heads=12 (GLU, RMSNorm, tied)"
echo "  Optimizer:    ${OPTIM}  lr=${LR}"
echo "  Steps:        ${STEPS}"
echo "  Seq / MBS:    ${SEQ_LEN} / ${MICRO_BS} per GPU"
echo "  GradAccum:    ${GRAD_ACCUM}  (auto)"
echo "  GPUs:         ${NPROC} (CUDA_VISIBLE_DEVICES=${DEVICES})"
echo "  Effective BS: ${EFFECTIVE_BS} samples = ${TOKENS_PER_STEP} tokens/step"
echo "  Output:       ${OUTPUT_BASE}"
echo "========================================"

cd "${REPO_DIR}"

RUN_DIR="${OUTPUT_BASE}/run"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${DEVICES}" \
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    train_hydra.py \
    --config-path 'config/experiments' \
    --config-name '160M_optimizer_sweep' \
    ++trainset_path="${TRAINSET_PATH}" \
    ++validset_path="${VALIDSET_PATH}" \
    ++optim="${OPTIM}" \
    ++lr="${LR}" \
    ++steps_budget="${STEPS}" \
    ++micro_batch_size="${MICRO_BS}" \
    ++grad_accumulation_steps="${GRAD_ACCUM}" \
    ++seq_len="${SEQ_LEN}" \
    ++use_wandb=False \
    ++eval=True \
    ++save_last_checkpoint=True \
    ++save_intermediate_checkpoints=False \
    ++eval_every_steps=100 \
    ++log_every_steps=100 \
    ++print_progress=True \
    ++exp_name="160M_smoke" \
    ++out_dir="${RUN_DIR}" \
    ++hydra.run.dir="${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/output.log"
