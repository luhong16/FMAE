#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-18}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiment/lipm/pretrained_model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [ "${DEVICE}" = "cuda" ] && [ "${NPROC_PER_NODE}" -gt 1 ]; then
  python -u -m torch.distributed.launch --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/pretrain_lipm_ev.py" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num_workers "${WORKERS}" \
    --seed "${SEED}"
else
  python -u "${SCRIPT_DIR}/pretrain_lipm_ev.py" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num_workers "${WORKERS}" \
    --seed "${SEED}"
fi
