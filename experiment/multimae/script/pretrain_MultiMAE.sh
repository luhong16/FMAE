#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CODE_DIR="${EXPERIMENT_DIR}/code"

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-800}"
MODEL="${MODEL:-pretrain_multimae_half}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${EXPERIMENT_DIR}/pretrained_model/ev}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${EXPERIMENT_DIR}/pretrain_logs/tensorboard}"
USE_WANDB="${USE_WANDB:-1}"

OUTPUT_PREFIX="${OUTPUT_PREFIX%/}"
wandb_args=()
if [ "${USE_WANDB}" = "1" ]; then
    wandb_args+=(--use-wandb)
fi

python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${CODE_DIR}/main_pretrain.py" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --model "${MODEL}" \
  --mask_patch_ratio 0.5 \
  --mask_channel_ratio 0 \
  --mask_snippet_num 0 \
  --num_snippet 0 \
  --blr 0.00015 \
  --min_lr 0.0 \
  --weight_decay 0.05 \
  --warmup_epochs 40 \
  --same_normalizer \
  --task batterybrandmileage \
  --output_dir "${OUTPUT_PREFIX}" \
  --log_dir "${TENSORBOARD_DIR}" \
  "${wandb_args[@]}" \
  "$@"
