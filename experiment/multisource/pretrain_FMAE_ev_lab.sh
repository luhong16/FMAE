#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-800}"
MODEL="${MODEL:-mae_vit_half_patch16}"
EV_DICT_DIR="${EV_DICT_DIR:-five_fold_utils/five_fold_utils_six_brand_all}"
LAB_DICT_DIR="${LAB_DICT_DIR:-five_fold_utils/five_fold_utils_lab_capacity}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/pretrained_model/}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SCRIPT_DIR}/pretrain_logs/tensorboard}"
OUTPUT_ROOT="${OUTPUT_ROOT%/}/"

python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main_pretrain.py \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --model "${MODEL}" \
  --mask_patch_ratio 0.5 \
  --mask_channel_ratio 0.4 \
  --mask_snippet_num 0 \
  --num_snippet 5 \
  --pos_embed_dim 12 \
  --decoder_pos_embed_dim 8 \
  --decoder_type combine \
  --decoder_pad_type soc_current_mileage_embed \
  --blr 0.00015 \
  --weight_decay 0.05 \
  --warmup_epochs 40 \
  --same_normalizer \
  --task batterybrandmileage \
  --pretrain_data ev_lab \
  --pretrain_ev_car_dict_dir "${EV_DICT_DIR}" \
  --pretrain_lab_car_dict_dir "${LAB_DICT_DIR}" \
  --output_dir "${OUTPUT_ROOT}" \
  --log_dir "${TENSORBOARD_DIR}" \
  "$@"
