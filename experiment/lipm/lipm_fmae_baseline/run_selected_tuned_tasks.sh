#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash experiment/lipm/lipm_fmae_baseline/run_selected_tuned_tasks.sh /path/to/lipm_backbone_checkpoint.pth [output_root]" >&2
  exit 2
fi

PRETRAINED_BACKBONE="$1"
if [ ! -f "${PRETRAINED_BACKBONE}" ]; then
  echo "Checkpoint not found: ${PRETRAINED_BACKBONE}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="${SCRIPT_DIR}/main_finetune_lipm.py"
OUT_ROOT="${2:-${ROOT_DIR}/logs/lipm_tuned/selected_v1}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-5}"
WORKERS="${WORKERS:-8}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
LIPM_SETTING="${LIPM_SETTING:-3}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_GPUS="${#GPUS[@]}"

running=0
next_gpu=0

wait_for_batch() {
  if [ "${running}" -gt 0 ]; then
    wait
    running=0
  fi
}

launch_one() {
  local downstream="$1"
  local brand="$2"
  local fold="$3"
  local epochs="$4"
  local lr="$5"
  local extra="$6"
  local gpu="${GPUS[${next_gpu}]}"
  next_gpu=$(( (next_gpu + 1) % NUM_GPUS ))

  local out_dir="${OUT_ROOT}/${downstream}/b${brand}/f${fold}"
  mkdir -p "${out_dir}"

  echo "[launch] $(date '+%F %T') gpu=${gpu} downstream=${downstream} brand=${brand} fold=${fold} epochs=${epochs} lr=${lr}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u "${PY}" \
    --pretrained_backbone "${PRETRAINED_BACKBONE}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${lr}" \
    --num_workers "${WORKERS}" \
    --seed "${SEED}" \
    --same_normalizer \
    --downstream "${downstream}" \
    --brand_num "${brand}" \
    --fold_num "${fold}" \
    --head_hidden_dim "${HEAD_HIDDEN_DIM}" \
    --lipm_setting "${LIPM_SETTING}" \
    --output_dir "${out_dir}" \
    --epochs "${epochs}" \
    ${extra} > "${out_dir}.txt" 2>&1 &

  running=$((running + 1))
  if [ "${running}" -ge "${NUM_GPUS}" ]; then
    wait_for_batch
  fi
}

mkdir -p "${OUT_ROOT}"

# Capacity on lab datasets: representative tuning improved MIT1 from 1.647 to
# 1.429 with no dropout/weight decay while keeping raw voltage/current inputs.
for brand in 10 11 12 13; do
  for fold in 0 1 2 3 4; do
    launch_one capacity "${brand}" "${fold}" 30 1e-4 "--no_normalize_vc --dropout 0 --weight_decay 0 --data_percent 5"
  done
done

# RUL: match FMAE's multi-snippet setup and use the best representative LR.
for brand in 10 12 13; do
  for fold in 0 1 2 3 4; do
    launch_one RUL "${brand}" "${fold}" 120 3e-5 "--no_normalize_vc --num_snippet 2 --cycle_gap 20 --cycle_id 40 60 80 100 --inference_cycle_id 100"
  done
done

wait_for_batch

echo "Selected LiPM tuned downstream tasks completed. Logs are under ${OUT_ROOT}."
