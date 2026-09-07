#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash experiment/lipm/lipm_fmae_baseline/run_all_tasks.sh /path/to/lipm_backbone_checkpoint.pth" >&2
  exit 2
fi

PRETRAINED_BACKBONE="$1"
if [ ! -f "${PRETRAINED_BACKBONE}" ]; then
  echo "Checkpoint not found: ${PRETRAINED_BACKBONE}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${SCRIPT_DIR}/main_finetune_lipm.py"

DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-5}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-8}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
LIPM_SETTING="${LIPM_SETTING:-3}"

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
  local gpu="${next_gpu}"
  next_gpu=$(( (next_gpu + 1) % NUM_GPUS ))

  local out_dir="${ROOT_DIR}/logs/lipm/${downstream}/b${brand}/f${fold}"
  mkdir -p "${out_dir}"

  echo "[launch] gpu=${gpu} downstream=${downstream} brand=${brand} fold=${fold} epochs=${epochs} lr=${lr}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u "${PY}" \
    --pretrained_backbone "${PRETRAINED_BACKBONE}" \
    --device "${DEVICE}" \
    --epochs "${epochs}" \
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
    ${extra} > "${out_dir}.txt" 2>&1 &

  running=$((running + 1))
  if [ "${running}" -ge "${NUM_GPUS}" ]; then
    wait_for_batch
  fi
}

# Capacity: EV brands 1-6.
for brand in 1 2 3 4 5 6; do
  for fold in 0 1 2 3 4; do
    launch_one capacity "${brand}" "${fold}" 20 "${LR}" "--data_percent 5"
  done
done

# Capacity: storage brand 7.
for fold in 0 1 2 3 4; do
  launch_one capacity 7 "${fold}" 20 "${LR}" "--data_percent 5"
done

# Capacity: lab brands 10-13. Tuned setting from run_selected_tuned_tasks.sh.
for brand in 10 11 12 13; do
  for fold in 0 1 2 3 4; do
    launch_one capacity "${brand}" "${fold}" 30 1e-4 "--no_normalize_vc --dropout 0 --weight_decay 0 --data_percent 5"
  done
done

# IR: lab brand 10.
for fold in 0 1 2 3 4; do
  launch_one IR 10 "${fold}" 20 "${LR}" "--data_percent 20"
done

# RUL: lab brands 10, 12, 13. Tuned setting from run_selected_tuned_tasks.sh.
for brand in 10 12 13; do
  for fold in 0 1 2 3 4; do
    launch_one RUL "${brand}" "${fold}" 120 3e-5 "--no_normalize_vc --num_snippet 2 --cycle_gap 20 --cycle_id 40 60 80 100 --inference_cycle_id 100"
  done
done

wait_for_batch

echo "LiPM downstream fine-tuning completed. Logs are under logs/lipm."
