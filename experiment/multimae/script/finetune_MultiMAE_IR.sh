#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_finetune_common.sh"

epoch="20"
warmupepoch="4"
blr="5e-1"
batchsize="32"
layerdecay="0.8"
weightdecay="0.01"
droppath="0.0"
datapercent="20"
mask_type="max_min_volt_temp"
downstream="IR"
log_dir="${log_root}/IR/${mask_type}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"

run_folds "${downstream}" "10" "${log_dir}" "no" \
    --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
    --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
    --same_normalizer --mask_type "${mask_type}" --data_percent "${datapercent}"

finish_jobs "MultiMAE IR"
