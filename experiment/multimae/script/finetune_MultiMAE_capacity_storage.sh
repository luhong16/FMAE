#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_finetune_common.sh"

epoch="20"
warmupepoch="4"
blr="8e-0"
batchsize="32"
layerdecay="0.8"
weightdecay="0.01"
droppath="0.0"
datapercent="5"
mask_type="no"
downstream="capacity"
log_dir="${log_root}/capacity/${mask_type}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"

run_folds "${downstream}" "7" "${log_dir}" "yes" \
    --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
    --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
    --same_normalizer --mask_type "${mask_type}" --data_percent "${datapercent}"

finish_jobs "MultiMAE storage capacity"
