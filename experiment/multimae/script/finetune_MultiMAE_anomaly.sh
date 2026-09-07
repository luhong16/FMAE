#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_finetune_common.sh"

epoch="10"
warmupepoch="2"
blr="5e-3"
batchsize="32"
layerdecay="0.65"
weightdecay="0.01"
droppath="0.1"
datapercent="20"
mask_type="no"
downstream="anomaly"
log_dir="${log_root}/anomaly/${mask_type}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"

for brandnum in 1 2 4 5 6; do
    run_folds "${downstream}" "${brandnum}" "${log_dir}" "no" \
        --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
        --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
        --same_normalizer --h 100 --mask_type "${mask_type}" --data_percent "${datapercent}"
done

finish_jobs "MultiMAE anomaly"
