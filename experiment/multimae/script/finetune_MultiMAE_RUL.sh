#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_finetune_common.sh"

epoch="100"
warmupepoch="10"
blr="1e-0"
batchsize="32"
layerdecay="0.5"
weightdecay="0.005"
droppath="0.0"
cyclegap="20"
mask_type="max_min_volt_temp"
numsnippet="0"
cycle_id=(40 60 80 100)
inference_cycle_id=(100)
downstream="RUL"
log_dir="${log_root}/RUL/${mask_type}_cyclegap${cyclegap}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"

for brandnum in 10 12 13; do
    run_folds "${downstream}" "${brandnum}" "${log_dir}" "no" \
        --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
        --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
        --same_normalizer --mask_type "${mask_type}" --num_snippet "${numsnippet}" \
        --cycle_id "${cycle_id[@]}" --inference_cycle_id "${inference_cycle_id[@]}" --cycle_gap "${cyclegap}"
done

finish_jobs "MultiMAE RUL"
