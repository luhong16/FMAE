#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

dirname="pretrained_model/mae_pretrainmae_vit_half_patch16_mpr0.5_mcr0.4_msn0_lrNone_blr0.00015_minlr0.0_weightdecay0.05_warmupepoch40_numsnippet5_pos_embed_dim12_d_pos_embed_dim8_d_typecombine_d_pad_typesoc_current_mileage_embed_epochs800_s0"
ckpt="799"
modelname="vit_half_patch16"
epoch="20"
warmupepoch="4"
blr="5e-2"
seed="5"
batchsize="32"
layerdecay="0.65"
weightdecay="0.01"
droppath="0.1"
datapercent="5"
offset="${GPU_OFFSET:-0}"
pos_embed_dim="12"
mask_type="max_min_volt_temp"
downstream="capacity"
noise_tag="gaussian_noise_std0.005"

finetune_ckpt="./${dirname}/checkpoint-${ckpt}.pth"
if [[ ! -f "${finetune_ckpt}" ]]; then
    echo "Checkpoint not found: ${finetune_ckpt}" >&2
    exit 1
fi

htail="${mask_type}_${noise_tag}"
log_dir="${SCRIPT_DIR}/logs/capacity/${htail}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"
mkdir -p "${log_dir}"

common_args=(
    --finetune "${finetune_ckpt}"
    --batch_size "${batchsize}"
    --model "${modelname}"
    --epochs "${epoch}"
    --warmup_epochs "${warmupepoch}"
    --blr "${blr}"
    --layer_decay "${layerdecay}"
    --weight_decay "${weightdecay}"
    --drop_path "${droppath}"
    --same_normalizer
    --mask_type "${mask_type}"
    --apply_noise
    --pos_embed_dim "${pos_embed_dim}"
    --seed "${seed}"
    --downstream "${downstream}"
    --data_percent "${datapercent}"
    --task batterybrandmileage
)

for brandnum in 10 11 12 13; do
    brand_offset=$((brandnum - 10))
    for fold in 0 1 2 3 4; do
        gpu=$((((brand_offset + fold) % 4) + offset))
        run_name="f${fold}_b${brandnum}"
        CUDA_VISIBLE_DEVICES="${gpu}" nohup python -u main_finetune.py \
            "${common_args[@]}" \
            --fold_num "${fold}" \
            --brand_num "${brandnum}" \
            --output_dir "${log_dir}/${run_name}" \
            > "${log_dir}/${run_name}.txt" 2>&1 &
        echo "Launched ${run_name} on GPU ${gpu}; log: ${log_dir}/${run_name}.txt"
    done
done
