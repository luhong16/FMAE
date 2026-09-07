#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

pretrain_name="mae_pretrainmae_vit_half_patch16_mpr0.8_mcr0.8_msn0_lrNone_blr0.00015_minlr0.0_weightdecay0.05_warmupepoch40_numsnippet5_pos_embed_dim12_d_pos_embed_dim8_d_typecombine_d_pad_typesoc_current_mileage_embed_epochs800_s0"
pretrain_dir="${SCRIPT_DIR}/pretrained_model/${pretrain_name}"
ckpt="799"
modelname="vit_half_patch16"
seed="5"
pos_embed_dim="12"
mask_tag="mpr0.8_mcr0.8"
log_root="${SCRIPT_DIR}/logs/${mask_tag}"
max_jobs="${MAX_JOBS:-8}"
gpu_list="${GPU_LIST:-0 1 2 3}"
dry_run="${DRY_RUN:-0}"

read -r -a gpus <<< "${gpu_list}"
if [ "${#gpus[@]}" -eq 0 ]; then
    echo "GPU_LIST is empty" >&2
    exit 1
fi

if [ "${max_jobs}" -lt 1 ]; then
    echo "MAX_JOBS must be at least 1" >&2
    exit 1
fi

finetune_ckpt="${pretrain_dir}/checkpoint-${ckpt}.pth"
if [ ! -f "${finetune_ckpt}" ]; then
    echo "Missing checkpoint: ${finetune_ckpt}" >&2
    exit 1
fi

mkdir -p "${log_root}"
: > "${log_root}/launch.log"

running=0
job_idx=0
failed=0

reap_one() {
    if ! wait -n; then
        failed=$((failed + 1))
    fi
    running=$((running - 1))
}

wait_for_slot() {
    while [ "${running}" -ge "${max_jobs}" ]; do
        reap_one
    done
}

launch_job() {
    local gpu="$1"
    local log_file="$2"
    shift 2

    mkdir -p "$(dirname "${log_file}")"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] GPU ${gpu}: $*" >> "${log_root}/launch.log"
    if [ "${dry_run}" = "1" ]; then
        job_idx=$((job_idx + 1))
        return
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" python -u main_finetune.py "$@" > "${log_file}" 2>&1 &
    running=$((running + 1))
    job_idx=$((job_idx + 1))
}

run_finetune() {
    local log_file="$1"
    shift

    wait_for_slot
    local gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    launch_job "${gpu}" "${log_file}" \
        --finetune "${finetune_ckpt}" \
        --batch_size "${batchsize}" \
        --model "${modelname}" \
        --pos_embed_dim "${pos_embed_dim}" \
        --seed "${seed}" \
        --task batterybrandmileage \
        "$@"
}

run_folds() {
    local downstream="$1"
    local brandnum="$2"
    local log_dir="$3"
    local has_output_dir="$4"
    shift 4

    for fold in 0 1 2 3 4; do
        local run_name="f${fold}_b${brandnum}"
        if [ "${has_output_dir}" = "yes" ]; then
            run_finetune "${log_dir}/${run_name}.txt" \
                "$@" \
                --fold_num "${fold}" \
                --brand_num "${brandnum}" \
                --downstream "${downstream}" \
                --output_dir "${log_dir}/${run_name}"
        else
            run_finetune "${log_dir}/${run_name}.txt" \
                "$@" \
                --fold_num "${fold}" \
                --brand_num "${brandnum}" \
                --downstream "${downstream}"
        fi
    done
}

batchsize="32"
epoch="10"
warmupepoch="2"
blr="5e-3"
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

batchsize="32"
epoch="20"
warmupepoch="4"
blr="5e-2"
layerdecay="0.5"
weightdecay="0.005"
droppath="0.0"
datapercent="5"
mask_type="no"
downstream="capacity"
log_dir="${log_root}/capacity/${mask_type}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"
for brandnum in 1 2 3 4 5 6; do
    run_folds "${downstream}" "${brandnum}" "${log_dir}" "yes" \
        --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
        --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
        --same_normalizer --mask_type "${mask_type}" --data_percent "${datapercent}"
done

batchsize="32"
epoch="20"
warmupepoch="4"
blr="5e-1"
layerdecay="0.5"
weightdecay="0.03"
droppath="0.1"
datapercent="5"
mask_type="no"
downstream="capacity"
log_dir="${log_root}/capacity_storage/${mask_type}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"
run_folds "${downstream}" "7" "${log_dir}" "yes" \
    --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
    --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
    --same_normalizer --mask_type "${mask_type}" --data_percent "${datapercent}"

batchsize="32"
epoch="20"
warmupepoch="4"
blr="5e-2"
layerdecay="0.65"
weightdecay="0.01"
droppath="0.1"
datapercent="5"
mask_type="max_min_volt_temp"
downstream="capacity"
log_dir="${log_root}/capacity_lab/${mask_type}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"
for brandnum in 10 11 12 13; do
    run_folds "${downstream}" "${brandnum}" "${log_dir}" "yes" \
        --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
        --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
        --same_normalizer --mask_type "${mask_type}" --data_percent "${datapercent}"
done

batchsize="32"
epoch="100"
warmupepoch="10"
blr="1e-2"
layerdecay="0.5"
weightdecay="0.005"
droppath="0.0"
cyclegap="20"
mask_type="max_min_volt_temp"
numsnippet="2"
downstream="RUL"
log_dir="${log_root}/RUL/${mask_type}_cyclegap${cyclegap}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"
for brandnum in 10 12 13; do
    run_folds "${downstream}" "${brandnum}" "${log_dir}" "no" \
        --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
        --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
        --same_normalizer --mask_type "${mask_type}" --num_snippet "${numsnippet}" \
        --cycle_id 40 60 80 100 --inference_cycle_id 100 --cycle_gap "${cyclegap}"
done

batchsize="32"
epoch="20"
warmupepoch="4"
blr="5e-3"
layerdecay="0.8"
weightdecay="0.05"
droppath="0.1"
datapercent="20"
mask_type="max_min_volt_temp"
downstream="IR"
log_dir="${log_root}/IR/${mask_type}/${ckpt}_pretrain_${modelname}/epoch${epoch}_warmupepoch${warmupepoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"
run_folds "${downstream}" "10" "${log_dir}" "no" \
    --epochs "${epoch}" --warmup_epochs "${warmupepoch}" --blr "${blr}" \
    --layer_decay "${layerdecay}" --weight_decay "${weightdecay}" --drop_path "${droppath}" \
    --same_normalizer --mask_type "${mask_type}" --data_percent "${datapercent}"

while [ "${running}" -gt 0 ]; do
    reap_one
done

if [ "${failed}" -eq 0 ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] all mask-ratio jobs finished" >> "${log_root}/launch.log"
else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] mask-ratio jobs finished with ${failed} failed jobs" >> "${log_root}/launch.log"
    exit 1
fi
