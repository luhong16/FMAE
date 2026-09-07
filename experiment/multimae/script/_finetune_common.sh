#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CODE_DIR="${EXPERIMENT_DIR}/code"

pretrain_name="evMultiMAE_pretrainpretrain_multimae_half_mpr0.5_lrNone_blr0.00015_minlr0.0_weightdecay0.05_warmupepoch40_numsnippet0_epochs800_s0"
pretrain_dir="${EXPERIMENT_DIR}/pretrained_model/${pretrain_name}"
ckpt="799"
modelname="multivit_half"
seed="5"
pos_embed_dim="12"
log_root="${EXPERIMENT_DIR}/logs"
run_name="$(basename "${0}" .sh)"
launch_log="${log_root}/launch/${run_name}.log"
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

if [ ! -f "${CODE_DIR}/main_finetune.py" ]; then
    echo "Missing MultiMAE code: ${CODE_DIR}/main_finetune.py" >&2
    exit 1
fi

mkdir -p "$(dirname "${launch_log}")"
: > "${launch_log}"
cd "${REPO_ROOT}"

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
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] GPU ${gpu}: $*" >> "${launch_log}"
    if [ "${dry_run}" = "1" ]; then
        job_idx=$((job_idx + 1))
        return
    fi

    PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python -u "${CODE_DIR}/main_finetune.py" "$@" > "${log_file}" 2>&1 &
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
        local run_id="f${fold}_b${brandnum}"
        if [ "${has_output_dir}" = "yes" ]; then
            run_finetune "${log_dir}/${run_id}.txt" \
                "$@" \
                --fold_num "${fold}" \
                --brand_num "${brandnum}" \
                --downstream "${downstream}" \
                --output_dir "${log_dir}/${run_id}"
        else
            run_finetune "${log_dir}/${run_id}.txt" \
                "$@" \
                --fold_num "${fold}" \
                --brand_num "${brandnum}" \
                --downstream "${downstream}"
        fi
    done
}

finish_jobs() {
    local label="$1"

    while [ "${running}" -gt 0 ]; do
        reap_one
    done

    if [ "${failed}" -eq 0 ]; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] all ${label} jobs finished" >> "${launch_log}"
    else
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ${label} finished with ${failed} failed jobs" >> "${launch_log}"
        exit 1
    fi
}
