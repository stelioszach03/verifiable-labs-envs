#!/usr/bin/env bash
# scripts/experiments/run_E6.sh
#
# E6 — Teacher source ablation. Two sequential sub-runs:
#   (a) teacher_source=env     — closed-form ground truth only
#   (b) teacher_source=hybrid  — 90% env + 10% frontier judge
#
# Author: Stelios <sdi2200243@di.uoa.gr>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

export EXPERIMENT_ID="E6"
export EXPERIMENT_DESCRIPTION="E6 — Teacher source ablation (env vs hybrid)"
export ENV_DATASET="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_train_env.jsonl"
export HYBRID_DATASET="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_train_hybrid.jsonl"
export EVAL_SET_PATH="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_eval.jsonl"
export CALIB_SET_PATH="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_calib.jsonl"
export LEARNING_RATE="2e-4"
export EPOCHS=3
export BATCH_SIZE=16
export GRAD_ACCUM=4
export MIN_VRAM_GB=24
export MIN_DISK_GB=25

parse_runner_args "$@"
preflight_or_die "${MIN_VRAM_GB}" "${MIN_DISK_GB}" "${ENV_DATASET}"
setup_run_paths

export EXPERIMENT_CONFIG_JSON
EXPERIMENT_CONFIG_JSON=$(cat <<JSON
{
  "base_model": "${VLABS_BASE_MODEL}",
  "subruns": [
    {"teacher_source": "env",    "dataset": "${ENV_DATASET}"},
    {"teacher_source": "hybrid", "dataset": "${HYBRID_DATASET}"}
  ],
  "eval_set": "${EVAL_SET_PATH}",
  "calib_set": "${CALIB_SET_PATH}",
  "learning_rate": ${LEARNING_RATE},
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "grad_accum": ${GRAD_ACCUM},
  "lora_r": ${LORA_R},
  "lora_alpha": ${LORA_ALPHA},
  "resume_from": "${RESUME_FROM}"
}
JSON
)

log_info "experiment:  ${EXPERIMENT_ID} — ${EXPERIMENT_DESCRIPTION}"
log_info "output:      ${OUTPUT_DIR}"
log_info "dry_run:     ${DRY_RUN}"

run_subrun() {
    local label="$1"
    local dataset="$2"
    local subrun_dir="${OUTPUT_DIR}/${label}"
    local subrun_log="${LOG_FILE%.log}_${label}.log"
    mkdir -p "${subrun_dir}"
    log_info "subrun: ${label} (dataset=${dataset})"

    local args=(
        --dataset "${dataset}"
        --base-model "${VLABS_BASE_MODEL}"
        --output-dir "${subrun_dir}"
        --eval-set "${EVAL_SET_PATH}"
        --calib-set "${CALIB_SET_PATH}"
        --lr "${LEARNING_RATE}"
        --epochs "${EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --grad-accum "${GRAD_ACCUM}"
        --lora-r "${LORA_R}"
        --lora-alpha "${LORA_ALPHA}"
        --wandb-mode "${WANDB_MODE}"
    )

    if [[ "${DRY_RUN}" == "true" ]]; then
        vlabs-reward-train dry-run "${args[@]}"
        log_info "[${EXPERIMENT_ID}/${label}] dry-run OK"
        return 0
    fi

    vlabs-reward-train train "${args[@]}" 2>&1 | tee "${subrun_log}"
    return ${PIPESTATUS[0]}
}

if [[ "${DRY_RUN}" == "true" ]]; then
    run_subrun env "${ENV_DATASET}"
    run_subrun hybrid "${HYBRID_DATASET}"
    log_info "[${EXPERIMENT_ID}] dry-run OK (both subruns)"
    exit 0
fi

trap 'write_manifest $?' EXIT
mkdir -p "${OUTPUT_DIR}"
run_subrun env "${ENV_DATASET}"
run_subrun hybrid "${HYBRID_DATASET}"
log_info "[${EXPERIMENT_ID}] both subruns complete"
exit 0
