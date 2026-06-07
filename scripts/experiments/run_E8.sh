#!/usr/bin/env bash
# scripts/experiments/run_E8.sh
#
# E8 — Final production benchmark. Uses the best config picked from
# E1..E7's manifests. The aggregator script (scripts/training/
# pick_best_config.py — future commit) reads runs/E*/manifest.json and
# emits the chosen knobs. For now we default to E4's recipe + a knob
# override mechanism via VLABS_BEST_* env vars.
#
# Author: Stelios <sdi2200243@di.uoa.gr>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

export EXPERIMENT_ID="E8"
export EXPERIMENT_DESCRIPTION="E8 — Final benchmark (best config from E1-E7)"
export DATASET_PATH="${VLABS_BEST_DATASET:-${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_train.jsonl}"
export EVAL_SET_PATH="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_eval.jsonl"
export CALIB_SET_PATH="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_calib.jsonl"
export BEST_MODEL="${VLABS_BEST_MODEL:-${VLABS_BASE_MODEL}}"
export LEARNING_RATE="${VLABS_BEST_LR:-2e-4}"
export EPOCHS="${VLABS_BEST_EPOCHS:-3}"
export BATCH_SIZE="${VLABS_BEST_BATCH_SIZE:-16}"
export GRAD_ACCUM="${VLABS_BEST_GRAD_ACCUM:-4}"
export MIN_VRAM_GB=24
export MIN_DISK_GB=20

parse_runner_args "$@"
preflight_or_die "${MIN_VRAM_GB}" "${MIN_DISK_GB}" "${DATASET_PATH}"
setup_run_paths

export EXPERIMENT_CONFIG_JSON
EXPERIMENT_CONFIG_JSON=$(cat <<JSON
{
  "base_model": "${BEST_MODEL}",
  "dataset": "${DATASET_PATH}",
  "eval_set": "${EVAL_SET_PATH}",
  "calib_set": "${CALIB_SET_PATH}",
  "learning_rate": ${LEARNING_RATE},
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "grad_accum": ${GRAD_ACCUM},
  "lora_r": ${LORA_R},
  "lora_alpha": ${LORA_ALPHA},
  "source": "best of E1..E7",
  "resume_from": "${RESUME_FROM}"
}
JSON
)

log_info "experiment:  ${EXPERIMENT_ID} — ${EXPERIMENT_DESCRIPTION}"
log_info "best_model:  ${BEST_MODEL}"
log_info "output:      ${OUTPUT_DIR}"
log_info "dry_run:     ${DRY_RUN}"

common_args=(
    --dataset "${DATASET_PATH}"
    --base-model "${BEST_MODEL}"
    --output-dir "${OUTPUT_DIR}"
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
    vlabs-reward-train dry-run "${common_args[@]}"
    log_info "[${EXPERIMENT_ID}] dry-run OK"
    exit 0
fi

trap 'write_manifest $?' EXIT
mkdir -p "${OUTPUT_DIR}"
vlabs-reward-train train "${common_args[@]}" 2>&1 | tee "${LOG_FILE}"
exit ${PIPESTATUS[0]}
