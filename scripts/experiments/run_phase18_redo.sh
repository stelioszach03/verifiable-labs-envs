#!/usr/bin/env bash
# scripts/experiments/run_phase18_redo.sh
#
# Phase 18 redo — GRPO on Qwen-1.5B with calibrated rewards from the
# sparse-fourier-recovery environment. Fixes the weekend attempt that
# burned $25 before reward != 0 ever showed up.
#
# Usage:
#   bash run_phase18_redo.sh --dry-run
#   bash run_phase18_redo.sh
#   bash run_phase18_redo.sh --resume runs/phase18-redo_<ts>/last_checkpoint
#
# Estimated: ~7h on RTX 5090, ~$7. vLLM guided decoding ENABLED.
#
# Author: Stelios <sdi2200243@di.uoa.gr>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

# --- experiment identity --------------------------------------------------
export EXPERIMENT_ID="phase18-redo"
export EXPERIMENT_DESCRIPTION="Phase 18 redo — GRPO on Qwen-1.5B (sparse-fourier-recovery)"
export ENV_SLUG="sparse-fourier-recovery"
export DATASET_PATH="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_train.jsonl"
export MAX_STEPS=200
export LEARNING_RATE="1e-6"
export EXPERIMENT_NUM_GENERATIONS="${NUM_GENERATIONS}"  # 4 by default
export EXPERIMENT_TEMPERATURE="${TRAINING_TEMPERATURE}"
export MIN_VRAM_GB=24
export MIN_DISK_GB=20

# --- parse CLI ------------------------------------------------------------
parse_runner_args "$@"

# --- preflight ------------------------------------------------------------
preflight_or_die "${MIN_VRAM_GB}" "${MIN_DISK_GB}" "${DATASET_PATH}"

# --- output paths ---------------------------------------------------------
setup_run_paths

# Capture the resolved config so write_manifest can embed it.
export EXPERIMENT_CONFIG_JSON
EXPERIMENT_CONFIG_JSON=$(cat <<JSON
{
  "env": "${ENV_SLUG}",
  "base_model": "${VLABS_BASE_MODEL}",
  "dataset": "${DATASET_PATH}",
  "max_steps": ${MAX_STEPS},
  "learning_rate": ${LEARNING_RATE},
  "lora_r": ${LORA_R},
  "lora_alpha": ${LORA_ALPHA},
  "num_generations": ${EXPERIMENT_NUM_GENERATIONS},
  "max_prompt_length": ${MAX_PROMPT_LENGTH},
  "max_completion_length": ${MAX_COMPLETION_LENGTH},
  "training_temperature": ${EXPERIMENT_TEMPERATURE},
  "vllm_mode": "${VLLM_MODE}",
  "vllm_gpu_memory_utilization": ${VLLM_GPU_MEMORY_UTILIZATION},
  "resume_from": "${RESUME_FROM}"
}
JSON
)

log_info "experiment:  ${EXPERIMENT_ID}"
log_info "description: ${EXPERIMENT_DESCRIPTION}"
log_info "env:         ${ENV_SLUG}"
log_info "model:       ${VLABS_BASE_MODEL}"
log_info "max_steps:   ${MAX_STEPS}"
log_info "output:      ${OUTPUT_DIR}"
log_info "log:         ${LOG_FILE}"
log_info "dry_run:     ${DRY_RUN}"
[[ -n "${RESUME_FROM}" ]] && log_info "resume_from: ${RESUME_FROM}"

# --- build training command -----------------------------------------------
common_args=(
    --dataset "${DATASET_PATH}"
    --base-model "${VLABS_BASE_MODEL}"
    --output-dir "${OUTPUT_DIR}"
    --lr "${LEARNING_RATE}"
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --wandb-mode "${WANDB_MODE}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "DRY-RUN — calling vlabs-reward-train dry-run, no GPU training"
    vlabs-reward-train dry-run "${common_args[@]}"
    log_info "[${EXPERIMENT_ID}] dry-run OK — wiring valid, ready to fire"
    exit 0
fi

# Real run: register manifest writer on exit.
trap 'write_manifest $?' EXIT

mkdir -p "${OUTPUT_DIR}"
log_info "firing full GRPO training. logs -> ${LOG_FILE}"

vlabs-reward-train train "${common_args[@]}" 2>&1 | tee "${LOG_FILE}"
rc=${PIPESTATUS[0]}
log_info "[${EXPERIMENT_ID}] training exited with code ${rc}"
exit "${rc}"
