#!/usr/bin/env bash
# scripts/experiments/run_E2.sh
#
# E2 — Multi-env transfer within the imaging family. Train on
# sparse-fourier, phase-retrieval, super-resolution (3 envs). Hold out
# mri-knee-reconstruction for eval only.
#
# Author: Stelios <sdi2200243@di.uoa.gr>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

export EXPERIMENT_ID="E2"
export EXPERIMENT_DESCRIPTION="E2 — Multi-env transfer (imaging family, 3 envs)"
export ENV_SLUGS="sparse-fourier-recovery,phase-retrieval,super-resolution-div2k-x4"
export HELDOUT_ENV="mri-knee-reconstruction"
export DATASET_PATH="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_train_multi.jsonl"
export MAX_STEPS=200
export LEARNING_RATE="1e-6"
export MIN_VRAM_GB=24
export MIN_DISK_GB=30

parse_runner_args "$@"
preflight_or_die "${MIN_VRAM_GB}" "${MIN_DISK_GB}" "${DATASET_PATH}"
setup_run_paths

export EXPERIMENT_CONFIG_JSON
EXPERIMENT_CONFIG_JSON=$(cat <<JSON
{
  "envs": "${ENV_SLUGS}",
  "heldout_env": "${HELDOUT_ENV}",
  "base_model": "${VLABS_BASE_MODEL}",
  "dataset": "${DATASET_PATH}",
  "max_steps": ${MAX_STEPS},
  "learning_rate": ${LEARNING_RATE},
  "lora_r": ${LORA_R},
  "lora_alpha": ${LORA_ALPHA},
  "num_generations": ${NUM_GENERATIONS},
  "resume_from": "${RESUME_FROM}"
}
JSON
)

log_info "experiment:  ${EXPERIMENT_ID} — ${EXPERIMENT_DESCRIPTION}"
log_info "envs:        ${ENV_SLUGS}"
log_info "heldout:     ${HELDOUT_ENV} (eval only)"
log_info "output:      ${OUTPUT_DIR}"
log_info "dry_run:     ${DRY_RUN}"

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
    vlabs-reward-train dry-run "${common_args[@]}"
    log_info "[${EXPERIMENT_ID}] dry-run OK"
    exit 0
fi

trap 'write_manifest $?' EXIT
mkdir -p "${OUTPUT_DIR}"
vlabs-reward-train train "${common_args[@]}" 2>&1 | tee "${LOG_FILE}"
exit ${PIPESTATUS[0]}
