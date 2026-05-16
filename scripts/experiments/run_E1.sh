#!/usr/bin/env bash
# scripts/experiments/run_E1.sh
#
# E1 — Replicate the Phase 13.2 baseline: GRPO on Qwen-1.5B over
# sparse-fourier-recovery. Must reproduce +12.6% reward delta and the
# p<1.2e-14 paired-test signal from the paper.
#
# Author: Stelios <sdi2200243@di.uoa.gr>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

export EXPERIMENT_ID="E1"
export EXPERIMENT_DESCRIPTION="E1 — Phase 13.2 baseline replication (sparse-fourier-recovery)"
export ENV_SLUG="sparse-fourier-recovery"
export DATASET_PATH="${VLABS_REPO_ROOT}/reports/reward_distillation/v0.0.1_train.jsonl"
export MAX_STEPS=150
export LEARNING_RATE="1e-6"
export MIN_VRAM_GB=24
export MIN_DISK_GB=20

parse_runner_args "$@"
preflight_or_die "${MIN_VRAM_GB}" "${MIN_DISK_GB}" "${DATASET_PATH}"
setup_run_paths

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
  "num_generations": ${NUM_GENERATIONS},
  "expected_reward_delta": "+0.126 (per Phase 13.2 paper)",
  "expected_p_value": "<1.2e-14",
  "resume_from": "${RESUME_FROM}"
}
JSON
)

log_info "experiment:  ${EXPERIMENT_ID} — ${EXPERIMENT_DESCRIPTION}"
log_info "output:      ${OUTPUT_DIR}"
log_info "log:         ${LOG_FILE}"
log_info "dry_run:     ${DRY_RUN}"
[[ -n "${RESUME_FROM}" ]] && log_info "resume_from: ${RESUME_FROM}"

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
