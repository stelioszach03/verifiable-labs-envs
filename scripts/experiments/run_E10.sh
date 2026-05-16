#!/usr/bin/env bash
# scripts/experiments/run_E10.sh
#
# E10 — Adversarial robustness. Apply adversarial prompts against the
# E8 best checkpoint using vlabs-audit (Phase 17 toolchain) and emit
# a robustness PDF report. Short run (~3h, ~$3).
#
# Author: Stelios <sdi2200243@di.uoa.gr>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

export EXPERIMENT_ID="E10"
export EXPERIMENT_DESCRIPTION="E10 — Adversarial robustness audit (E8 checkpoint)"
export AUDIT_MODEL="${VLABS_AUDIT_MODEL:-anthropic/claude-haiku-4.5}"
export AUDIT_ENVS="${VLABS_AUDIT_ENVS:-sparse-fourier-recovery,phase-retrieval,math-algebra}"
export EPISODES="${VLABS_AUDIT_EPISODES:-15}"
export ALPHA="${VLABS_AUDIT_ALPHA:-0.1}"
export PARALLEL="${VLABS_AUDIT_PARALLEL:-4}"
export MIN_VRAM_GB=24
export MIN_DISK_GB=15

parse_runner_args "$@"
preflight_or_die "${MIN_VRAM_GB}" "${MIN_DISK_GB}" ""
setup_run_paths

REPORT_PATH="${OUTPUT_DIR}/adversarial_report.pdf"

export EXPERIMENT_CONFIG_JSON
EXPERIMENT_CONFIG_JSON=$(cat <<JSON
{
  "audit_model": "${AUDIT_MODEL}",
  "audit_envs": "${AUDIT_ENVS}",
  "episodes_per_env": ${EPISODES},
  "alpha": ${ALPHA},
  "parallel": ${PARALLEL},
  "report_path": "${REPORT_PATH}",
  "mission": "adversarial robustness probe vs E8 best checkpoint",
  "resume_from": "${RESUME_FROM}"
}
JSON
)

log_info "experiment:  ${EXPERIMENT_ID} — ${EXPERIMENT_DESCRIPTION}"
log_info "model:       ${AUDIT_MODEL}"
log_info "envs:        ${AUDIT_ENVS}"
log_info "episodes:    ${EPISODES} per env"
log_info "report:      ${REPORT_PATH}"
log_info "dry_run:     ${DRY_RUN}"

audit_args=(
    --model "${AUDIT_MODEL}"
    --envs "${AUDIT_ENVS}"
    --episodes "${EPISODES}"
    --alpha "${ALPHA}"
    --parallel "${PARALLEL}"
    --output "${REPORT_PATH}"
)

if [[ -n "${RESUME_FROM}" ]]; then
    audit_args+=(--resume "${RESUME_FROM}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    vlabs-audit audit "${audit_args[@]}" --dry-run
    log_info "[${EXPERIMENT_ID}] dry-run OK"
    exit 0
fi

trap 'write_manifest $?' EXIT
mkdir -p "${OUTPUT_DIR}"
vlabs-audit audit "${audit_args[@]}" 2>&1 | tee "${LOG_FILE}"
exit ${PIPESTATUS[0]}
