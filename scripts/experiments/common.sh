# scripts/experiments/common.sh — shared env vars + preflight helpers for
# Phase 18 / E1..E10 runner scripts.
#
# Each per-experiment runner sources this file before doing anything else.
# Put ONLY shared state + helper functions here; experiment-specific
# variables (env slug, max_steps, dataset, etc.) belong in the runner.
#
# Author: Stelios <sdi2200243@di.uoa.gr>

# -------- venv activation --------------------------------------------------
# RunPod pods ship Python 3.12 with a PEP-668 externally-managed system
# Python, so all training deps live in /workspace/.venv. Sourcing
# `activate` here lets every runner call plain `python` / `vlabs-*`
# without prefixing with the full venv path.
VLABS_VENV="${VLABS_VENV:-/workspace/.venv}"
VLABS_VENV_PYTHON="${VLABS_VENV}/bin/python"

if [[ -f "${VLABS_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VLABS_VENV}/bin/activate"
fi

# -------- repo / runtime paths --------------------------------------------
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLABS_REPO_ROOT="${VLABS_REPO_ROOT:-$(cd "${COMMON_DIR}/../.." && pwd)}"
VLABS_OUTPUT_BASE="${VLABS_OUTPUT_BASE:-${VLABS_REPO_ROOT}/runs}"
VLABS_LOG_BASE="${VLABS_LOG_BASE:-${VLABS_REPO_ROOT}/logs}"

mkdir -p "${VLABS_OUTPUT_BASE}" "${VLABS_LOG_BASE}"

# -------- model + caches ---------------------------------------------------
export VLABS_BASE_MODEL="${VLABS_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export VLABS_LARGE_STUDENT="${VLABS_LARGE_STUDENT:-meta-llama/Llama-3.2-3B-Instruct}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# -------- vLLM (colocate mode) ---------------------------------------------
export VLLM_MODE="${VLLM_MODE:-colocate}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

# -------- sampling temperatures -------------------------------------------
export TRAINING_TEMPERATURE="${TRAINING_TEMPERATURE:-0.5}"
export INFERENCE_TEMPERATURE="${INFERENCE_TEMPERATURE:-1.0}"

# -------- LoRA defaults ----------------------------------------------------
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# -------- compute budget ---------------------------------------------------
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export BF16="${BF16:-true}"
export HOURLY_RATE_USD="${HOURLY_RATE_USD:-0.99}"

# -------- helper functions -------------------------------------------------
log_info()  { printf '[%s] [INFO]  %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
log_warn()  { printf '[%s] [WARN]  %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
log_error() { printf '[%s] [ERROR] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# Check that the venv is active.
check_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_error "venv not active. Expected source ${VLABS_VENV}/bin/activate."
        return 1
    fi
    log_info "venv: ${VIRTUAL_ENV}"
}

# Check that torch sees a CUDA device with enough VRAM.
check_cuda() {
    local min_vram_gb="${1:-20}"
    "${VLABS_VENV_PYTHON}" - <<PY || return 1
import sys
try:
    import torch
except ImportError:
    print("torch not installed", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False", file=sys.stderr)
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / (1024 ** 3)
required = float(${min_vram_gb})
if vram_gb < required:
    print(f"VRAM {vram_gb:.1f} GB < required {required:.1f} GB on {props.name}", file=sys.stderr)
    sys.exit(1)
print(f"cuda: {props.name} ({vram_gb:.1f} GB, capability {props.major}.{props.minor})")
PY
}

# Check that ``path`` has at least ``min_gb`` of free space.
check_disk_free() {
    local path="${1:-${VLABS_REPO_ROOT}}"
    local min_gb="${2:-10}"
    "${VLABS_VENV_PYTHON}" - <<PY || return 1
import shutil, sys, pathlib
path = pathlib.Path("${path}")
min_gb = float(${min_gb})
try:
    usage = shutil.disk_usage(str(path))
except FileNotFoundError:
    print(f"path missing: {path}", file=sys.stderr)
    sys.exit(1)
free_gb = usage.free / (1024 ** 3)
if free_gb < min_gb:
    print(f"{free_gb:.1f} GB free at {path} < required {min_gb:.1f} GB", file=sys.stderr)
    sys.exit(1)
print(f"disk: {free_gb:.1f} GB free at {path}")
PY
}

# Confirm that ``dataset_path`` is non-empty and looks like JSONL.
check_dataset() {
    local dataset_path="$1"
    if [[ -z "${dataset_path}" ]]; then
        log_info "dataset: (none — runner uses tool's default)"
        return 0
    fi
    if [[ ! -f "${dataset_path}" ]]; then
        log_warn "dataset path does not exist (yet): ${dataset_path}"
        return 0  # Non-fatal: dataset may be generated by the runner.
    fi
    local first_byte
    first_byte=$(head -c 1 "${dataset_path}" 2>/dev/null)
    if [[ "${first_byte}" != "{" && "${first_byte}" != "[" ]]; then
        log_warn "dataset doesn't look like JSON/JSONL: ${dataset_path}"
    fi
    log_info "dataset: ${dataset_path} ($(wc -l <"${dataset_path}") lines)"
}

# Compose pre-flight: ``preflight_or_die <min_vram_gb> <min_disk_gb> [dataset_path]``.
preflight_or_die() {
    local min_vram_gb="${1:-20}"
    local min_disk_gb="${2:-10}"
    local dataset_path="${3:-}"

    log_info "preflight: min_vram=${min_vram_gb}GB min_disk=${min_disk_gb}GB"

    check_venv || { log_error "venv check failed"; exit 1; }
    check_cuda "${min_vram_gb}" || { log_error "cuda check failed"; exit 1; }
    check_disk_free "${VLABS_REPO_ROOT}" "${min_disk_gb}" || {
        log_error "disk check failed"
        exit 1
    }
    check_dataset "${dataset_path}"
    log_info "preflight: OK"
}

# Write the post-run manifest under ``${OUTPUT_DIR}/manifest.json``.
# Expects the caller to have set: EXPERIMENT_ID, EXPERIMENT_DESCRIPTION,
# OUTPUT_DIR, LOG_FILE, START_TS, plus optional EXPERIMENT_CONFIG_JSON.
write_manifest() {
    local exit_code="${1:-0}"
    local end_ts
    end_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    mkdir -p "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
    local manifest="${OUTPUT_DIR}/manifest.json"

    "${VLABS_VENV_PYTHON}" - <<PY
import json, os
from datetime import datetime

start_raw = "${START_TS:-}"
end_raw = "${end_ts}"
exit_code = ${exit_code}
hours = 0.0
if start_raw:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start = datetime.strptime(start_raw, fmt)
    end = datetime.strptime(end_raw, fmt)
    hours = (end - start).total_seconds() / 3600.0

cost = round(hours * float("${HOURLY_RATE_USD}"), 4)
gpu_hours = round(hours, 3)

config_raw = """${EXPERIMENT_CONFIG_JSON:-{}}"""
try:
    config = json.loads(config_raw)
except json.JSONDecodeError:
    config = {"_raw": config_raw}

manifest = {
    "experiment_id": "${EXPERIMENT_ID:?EXPERIMENT_ID must be set}",
    "description": "${EXPERIMENT_DESCRIPTION:-}",
    "started_at": start_raw,
    "completed_at": end_raw,
    "exit_code": exit_code,
    "ok": exit_code == 0,
    "output_dir": "${OUTPUT_DIR}",
    "log_file": "${LOG_FILE:-}",
    "checkpoint_path": "${OUTPUT_DIR}/final_model",
    "config": config,
    "gpu_hours": gpu_hours,
    "cost_usd": cost,
    "hourly_rate_usd": float("${HOURLY_RATE_USD}"),
}

with open("${manifest}", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[manifest] wrote ${manifest} (exit_code={exit_code}, cost=\${cost:.2f})")
PY
}

# Setup the per-run output dir + the timestamped log file.
# Sets: OUTPUT_DIR, LOG_FILE, START_TS (UTC ISO-8601).
setup_run_paths() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${VLABS_OUTPUT_BASE}/${EXPERIMENT_ID:?EXPERIMENT_ID must be set}_${ts}"
    LOG_FILE="${VLABS_LOG_BASE}/${EXPERIMENT_ID}_${ts}.log"
    START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    export OUTPUT_DIR LOG_FILE START_TS
}

# Parse --dry-run / --resume <path> from $@. Sets DRY_RUN + RESUME_FROM.
parse_runner_args() {
    DRY_RUN="false"
    RESUME_FROM=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --resume)
                RESUME_FROM="${2:?--resume requires a checkpoint path}"
                shift 2
                ;;
            --resume=*)
                RESUME_FROM="${1#*=}"
                shift
                ;;
            --help|-h)
                echo "Usage: $(basename "$0") [--dry-run] [--resume CHECKPOINT_PATH]"
                exit 0
                ;;
            *)
                log_warn "unknown arg: $1"
                shift
                ;;
        esac
    done
    export DRY_RUN RESUME_FROM
}
