#!/usr/bin/env bash
# scripts/run_all_experiments.sh — Phase 18 + E1-E10 master orchestrator.
#
# Workflow per experiment:
#   1. Run scripts/smoke_test_experiment.py (~$0.20 on RTX 5090).
#   2. If smoke passes -> fire scripts/experiments/run_<id>.sh.
#   3. If smoke fails  -> STOP. The whole point of the protocol is to
#      catch problems on the $0.20 budget, not the $7+ budget.
#
# Resume:
#   Per-experiment status is recorded in `.smoke_test_state.txt`
#   (one line per id: `<id>:<status>`). Re-running skips ids already at
#   `:done` and picks up from the first non-`:done`.
#
# Author: Stelios <sdi2200243@di.uoa.gr>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
STATE_FILE="${REPO_ROOT}/.smoke_test_state.txt"
SMOKE_SCRIPT="${REPO_ROOT}/scripts/smoke_test_experiment.py"
HEALTH_SCRIPT="${REPO_ROOT}/scripts/check_pipeline_health.py"
EXPERIMENTS_DIR="${REPO_ROOT}/scripts/experiments"

# Order matters: phase18-redo first, then E1..E10 (E9 intentionally omitted).
EXPERIMENTS=(
    "phase18-redo"
    "E1"
    "E2"
    "E3"
    "E4"
    "E5a"
    "E5b"
    "E6"
    "E7"
    "E8"
    "E10"
)

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$LOG_DIR"
touch "$STATE_FILE"

record_state() {
    local exp="$1"
    local status="$2"
    local tmp
    tmp="$(mktemp)"
    # Drop any prior entries for this id so each line is the latest state.
    grep -v "^${exp}:" "$STATE_FILE" > "$tmp" || true
    printf '%s:%s\n' "$exp" "$status" >> "$tmp"
    mv "$tmp" "$STATE_FILE"
}

already_done() {
    local exp="$1"
    grep -qx "${exp}:done" "$STATE_FILE"
}

run_health_check() {
    if [[ -x "$HEALTH_SCRIPT" || -f "$HEALTH_SCRIPT" ]]; then
        echo "  > pipeline health check"
        if ! "$PYTHON_BIN" "$HEALTH_SCRIPT" 2>&1 | sed 's/^/    /'; then
            echo "  ! pipeline health check FAILED — investigate before continuing"
            return 1
        fi
    fi
    return 0
}

for exp in "${EXPERIMENTS[@]}"; do
    if already_done "$exp"; then
        echo "..skip $exp (already done)"
        continue
    fi

    echo "============================================================"
    echo "Smoke test: $exp"
    echo "============================================================"

    smoke_log="${LOG_DIR}/${exp}_smoke.log"
    if ! "$PYTHON_BIN" "$SMOKE_SCRIPT" --experiment "$exp" 2>&1 | tee "$smoke_log"; then
        rc=${PIPESTATUS[0]}
        echo "x Smoke test failed for $exp (exit $rc) — see $smoke_log"
        record_state "$exp" "smoke-fail:${rc}"
        exit 1
    fi
    echo "+ Smoke test passed for $exp"
    record_state "$exp" "smoke-ok"

    # Runner filenames use underscores even when the experiment id has
    # hyphens (e.g. `phase18-redo` -> `run_phase18_redo.sh`). Convert.
    runner_basename="run_${exp//-/_}.sh"
    runner="${EXPERIMENTS_DIR}/${runner_basename}"
    if [[ ! -f "$runner" ]]; then
        echo "= No runner at ${runner} — pausing here. Author the runner then re-run."
        exit 0
    fi

    ts="$(date +%s)"
    run_log="${LOG_DIR}/${exp}_${ts}.log"
    echo "> Firing full run: $runner (log: $run_log)"
    if ! bash "$runner" 2>&1 | tee "$run_log"; then
        rc=${PIPESTATUS[0]}
        echo "x Full run failed for $exp (exit $rc) — see $run_log"
        record_state "$exp" "run-fail:${rc}"
        exit 1
    fi

    record_state "$exp" "done"
    echo "+ $exp complete"

    if ! run_health_check; then
        record_state "$exp" "done:health-fail"
        exit 1
    fi
done

echo "============================================================"
echo "+ All experiments processed"
echo "============================================================"
