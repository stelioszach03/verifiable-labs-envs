#!/usr/bin/env bash
# scripts/publish/publish_paced.sh — hour-aligned PyPI batch publisher.
#
# The previous publish_with_retry.sh hit PyPI's "20 new project
# creations per hour, per account" rate limit and then wasted 30+ min
# per package re-trying every 10 minutes against a tight window. After
# 6 hours of wall-time it had successfully uploaded ZERO additional
# packages.
#
# This rewrite respects PyPI's hourly window:
#
#   1. Build every package upfront (CPU-only, no network).
#   2. Upload up to BATCH_SIZE=18 packages per hour (safety margin
#      under PyPI's 20-creations cap).
#   3. When a batch is done OR a 429 fires mid-batch, sleep until the
#      next hour boundary + 5-minute safety pad, then resume.
#   4. State file (dist/publish/.state.txt) is shared with the old
#      orchestrator — re-runs skip already-uploaded packages.
#
# Estimated runtime for 28 NEW projects:
#   wave 1: 18 packages × ~3 s = ~1 min upload, then sleep ~59 min
#   wave 2: remaining 10 × ~3 s = ~30 s upload, done.
# Total: ~1h 30 min wall-time, ~99% of which is hourly-window waits.
#
# Sources tokens from ~/.vlabs-secrets/pypi-tokens.env (same as the
# old orchestrator).
#
# Usage:
#   bash scripts/publish/publish_paced.sh --prod
#   bash scripts/publish/publish_paced.sh --prod --batch-size 16
#   bash scripts/publish/publish_paced.sh --prod --reset    # wipe state

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_ROOT="${REPO_ROOT}/dist/publish"
STATE_FILE="${DIST_ROOT}/.state.txt"
LOG_FILE="${ORCH_LOG:-/tmp/publish_paced.log}"

# ── token source ─────────────────────────────────────────────────
_TOKENS_FILE="${HOME}/.vlabs-secrets/pypi-tokens.env"
if [ -f "$_TOKENS_FILE" ]; then
    if [ -z "${PYPI_API_TOKEN:-}" ] || [ -z "${TEST_PYPI_API_TOKEN:-}" ]; then
        # shellcheck disable=SC1090
        set -a; . "$_TOKENS_FILE"; set +a
    fi
fi

# ── arg parsing ──────────────────────────────────────────────────
MODE=""           # test | prod
BATCH_SIZE=18
SAFETY_PAD_MIN=5  # minutes past the hour boundary before resuming
PACE=4            # seconds between successful uploads (PyPI burst protection)

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prod)            MODE="prod"; shift ;;
        --test)            MODE="test"; shift ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --pace)            PACE="$2"; shift 2 ;;
        --reset)           rm -f "$STATE_FILE"; echo "state reset"; shift ;;
        -h|--help)         sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "ERROR: pick --test or --prod" >&2; exit 2
fi
TOKEN="${PYPI_API_TOKEN:-}"
[ "$MODE" = "test" ] && TOKEN="${TEST_PYPI_API_TOKEN:-}"
if [ -z "$TOKEN" ]; then
    echo "ERROR: token missing for mode=$MODE" >&2; exit 2
fi
UPLOAD_URL="https://upload.pypi.org/legacy/"
[ "$MODE" = "test" ] && UPLOAD_URL="https://test.pypi.org/legacy/"

mkdir -p "$DIST_ROOT"
touch "$STATE_FILE"

log() {
    local ts
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

state_has() { grep -qx "$1" "$STATE_FILE" 2>/dev/null; }
state_add() { echo "$1" >> "$STATE_FILE"; }

# ── enumerate packages ───────────────────────────────────────────
readarray -t PACKAGES < <(python3 - <<'PYEOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "_pypi_helpers", Path("scripts/publish/_pypi_helpers.py"),
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
for p in m.discover_packages():
    print(f"{p.name}|{p.version}|{p.package_dir}")
PYEOF
)

if [ "${#PACKAGES[@]}" -eq 0 ]; then
    log "ERROR: no packages discovered."; exit 1
fi

# ── build all wheels upfront ─────────────────────────────────────
PENDING=()
for line in "${PACKAGES[@]}"; do
    IFS='|' read -r name version pkg_dir <<< "$line"
    state_key="${MODE}:${name}=${version}"
    if state_has "$state_key"; then
        continue
    fi
    pkg_dist="${DIST_ROOT}/${name}"
    if [ ! -f "${pkg_dist}/${name//-/_}-${version}-py3-none-any.whl" ] && \
       [ ! -d "$pkg_dist" -o -z "$(ls -A "$pkg_dist"/*.whl 2>/dev/null)" ]; then
        log "  · building ${name} ${version}..."
        rm -rf "$pkg_dist"; mkdir -p "$pkg_dist"
        if ! (cd "$pkg_dir" && python3 -m build --outdir "$pkg_dist" \
                > "${pkg_dist}/build.log" 2>&1); then
            log "    ⨯ build failed — see ${pkg_dist}/build.log"
            continue
        fi
    fi
    PENDING+=("$name|$version|$pkg_dir")
done

TOTAL_PENDING=${#PENDING[@]}
log "=== publish_paced  mode=$MODE  batch=$BATCH_SIZE  pending=${TOTAL_PENDING} ==="
if [ "$TOTAL_PENDING" -eq 0 ]; then
    log "  ✓ nothing pending — all packages already uploaded"; exit 0
fi

# ── upload in hour-aligned batches ───────────────────────────────
sleep_to_next_hour_plus_pad() {
    local now next_hour wait_s
    now=$(date -u +%s)
    next_hour=$(( ((now / 3600) + 1) * 3600 ))
    wait_s=$(( next_hour + (SAFETY_PAD_MIN * 60) - now ))
    log "  → sleeping ${wait_s}s (until $(date -u -d @$((next_hour + SAFETY_PAD_MIN*60)) +%H:%MUTC))"
    sleep "$wait_s"
}

upload_one() {
    local name="$1" pkg_dist="${DIST_ROOT}/$1" twine_log="${DIST_ROOT}/$1/twine.${MODE}.log"
    TWINE_USERNAME=__token__ \
    TWINE_PASSWORD="$TOKEN" \
    TWINE_NON_INTERACTIVE=1 \
        python3 -m twine upload \
            --repository-url "$UPLOAD_URL" \
            --skip-existing \
            "${pkg_dist}"/*.whl "${pkg_dist}"/*.tar.gz \
            > "$twine_log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        return 0
    fi
    # ANSI-strip + check for 429
    local clean
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$twine_log")
    if echo "$clean" | grep -q "429 Too Many Requests"; then
        return 1
    fi
    return 2
}

idx=0
batch_used=0
DONE=0
FAILED=0
HIT_429=0
while [ "$idx" -lt "${#PENDING[@]}" ]; do
    line="${PENDING[$idx]}"
    IFS='|' read -r name version pkg_dir <<< "$line"
    state_key="${MODE}:${name}=${version}"

    if [ "$batch_used" -ge "$BATCH_SIZE" ] || [ "$HIT_429" -eq 1 ]; then
        log "  → batch full ($batch_used uploads) — waiting for next hourly window"
        sleep_to_next_hour_plus_pad
        batch_used=0
        HIT_429=0
    fi

    log "  · ${name} ${version}  [batch ${batch_used}/${BATCH_SIZE}, slot $((idx+1))/${TOTAL_PENDING}]"
    upload_one "$name"
    rc=$?
    case "$rc" in
        0)
            log "    ✓ uploaded"
            state_add "$state_key"
            DONE=$((DONE+1))
            batch_used=$((batch_used+1))
            idx=$((idx+1))
            sleep "$PACE"
            ;;
        1)
            log "    ⏳ 429 — locking the rest of this batch, will wait for next hour"
            HIT_429=1
            # Don't increment idx — retry this package next batch.
            ;;
        2)
            log "    ⨯ non-retryable error — see ${DIST_ROOT}/${name}/twine.${MODE}.log"
            FAILED=$((FAILED+1))
            idx=$((idx+1))
            ;;
    esac
done

log "=== summary  done=$DONE  failed=$FAILED  total_pending=$TOTAL_PENDING ==="
