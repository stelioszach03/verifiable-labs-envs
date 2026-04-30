#!/bin/bash
# Driver for M7: run baseline-eval-format on 3 GRPO checkpoints, sequentially.
#
# Each call: 100 seeds × 3 samples = 300 episodes, ~21 min each (after model load).
# Total: ~75 min wall (incl. ~3-4 min/load × 3 from Drive).
#
# Outputs go to /content/drive/MyDrive/verifiable-labs/training_outputs/.
set -euo pipefail

REPO_DIR="/content/verifiable-labs-envs"
CKPT_BASE="/content/drive/MyDrive/verifiable-labs/checkpoints/qwen15b_grpo_sf_v1"
OUT_BASE="/content/drive/MyDrive/verifiable-labs/training_outputs"

cd "$REPO_DIR"

for STEP in 100 250 500; do
  CKPT="$CKPT_BASE/checkpoint-$STEP"
  OUT="$OUT_BASE/qwen15b_grpo_eval_ckpt${STEP}.jsonl"
  STATS="$OUT_BASE/qwen15b_grpo_eval_ckpt${STEP}_stats.json"

  if [ ! -d "$CKPT" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT" >&2
    exit 1
  fi
  if [ -f "$OUT" ]; then
    echo "SKIP ckpt-$STEP: $OUT already exists"
    continue
  fi

  echo
  echo "=== EVAL CHECKPOINT-$STEP at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python examples/training/eval_qwen_baseline.py \
    --model-id "$CKPT" \
    --out "$OUT" \
    --stats "$STATS"
done

echo
echo "=== ALL 3 EVALS DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$OUT_BASE"/qwen15b_grpo_eval_ckpt*.jsonl
