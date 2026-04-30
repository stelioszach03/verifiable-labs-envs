"""Universal eval entry point — works across base-model families.

Thin wrapper around :mod:`eval_qwen_baseline` that validates the chosen
``--base-model-id`` against :data:`MODEL_REGISTRY` (from
:mod:`train_grpo_universal`) and resolves ``--checkpoint-path`` into the
``--model-id`` argument expected downstream. Same eval protocol as M5
(100 seeds × 3 samples × temp=0.9, paired by ``instance_hash``) so the
output JSONL schema is identical across models — paired comparisons in
the capability report stay apples-to-apples.

Usage
-----
    # Baseline eval of a fresh base model (no checkpoint):
    python eval_universal.py --base-model-id Qwen/Qwen2.5-1.5B-Instruct \
        --out /content/.../qwen15b_base_eval.jsonl \
        --stats /content/.../qwen15b_base_eval_stats.json

    # Eval a fine-tuned checkpoint:
    python eval_universal.py --base-model-id Qwen/Qwen2.5-1.5B-Instruct \
        --checkpoint-path /content/.../checkpoint-500 \
        --out /content/.../qwen15b_grpo_eval_ckpt500.jsonl

    # Dry-run validation only (no eval, no model load):
    python eval_universal.py --base-model-id Qwen/Qwen2.5-1.5B-Instruct --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import the model registry + dry-run helpers from the universal trainer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grpo_universal import (  # type: ignore[import-not-found]
    MODEL_REGISTRY,
    dry_run_check,
    _print_report,
    vram_budget_warning,
)

EVAL_BASELINE_SCRIPT = Path(__file__).resolve().parent / "eval_qwen_baseline.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model-id", required=True,
                    help="HuggingFace model id (must be in MODEL_REGISTRY)")
    ap.add_argument("--checkpoint-path", default=None,
                    help="optional local path to a fine-tuned checkpoint dir; "
                         "if set, this is passed to eval_qwen_baseline as --model-id")
    ap.add_argument("--out", default=None, help="output JSONL path")
    ap.add_argument("--stats", default=None, help="output stats JSON path")
    ap.add_argument("--env-id", default="sparse-fourier-recovery")
    ap.add_argument("--seeds-start", type=int, default=2000)
    ap.add_argument("--seeds-end", type=int, default=2099)
    ap.add_argument("--samples-per-seed", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--use-tags", action="store_true",
                    help="reasoning-tags prompt + format gate (off by default to "
                         "preserve the M5 baseline protocol)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate model registry + tokenizer + chat template only; "
                         "do NOT call eval_qwen_baseline")
    args = ap.parse_args()

    if args.base_model_id not in MODEL_REGISTRY:
        print(f"❌ {args.base_model_id} not in MODEL_REGISTRY. "
              f"Add an entry first.", file=sys.stderr)
        return 2

    print("=== Universal eval — DRY-RUN model validation ===\n")
    report = dry_run_check(args.base_model_id, require_token_present=False)
    _print_report(report)
    warn = vram_budget_warning(args.base_model_id)
    if warn:
        print(f"    {warn}")
    print()

    if report["errors"]:
        print(f"❌ Dry-run failed: {len(report['errors'])} error(s)")
        return 2

    if args.dry_run:
        print(f"[DRY-RUN] {args.base_model_id} ready for eval; not running.")
        return 0

    effective_model_id = (
        args.checkpoint_path if args.checkpoint_path else args.base_model_id
    )

    cmd = [
        sys.executable,
        str(EVAL_BASELINE_SCRIPT),
        "--model-id", effective_model_id,
        "--env-id", args.env_id,
        "--seeds-start", str(args.seeds_start),
        "--seeds-end", str(args.seeds_end),
        "--samples-per-seed", str(args.samples_per_seed),
        "--temperature", str(args.temperature),
    ]
    if args.out:
        cmd.extend(["--out", args.out])
    if args.stats:
        cmd.extend(["--stats", args.stats])
    if args.use_tags:
        cmd.append("--use-tags")

    print(f"=== Launching downstream eval ===")
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
