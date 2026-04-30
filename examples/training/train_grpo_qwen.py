"""GRPO training for Qwen2.5-1.5B-Instruct on sparse-fourier-recovery.

Modes
-----
Smoke (10 steps, TRAIN seeds 0-9, log_completions=True):
    python examples/training/train_grpo_qwen.py --smoke

Full (500 steps, TRAIN seeds 0-999, log_completions=False):
    python examples/training/train_grpo_qwen.py

Resume after VM death (cf. M6 plan):
    1. New Colab session, mount Drive, ensure repo + cloudflared.
    2. ssh kevin 'ls /content/drive/MyDrive/verifiable-labs/checkpoints/qwen15b_grpo_sf_v1/'
       → find the highest checkpoint-NNN directory.
    3. ssh kevin 'cd /content/verifiable-labs-envs && \
            nohup python examples/training/train_grpo_qwen.py \
                --resume-from-checkpoint /path/to/checkpoint-NNN \
                > /content/drive/.../grpo_resume_$(date +%Y%m%dT%H%M%S).log 2>&1 &'
    4. Confirm "Resuming from step NNN" in the log.

`--smoke` and `--resume-from-checkpoint` are mutually exclusive.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Set HF cache to Drive BEFORE importing transformers/trl — survives VM death.
DEFAULT_CACHE = "/content/drive/MyDrive/verifiable-labs/hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(DEFAULT_CACHE) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(DEFAULT_CACHE) / "transformers"))

import torch

# PyTorch 2.6+ defaults torch.load(..., weights_only=True), which rejects
# the numpy globals embedded in HF Trainer's rng_state.pth. Our checkpoints
# are produced by this same script in the same Drive — fully trusted — so
# we restore the pre-2.6 behaviour. Must be applied before any
# transformers.Trainer code path imports torch.load via a closure.
_ORIG_TORCH_LOAD = torch.load


def _torch_load_full_pickle(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_full_pickle  # type: ignore[assignment]

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from verifiable_labs_envs import __version__, load_environment
from verifiable_labs_envs.repro import config_hash
from verifiable_labs_envs.solvers.llm_solver import get_adapter
from verifiable_labs_envs.training import make_reward_fn


# ── Constants (M6 spec) ───────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ENV_ID = "sparse-fourier-recovery"

TRAIN_SEEDS = list(range(0, 1000))
VAL_SEEDS = list(range(1000, 1200))
TEST_SEEDS = list(range(2000, 2100))

OUTPUT_DIR_FULL = Path(
    "/content/drive/MyDrive/verifiable-labs/checkpoints/qwen15b_grpo_sf_v1"
)
OUTPUT_DIR_SMOKE = Path(
    "/content/drive/MyDrive/verifiable-labs/checkpoints/qwen15b_grpo_sf_smoke_v1"
)
SMOKE_TRAIN_SEEDS = list(range(0, 10))


# Logic-RL / DeepSeek-R1 reasoning-tags system prompt, opt-in via --use-tags.
# References:
#   * Logic-RL (Xie et al. 2025, arXiv:2502.14768)
#   * DeepSeek-R1 distilled chat template
TAGGED_SYSTEM_PROMPT = """You are an expert in sparse signal recovery from compressed measurements.

Given a measurement vector y obtained via y = M @ F^* @ x + noise, where x is sparse with k non-zero components, recover the support and amplitudes of x.

Think step-by-step about the recovery problem:
1. Identify the measurement structure
2. Reason about which indices are most likely non-zero
3. Estimate amplitudes for those indices

Place your reasoning inside <think>...</think> tags.
Place your final answer inside <answer>...</answer> tags as JSON with keys 'support_idx' (k integers) and 'support_amp_x1000' (k integers, amplitudes scaled by 1000)."""


# ── Pre-flight ────────────────────────────────────────────────────────


def assert_seeds_disjoint() -> None:
    s_train, s_val, s_test = set(TRAIN_SEEDS), set(VAL_SEEDS), set(TEST_SEEDS)
    overlap_tv = s_train & s_val
    overlap_te = s_train & s_test
    overlap_ve = s_val & s_test
    if overlap_tv or overlap_te or overlap_ve:
        raise AssertionError(
            f"TRAIN/VAL/TEST not disjoint: "
            f"TRAIN∩VAL={sorted(overlap_tv)} "
            f"TRAIN∩TEST={sorted(overlap_te)} "
            f"VAL∩TEST={sorted(overlap_ve)}"
        )


def _git_sha(repo_root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ── Dataset construction ──────────────────────────────────────────────


def build_dataset(seeds: list[int], env: Any, adapter: Any, tokenizer: Any,
                  *, use_tags: bool = False) -> Dataset:
    """Build a HF Dataset where each row carries:
       - prompt (str): pre-templated chat string ready for model.generate
       - instance_seed (int): consumed by reward_fn via TRL kwargs

    When ``use_tags`` is True, the system prompt swaps to
    :data:`TAGGED_SYSTEM_PROMPT` (Logic-RL / DeepSeek-R1 style) so the
    model is asked to emit ``<think>...</think>`` followed by
    ``<answer>...</answer>``. The reward function should be constructed
    with the same ``use_tags=True`` so its format gate matches.
    """
    if use_tags:
        system_prompt = TAGGED_SYSTEM_PROMPT
    else:
        system_prompt = adapter.system_prompt or "You are a helpful assistant."
    rows = []
    for seed in seeds:
        instance = env.generate_instance(seed=seed)
        user_prompt = adapter.build_user_prompt(instance)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        rows.append({"prompt": prompt_text, "instance_seed": int(seed)})
    return Dataset.from_list(rows)


# ── Logging callback ──────────────────────────────────────────────────


class JSONLLogCallback(TrainerCallback):
    """Write every Trainer.log() event to a JSONL file (one record per line)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Don't truncate on resume — append.
        if not self.path.exists():
            self.path.write_text("")

    def on_log(  # noqa: D401 — Trainer hook
        self, args, state, control, logs=None, **kwargs
    ):
        if not logs:
            return
        record = {
            "step": int(getattr(state, "global_step", 0)),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            **logs,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> int:
    """Convert a per-event JSONL training log to CSV with the union of all keys."""
    if not jsonl_path.exists():
        return 0
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        return 0
    keys = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    return len(rows)


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="10-step smoke run with 10 train seeds; log_completions=True")
    ap.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="absolute path to a checkpoint dir; output_dir is "
                         "inferred from the checkpoint's parent")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override max_steps; useful for resume tests "
                         "(e.g. resume from checkpoint-5 with --max-steps 10)")
    ap.add_argument("--use-tags", action="store_true",
                    help="enable Logic-RL/DeepSeek-R1 reasoning tags: system prompt "
                         "switches to TAGGED_SYSTEM_PROMPT and the reward function "
                         "format gate requires <think>...</think><answer>...</answer>")
    ap.add_argument("--repo-root", default="/content/verifiable-labs-envs")
    args = ap.parse_args()

    # Pre-flight: seeds disjoint
    assert_seeds_disjoint()

    if args.smoke:
        output_dir = OUTPUT_DIR_SMOKE
        max_steps = 10
        save_steps = 5
        logging_steps = 1
        log_completions = True
        train_seeds = SMOKE_TRAIN_SEEDS
    else:
        output_dir = OUTPUT_DIR_FULL
        max_steps = 500
        save_steps = 50
        logging_steps = 10
        log_completions = False
        train_seeds = TRAIN_SEEDS

    if args.resume_from_checkpoint:
        cp_path = Path(args.resume_from_checkpoint)
        assert cp_path.is_dir(), f"checkpoint dir not found: {cp_path}"
        ts_path = cp_path / "trainer_state.json"
        assert ts_path.exists(), f"missing trainer_state.json in {cp_path}"
        ts = json.loads(ts_path.read_text())
        # Resume always writes to the checkpoint's own parent directory.
        # This lets `--resume-from-checkpoint <smoke-dir>/checkpoint-5` work
        # with or without --smoke, and `<full-dir>/checkpoint-NNN` work
        # without --smoke (the typical VM-death recovery case).
        output_dir = cp_path.parent
        print(f"[resume] {cp_path.name} → global_step={ts.get('global_step')}; "
              f"output_dir set to {output_dir}",
              flush=True)

    # --max-steps overrides the smoke/full default (useful for resume tests
    # that need to extend or shorten an existing run's step budget).
    if args.max_steps is not None:
        max_steps = int(args.max_steps)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build config dict (for repro hash + saved config.json)
    cfg_dict: dict[str, Any] = {
        "model_id": MODEL_ID,
        "env_id": ENV_ID,
        "train_seeds_range": [train_seeds[0], train_seeds[-1]],
        "val_seeds_range": [VAL_SEEDS[0], VAL_SEEDS[-1]],
        "test_seeds_range": [TEST_SEEDS[0], TEST_SEEDS[-1]],
        "max_steps": max_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "learning_rate": 1e-6,
        "num_generations": 4,
        "max_prompt_length": 2048,
        "max_completion_length": 1024,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "bf16": True,
        "use_vllm": False,
        "beta": 0.04,
        "temperature": 0.9,
        "top_p": 1.0,
        "log_completions": log_completions,
        "smoke": bool(args.smoke),
        "use_tags": bool(args.use_tags),
        "git_sha": _git_sha(args.repo_root),
        "package_version": __version__,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "trl": __import__("trl").__version__,
            "accelerate": __import__("accelerate").__version__,
            "peft": __import__("peft").__version__,
            "datasets": __import__("datasets").__version__,
        },
    }
    cfg_dict["config_hash"] = config_hash({
        k: v for k, v in cfg_dict.items()
        if k not in {"git_sha", "library_versions"}
    })

    config_path = output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        events = existing.get("resume_events", [])
        events.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "from_checkpoint": (
                str(args.resume_from_checkpoint)
                if args.resume_from_checkpoint else None
            ),
            "git_sha": cfg_dict["git_sha"],
            "config_hash": cfg_dict["config_hash"],
        })
        existing["resume_events"] = events
        config_path.write_text(json.dumps(existing, indent=2))
        print(f"Updated existing config.json with resume_event #{len(events)}",
              flush=True)
    else:
        config_path.write_text(json.dumps(cfg_dict, indent=2))
        print(f"Wrote fresh config.json (config_hash={cfg_dict['config_hash']})",
              flush=True)

    # Env + adapter + reward_fn
    print(f"Loading env + adapter + reward_fn for {ENV_ID} "
          f"(use_tags={args.use_tags})...", flush=True)
    env = load_environment(ENV_ID, calibration_quantile=2.0)
    adapter = get_adapter(ENV_ID)
    reward_fn = make_reward_fn(ENV_ID, use_tags=args.use_tags)

    # Tokenizer
    print(f"Loading tokenizer for {MODEL_ID} from cache={DEFAULT_CACHE}...",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=False, cache_dir=DEFAULT_CACHE,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Datasets
    print(f"Building train dataset ({len(train_seeds)} seeds)...", flush=True)
    train_ds = build_dataset(train_seeds, env, adapter, tokenizer,
                             use_tags=args.use_tags)
    val_ds = None  # eval pass is M7's job; skip during training to keep wall time predictable

    # GRPOConfig
    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=1e-6,
        num_generations=4,
        max_prompt_length=2048,
        max_completion_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        use_vllm=False,
        beta=0.04,
        temperature=0.9,
        top_p=1.0,
        log_completions=log_completions,
        num_completions_to_print=2 if log_completions else None,
        report_to="none",
        seed=42,
        save_strategy="steps",
        logging_strategy="steps",
        save_total_limit=10,
        remove_unused_columns=False,  # keep instance_seed for reward_fn
        gradient_checkpointing=False,
    )

    log_jsonl = output_dir / "training_log.jsonl"
    callbacks = [JSONLLogCallback(log_jsonl)]

    print("Constructing GRPOTrainer (model load follows; ~30-60s)...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print(f"Starting training (max_steps={max_steps})...", flush=True)
    t0 = time.perf_counter()
    if args.resume_from_checkpoint:
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        result = trainer.train()
    wall_time = time.perf_counter() - t0

    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )

    summary = {
        "max_steps_reached": int(trainer.state.global_step),
        "wall_time_sec": wall_time,
        "peak_vram_gb": peak_vram_gb,
        "log_history_n_events": len(trainer.state.log_history),
        "reward_fn_aggregate": reward_fn.stats.aggregate(),
        "smoke": bool(args.smoke),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "config_hash": cfg_dict["config_hash"],
        "git_sha": cfg_dict["git_sha"],
    }
    if hasattr(result, "metrics"):
        summary["train_metrics"] = {k: v for k, v in result.metrics.items()}
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    csv_path = output_dir / "training_log.csv"
    n_log = jsonl_to_csv(log_jsonl, csv_path)

    print()
    print("=== TRAINING SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {n_log} rows → {csv_path}")
    print(f"Wall time: {wall_time:.1f}s | Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
