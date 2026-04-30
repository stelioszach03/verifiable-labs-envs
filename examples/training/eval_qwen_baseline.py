"""Pre-training baseline evaluation for Qwen2.5-1.5B-Instruct on the
``sparse-fourier-recovery`` env.

Uses the SAME prompt CONTENT as the OpenAI-compatible agent
(``adapter.system_prompt`` + ``adapter.build_user_prompt(instance)``)
so the comparison with Claude / GPT benchmarks in the README stays
apples-to-apples. The two messages are then wrapped in ChatML via
``tokenizer.apply_chat_template`` because Qwen2.5-Instruct expects
that format.

Default sampling: 3 samples per seed × 100 seeds (TEST set 2000–2099)
= 300 episodes at ``temperature=0.9``. Per-seed reward is the mean of
its 3 samples; M7 paired comparison pairs by ``instance_hash``.

Pre-flight ``--smoke`` mode: 3 seeds × 1 sample, written to /tmp,
prints peak VRAM + per-episode latency for sanity-checking before the
~90-minute full run.

Usage:
    # Smoke (~ 1-2 min)
    python examples/training/eval_qwen_baseline.py --smoke

    # Full (~ 90 min on A100 80GB)
    python examples/training/eval_qwen_baseline.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Set HF cache to Drive BEFORE importing transformers — survives VM death.
DEFAULT_CACHE = "/content/drive/MyDrive/verifiable-labs/hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(DEFAULT_CACHE) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(DEFAULT_CACHE) / "transformers"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from verifiable_labs_envs import __version__, load_environment
from verifiable_labs_envs.repro import config_hash, instance_hash, reward_hash
from verifiable_labs_envs.solvers.llm_solver import get_adapter
from verifiable_labs_envs.training import make_reward_fn

DEFAULT_OUT = Path(
    "/content/drive/MyDrive/verifiable-labs/training_outputs/qwen15b_base_eval.jsonl"
)
DEFAULT_STATS = Path(
    "/content/drive/MyDrive/verifiable-labs/training_outputs/qwen15b_base_eval_stats.json"
)


# Logic-RL / DeepSeek-R1 reasoning-tags system prompt, opt-in via --use-tags.
TAGGED_SYSTEM_PROMPT = """You are an expert in sparse signal recovery from compressed measurements.

Given a measurement vector y obtained via y = M @ F^* @ x + noise, where x is sparse with k non-zero components, recover the support and amplitudes of x.

Think step-by-step about the recovery problem:
1. Identify the measurement structure
2. Reason about which indices are most likely non-zero
3. Estimate amplitudes for those indices

Place your reasoning inside <think>...</think> tags.
Place your final answer inside <answer>...</answer> tags as JSON with keys 'support_idx' (k integers) and 'support_amp_x1000' (k integers, amplitudes scaled by 1000)."""


def _resolve_out_paths(out: Path, stats: Path) -> tuple[Path, Path]:
    """If ``out`` already exists, append ``_v2`` to both paths."""
    if not out.exists():
        return out, stats
    out2 = out.with_name(out.stem + "_v2.jsonl")
    stats2 = stats.with_name(stats.stem + "_v2.json")
    print(
        f"WARNING: {out} exists — saving as {out2}",
        file=sys.stderr,
    )
    return out2, stats2


def _git_sha(repo_root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _bootstrap_ci95(values: np.ndarray, n_resamples: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(values[idx]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--env-id", default="sparse-fourier-recovery")
    ap.add_argument("--seeds-start", type=int, default=2000)
    ap.add_argument("--seeds-end", type=int, default=2099,
                    help="inclusive upper bound (default 2099)")
    ap.add_argument("--samples-per-seed", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument(
        "--smoke", action="store_true",
        help="3 seeds × 1 sample, output to /tmp — pre-flight sanity check",
    )
    ap.add_argument(
        "--repo-root", default="/content/verifiable-labs-envs",
        help="repository root (for git sha lookup)",
    )
    ap.add_argument(
        "--use-tags", action="store_true",
        help="enable Logic-RL/DeepSeek-R1 reasoning tags: system prompt switches "
             "to TAGGED_SYSTEM_PROMPT and the reward function format gate requires "
             "<think>...</think><answer>...</answer>",
    )
    args = ap.parse_args()

    if args.smoke:
        seeds = list(range(args.seeds_start, args.seeds_start + 3))
        samples_per_seed = 1
        out_path = Path("/tmp/qwen15b_smoke.jsonl")
        stats_path = Path("/tmp/qwen15b_smoke_stats.json")
    else:
        seeds = list(range(args.seeds_start, args.seeds_end + 1))
        samples_per_seed = args.samples_per_seed
        out_path, stats_path = _resolve_out_paths(args.out, args.stats)

    n_total = len(seeds) * samples_per_seed
    print(
        f"Config: {args.model_id} on {args.env_id} | "
        f"{len(seeds)} seeds × {samples_per_seed} samples = {n_total} episodes "
        f"| temp={args.temperature} | out={out_path}"
    )

    env = load_environment(args.env_id, calibration_quantile=2.0)
    adapter = get_adapter(args.env_id)
    reward_fn = make_reward_fn(args.env_id, use_tags=args.use_tags)

    print(f"Loading model + tokenizer (cache={args.cache_dir})...")
    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=False, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        cache_dir=args.cache_dir,
    )
    model.eval()
    load_sec = time.perf_counter() - t_load
    print(f"Model loaded in {load_sec:.1f}s; first param device: {next(model.parameters()).device}")

    cfg = {
        "model_id": args.model_id,
        "env_id": args.env_id,
        "n_seeds": len(seeds),
        "samples_per_seed": samples_per_seed,
        "seeds_range": [seeds[0], seeds[-1]],
        "temperature": args.temperature,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "use_tags": bool(args.use_tags),
    }
    cfg_hash_str = config_hash(cfg)
    git_sha = _git_sha(args.repo_root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_total = time.perf_counter()
    n_done = 0
    traces: list[dict[str, Any]] = []
    fout = out_path.open("w")
    try:
        for seed in seeds:
            instance = env.generate_instance(seed=seed)
            user_prompt = adapter.build_user_prompt(instance)
            if args.use_tags:
                system_prompt = TAGGED_SYSTEM_PROMPT
            else:
                system_prompt = adapter.system_prompt or "You are a helpful assistant."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            tokens = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
            )
            tokens = {k: v.to(model.device) for k, v in tokens.items()}
            n_prompt_tokens = int(tokens["input_ids"].shape[1])

            inst_hash_str = instance_hash(args.env_id, __version__, seed, {})

            for sample_idx in range(samples_per_seed):
                # Per-(seed, sample_idx) RNG so two runs with the same args
                # reproduce the same completions.
                set_seed(int(seed) * 1000 + int(sample_idx))

                t_ep = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(
                        **tokens,
                        max_new_tokens=args.max_completion_length,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_tokens = output[0, n_prompt_tokens:]
                completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                latency_ms = (time.perf_counter() - t_ep) * 1000.0
                n_completion_tokens = int(gen_tokens.shape[0])

                rewards = reward_fn(
                    prompts=[prompt_text],
                    completions=[completion],
                    instance_seed=[seed],
                )
                reward = float(rewards[0])
                rec = reward_fn.stats.per_call[-1]
                comps = rec["components"]
                ftype_value = rec.get("failure_type") or "none"
                parse_ok = comps["parse_valid"] == 1.0
                format_ok = comps["format_valid"] == 1.0

                trace = {
                    "env_name": args.env_id,
                    "agent_name": "qwen2.5-1.5b-instruct",
                    "model_name": args.model_id,
                    "reward": reward,
                    "parse_success": format_ok,
                    "schema_version": 1,
                    "seed": int(seed),
                    "sample_idx": int(sample_idx),
                    "reward_components": {
                        "nmse": comps["nmse"],
                        "support": comps["support"],
                        "conformal": comps["conformal"],
                        "parse_valid": comps["parse_valid"],
                        "format_valid": comps["format_valid"],
                    },
                    "latency_ms": latency_ms,
                    "failure_type": ftype_value,
                    "metadata": {
                        "status": "ok" if format_ok else "failed",
                        "config_hash": cfg_hash_str,
                        "instance_hash": inst_hash_str,
                        "reward_hash": reward_hash(reward),
                        "n_prompt_tokens": n_prompt_tokens,
                        "n_completion_tokens": n_completion_tokens,
                        "completion": completion,
                    },
                }
                fout.write(json.dumps(trace) + "\n")
                fout.flush()
                traces.append(trace)
                n_done += 1

                if args.smoke or n_done % 25 == 0 or n_done == n_total:
                    elapsed = time.perf_counter() - t_total
                    eta = elapsed * (n_total - n_done) / max(n_done, 1)
                    parse_tag = "parse_ok" if parse_ok else "parse_fail"
                    fmt_tag = "fmt_ok" if format_ok else "fmt_fail"
                    print(
                        f"  [{n_done:>4}/{n_total}] seed={seed} s={sample_idx} "
                        f"r={reward:.3f} {parse_tag} {fmt_tag} "
                        f"toks={n_completion_tokens} {latency_ms/1000:.1f}s "
                        f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                        flush=True,
                    )
    finally:
        fout.close()

    wall_time = time.perf_counter() - t_total
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )

    # Stats: per-seed mean reward (averaged over samples), then aggregate.
    by_seed: dict[int, list[float]] = {}
    for t in traces:
        by_seed.setdefault(t["seed"], []).append(t["reward"])
    per_seed_means = np.array(
        [statistics.fmean(v) for v in by_seed.values()], dtype=np.float64
    )

    # Component / parse rates aggregated over ALL episodes (not per-seed-mean).
    all_parse = np.array(
        [t["reward_components"]["parse_valid"] for t in traces], dtype=np.float64
    )
    all_format = np.array(
        [t["reward_components"]["format_valid"] for t in traces], dtype=np.float64
    )
    all_nmse = np.array(
        [t["reward_components"]["nmse"] for t in traces], dtype=np.float64
    )
    all_support = np.array(
        [t["reward_components"]["support"] for t in traces], dtype=np.float64
    )
    all_conformal = np.array(
        [t["reward_components"]["conformal"] for t in traces], dtype=np.float64
    )

    ci_lo, ci_hi = _bootstrap_ci95(per_seed_means)
    stats = {
        "model_id": args.model_id,
        "env_id": args.env_id,
        "n_seeds": len(seeds),
        "samples_per_seed": samples_per_seed,
        "n_episodes_total": n_done,
        "seeds": [seeds[0], seeds[-1]],
        "temperature": args.temperature,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "mean_reward": float(np.mean(per_seed_means)),
        "std_reward": float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0,
        "min_reward": float(np.min(per_seed_means)),
        "max_reward": float(np.max(per_seed_means)),
        "mean_reward_ci95": [ci_lo, ci_hi],
        "parse_failure_rate": float(1.0 - np.mean(all_parse)),
        "format_validity_rate": float(np.mean(all_format)),
        "mean_nmse": float(np.mean(all_nmse)),
        "mean_support": float(np.mean(all_support)),
        "mean_conformal": float(np.mean(all_conformal)),
        "config_hash": cfg_hash_str,
        "git_sha": git_sha,
        "wall_time_sec": wall_time,
        "peak_vram_gb": peak_vram_gb,
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=== STATS ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}" if abs(v) < 1e6 else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print()
    print(f"JSONL: {out_path}")
    print(f"Stats: {stats_path}")
    print(f"Wall time: {wall_time:.1f}s | Peak VRAM: {peak_vram_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
