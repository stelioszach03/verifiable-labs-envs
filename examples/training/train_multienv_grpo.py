"""Multi-env joint GRPO training for Qwen2.5-1.5B-Instruct.

Combines three training envs (sparse-fourier, phase-retrieval, super-resolution)
with a held-out env (mri-knee-reconstruction) for generalisation evaluation.
Uses RLVE adaptive difficulty (one tracker per train env) and P-GRPO
posterior-gated rewards (per-env outcome thresholds + reasoning tags).

PHASE C.4 SCOPE — implementation only. The default invocation runs the
pre-flight schema check + builds the dataset + constructs the GRPOTrainer,
but does NOT call ``trainer.train()``. Pass ``--launch`` explicitly to
actually train (intended for a separate session after pre-flight is
approved).

References
----------
* RLVE: Zeng et al. 2025, arXiv:2511.07317 (adaptive difficulty)
* P-GRPO: Fan et al. 2025, arXiv:2508.05170 (posterior reward gating)
* Logic-RL: Xie et al. 2025, arXiv:2502.14768 (reasoning tags)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

# HF cache → Drive (survives VM death).
DEFAULT_CACHE = "/content/drive/MyDrive/verifiable-labs/hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(DEFAULT_CACHE) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(DEFAULT_CACHE) / "transformers"))

import torch  # noqa: E402

# torch.load weights_only patch (PyTorch 2.6+ default change; same as
# train_grpo_qwen.py — see M6 writeup).
_ORIG_TORCH_LOAD = torch.load


def _torch_load_full_pickle(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_full_pickle  # type: ignore[assignment]

from datasets import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from verifiable_labs_envs import __version__, load_environment  # noqa: E402
from verifiable_labs_envs.repro import config_hash  # noqa: E402
from verifiable_labs_envs.solvers.llm_solver import get_adapter  # noqa: E402
from verifiable_labs_envs.training import (  # noqa: E402
    OUTCOME_THRESHOLDS_REGISTRY,
    AdaptiveDifficultyTracker,
    difficulty_to_kwargs,
    make_reward_fn_multienv,
    validate_env_schema,
)

# ── Constants ──────────────────────────────────────────────────────────


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_ENVS: list[str] = [
    "sparse-fourier-recovery",
    "phase-retrieval",
    "super-resolution-div2k-x4",
]
HELDOUT_ENV: str = "mri-knee-reconstruction"

# Disjoint per-env seed pools. Pool size 1000 per env × 3 envs = 3000 train
# instances. The held-out env's TEST seeds (3000-3099) are never used during
# training. The single-env M5 sparse-fourier TEST set (2000-2099) is also
# avoided here to keep cross-experiment comparability.
SEED_POOLS: dict[str, dict[str, range]] = {
    "sparse-fourier-recovery": {
        "train": range(0, 1000),
        "val":   range(1000, 1200),
        "test":  range(2000, 2100),  # M5 baseline TEST set
    },
    "phase-retrieval": {
        "train": range(10_000, 11_000),
        "val":   range(11_000, 11_200),
        "test":  range(12_000, 12_100),
    },
    "super-resolution-div2k-x4": {
        "train": range(20_000, 21_000),
        "val":   range(21_000, 21_200),
        "test":  range(22_000, 22_100),
    },
    "mri-knee-reconstruction": {  # held-out — never sampled in training
        "train": range(0, 0),  # empty: explicitly never trained on
        "val":   range(0, 0),
        "test":  range(30_000, 30_100),
    },
}

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/verifiable-labs/checkpoints/qwen15b_grpo_multi_v1"
)

# Logic-RL reasoning-tags system prompt; same wording as the single-env
# script, applies across all train envs.
TAGGED_SYSTEM_PROMPT = """You are an expert in inverse problems for compressed measurements.

Given a forward operator and noisy measurements, recover the underlying signal subject to the env-specific structure (sparse support, image grid, phase ambiguity, etc.).

Think step-by-step about the recovery problem:
1. Identify the measurement structure
2. Reason about which entries are most likely active / what the signal looks like
3. Estimate amplitudes / pixel values for the chosen support / region

Place your reasoning inside <think>...</think> tags.
Place your final answer inside <answer>...</answer> tags as a JSON object whose schema matches the env's expected output."""


# ── Pre-flight ──────────────────────────────────────────────────────────


def assert_seeds_disjoint() -> None:
    """Each env's TRAIN/VAL/TEST must be internally disjoint, AND the
    train pool of one env must never overlap with the test pool of any
    other env."""
    all_pools: list[tuple[str, str, set[int]]] = []
    for eid, pools in SEED_POOLS.items():
        for split_name, rng in pools.items():
            all_pools.append((eid, split_name, set(rng)))
    for i, (e1, s1, p1) in enumerate(all_pools):
        for e2, s2, p2 in all_pools[i + 1:]:
            overlap = p1 & p2
            if overlap:
                raise AssertionError(
                    f"seed pool overlap: ({e1}/{s1}) ∩ ({e2}/{s2}) = "
                    f"{sorted(overlap)[:5]}..."
                )


def run_preflight(envs: list[str]) -> dict[str, Any]:
    """Verify every env loads, generates instances at every difficulty
    anchor, and produces a `score()` dict whose schema matches the
    OUTCOME_THRESHOLDS_REGISTRY entry for that env.

    Returns a structured report (also useful for the milestone summary).
    """
    report: dict[str, Any] = {}
    print("=== PRE-FLIGHT (multi-env GRPO) ===\n")
    for eid in envs:
        print(f"  [{eid}]")
        rec: dict[str, Any] = {"env_id": eid, "errors": []}
        # 1. Schema registry exists.
        cfg = OUTCOME_THRESHOLDS_REGISTRY.get(eid)
        if cfg is None:
            rec["errors"].append("missing OUTCOME_THRESHOLDS_REGISTRY entry")
            report[eid] = rec
            continue
        rec["kind"] = cfg["kind"]

        # 2. Env loads + adapter loads.
        try:
            env = load_environment(eid, calibration_quantile=2.0)
            adapter = get_adapter(eid)
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"load_environment/get_adapter: {type(e).__name__}: {e}")
            report[eid] = rec
            continue
        rec["env_class"] = type(env).__name__
        rec["adapter_class"] = type(adapter).__name__

        # 3. Pick a probe seed outside every TRAIN/VAL/TEST split for this
        # env, run baseline at default difficulty + at difficulty 0.
        probe_seed = 999_999
        try:
            score_default = env.run_baseline(seed=probe_seed)
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"run_baseline (default): {type(e).__name__}: {e}")
            report[eid] = rec
            continue
        rec["baseline_reward"] = float(score_default.get("reward", 0.0))
        rec["score_components"] = list(
            (score_default.get("components") or {}).keys()
        )
        rec["score_meta"] = list((score_default.get("meta") or {}).keys())

        # 4. Schema validation against OUTCOME_THRESHOLDS_REGISTRY.
        schema_errs = validate_env_schema(eid, score_default)
        if schema_errs:
            rec["errors"].extend(schema_errs)

        # 5. Confirm difficulty_to_kwargs returns valid overrides at every
        # anchor; instantiate one instance per anchor.
        anchor_results: list[dict[str, Any]] = []
        from verifiable_labs_envs.training.adaptive_difficulty import ANCHOR_TABLES
        for thresh, _ in ANCHOR_TABLES.get(eid, []):
            kwargs = difficulty_to_kwargs(eid, thresh)
            try:
                inst = env.generate_instance(seed=probe_seed, **kwargs)
                anchor_results.append({
                    "difficulty": thresh,
                    "kwargs_keys": sorted(kwargs.keys()),
                    "instance_class": type(inst).__name__,
                })
            except Exception as e:  # noqa: BLE001
                rec["errors"].append(
                    f"difficulty {thresh}: generate_instance failed "
                    f"({type(e).__name__}): {e}"
                )
        rec["anchor_results"] = anchor_results

        # Print verdict.
        if rec["errors"]:
            print(f"    ❌ {len(rec['errors'])} error(s)")
            for er in rec["errors"]:
                print(f"       - {er}")
        else:
            print(f"    ✅ kind={cfg['kind']}  components={rec['score_components']}  "
                  f"meta={rec['score_meta']}  baseline_r={rec['baseline_reward']:.3f}  "
                  f"anchors={len(anchor_results)}")

        report[eid] = rec
    return report


# ── Per-env adaptive trackers + JSON persistence ───────────────────────


def make_per_env_trackers(env_ids: list[str], **tracker_kwargs: Any) -> dict[str, AdaptiveDifficultyTracker]:
    return {
        eid: AdaptiveDifficultyTracker(env_id=eid, **tracker_kwargs)
        for eid in env_ids
    }


def save_tracker_states(trackers: dict[str, AdaptiveDifficultyTracker],
                        path: str | Path) -> None:
    state = {eid: t.to_dict() for eid, t in trackers.items()}
    Path(path).write_text(json.dumps(state, indent=2))


def load_tracker_states(path: str | Path) -> dict[str, AdaptiveDifficultyTracker]:
    raw = json.loads(Path(path).read_text())
    return {eid: AdaptiveDifficultyTracker.from_dict(d) for eid, d in raw.items()}


# ── Sampler: per-batch (env, difficulty, seed) ─────────────────────────


def sample_batch_rows(
    trackers: dict[str, AdaptiveDifficultyTracker],
    *,
    n_rows: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Generate a batch of (env_id, difficulty, instance_seed) rows.

    Per row:
    1. Uniformly sample env_id from the trackers dict.
    2. Sample difficulty from that env's tracker.
    3. Sample a seed from the env's TRAIN pool (with replacement — TRL
       resamples across epochs anyway).

    The final dataset row that goes into TRL is (prompt, instance_seed,
    env_id) — difficulty is captured for telemetry only and NOT passed
    through to env.generate_instance here. (Phase 14's reward_fn would
    need to also re-apply difficulty kwargs, but C.4 keeps the dataset
    construction simple by deferring difficulty translation to a future
    extension.)
    """
    r = rng if rng is not None else random
    env_ids = list(trackers.keys())
    rows: list[dict[str, Any]] = []
    for _ in range(n_rows):
        eid = r.choice(env_ids)
        diff = trackers[eid].sample_difficulty(rng=r)
        train_pool = SEED_POOLS[eid]["train"]
        seed = r.choice(list(train_pool))
        rows.append({"env_id": eid, "difficulty": int(diff), "instance_seed": int(seed)})
    return rows


# ── Dataset construction (chat-templated prompts) ──────────────────────


def build_multienv_dataset(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    use_tags: bool,
) -> Dataset:
    """Build a HF Dataset where each row carries:
       - prompt (str): pre-templated chat string
       - instance_seed (int)
       - env_id (str): consumed by reward_fn dispatcher
       - difficulty (int): for telemetry / future curriculum use
    """
    # Cache one (env, adapter) per env_id touched.
    envs: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    for r in rows:
        eid = r["env_id"]
        if eid not in envs:
            envs[eid] = load_environment(eid, calibration_quantile=2.0)
            adapters[eid] = get_adapter(eid)

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        eid = r["env_id"]
        env = envs[eid]
        adapter = adapters[eid]
        # D.0 fix: thread per-row difficulty through to the env so the
        # curriculum actually drives instance generation (previously the
        # difficulty was sampled but never passed to env.generate_instance).
        diff_kwargs = difficulty_to_kwargs(eid, int(r["difficulty"]))
        instance = env.generate_instance(seed=r["instance_seed"], **diff_kwargs)
        user_prompt = adapter.build_user_prompt(instance)
        if use_tags:
            system_prompt = TAGGED_SYSTEM_PROMPT
        else:
            system_prompt = adapter.system_prompt or "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        out_rows.append({
            "prompt": prompt_text,
            "instance_seed": int(r["instance_seed"]),
            "env_id": eid,
            "difficulty": int(r["difficulty"]),
        })
    return Dataset.from_list(out_rows)


# ── Dry-run main ───────────────────────────────────────────────────────


def _git_sha(repo_root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _build_config_dict(args: argparse.Namespace, dataset_size: int) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "train_envs": list(TRAIN_ENVS),
        "heldout_env": HELDOUT_ENV,
        "seed_pools": {
            eid: {k: [r.start, r.stop] for k, r in pools.items()}
            for eid, pools in SEED_POOLS.items()
        },
        "outcome_thresholds": {
            eid: OUTCOME_THRESHOLDS_REGISTRY[eid]
            for eid in TRAIN_ENVS
        },
        "use_tags": True,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
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
        "save_total_limit": 10,
        "dataset_size": dataset_size,
        "tracker_init": {"tau_acc": args.tau_acc, "tau_num": args.tau_num,
                         "d_delta": args.d_delta},
        "git_sha": _git_sha(args.repo_root),
        "package_version": __version__,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launch", action="store_true",
                    help="Actually call trainer.train(). Default: dry-run only.")
    ap.add_argument("--dataset-size", type=int, default=300,
                    help="number of (env, seed) rows in the train dataset (default 300)")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="GRPOConfig.max_steps (default 300; RLVE intends fewer steps "
                         "via adaptive difficulty)")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--tau-acc", type=float, default=0.9)
    ap.add_argument("--tau-num", type=int, default=32)
    ap.add_argument("--d-delta", type=int, default=4)
    ap.add_argument("--no-tags", action="store_true",
                    help="Disable reasoning tags (NOT recommended for Phase 14).")
    ap.add_argument("--repo-root", default="/content/verifiable-labs-envs")
    args = ap.parse_args()

    use_tags = not args.no_tags

    # 1. Disjoint-pool assertion.
    assert_seeds_disjoint()
    print("✅ seed pools disjoint across all (env, split) combinations")

    # 2. Pre-flight schema check on TRAIN_ENVS only (held-out is for eval).
    report = run_preflight(TRAIN_ENVS)
    fatal = [eid for eid, rec in report.items() if rec.get("errors")]
    if fatal:
        print(f"\n❌ Pre-flight failed for: {fatal}")
        return 2

    # 3. Trackers + sampler smoke (purely combinatorial — no GPU).
    trackers = make_per_env_trackers(
        TRAIN_ENVS,
        tau_acc=args.tau_acc, tau_num=args.tau_num, d_delta=args.d_delta,
    )
    rng = random.Random(42)
    rows = sample_batch_rows(trackers, n_rows=args.dataset_size, rng=rng)
    env_dist: dict[str, int] = {}
    for r in rows:
        env_dist[r["env_id"]] = env_dist.get(r["env_id"], 0) + 1
    print(f"\n=== sampler distribution over {len(rows)} rows ===")
    for eid, c in sorted(env_dist.items()):
        print(f"  {eid:<32} {c:>4} rows ({100 * c / len(rows):.1f}%)")

    # 4. Reward function builds for all 3 train envs (no-op for held-out).
    reward_fn = make_reward_fn_multienv(TRAIN_ENVS, use_tags=use_tags)
    print(f"\n✅ make_reward_fn_multienv built for {TRAIN_ENVS}; "
          f"use_tags={reward_fn.use_tags}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dict = _build_config_dict(args, dataset_size=len(rows))
    cfg_dict["config_hash"] = config_hash({
        k: v for k, v in cfg_dict.items()
        if k not in {"git_sha", "package_version", "tracker_init"}
    })
    config_path = OUTPUT_DIR / "config.json"
    config_path.write_text(json.dumps(cfg_dict, indent=2, default=str))
    print(f"✅ wrote config.json (config_hash={cfg_dict['config_hash']}) "
          f"→ {config_path}")

    # Save initial tracker states for resumability tooling.
    tracker_path = OUTPUT_DIR / "tracker_state.json"
    save_tracker_states(trackers, tracker_path)
    print(f"✅ wrote initial tracker_state.json → {tracker_path}")

    if not args.launch:
        print("\n[DRY-RUN] all pre-flight checks passed; trainer.train() NOT called.")
        print("        Pass --launch to run actual training.")
        return 0

    # 5. (Only if --launch) construct tokenizer + dataset + GRPOTrainer.
    print("\n=== LAUNCH MODE ===")
    print(f"Loading tokenizer for {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=False, cache_dir=DEFAULT_CACHE,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Building train dataset ({len(rows)} rows)...", flush=True)
    train_ds = build_multienv_dataset(rows, tokenizer=tokenizer, use_tags=use_tags)

    from trl import GRPOConfig, GRPOTrainer  # imported here to avoid eager import

    grpo_cfg = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=False,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
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
        log_completions=False,
        report_to="none",
        seed=42,
        save_strategy="steps",
        logging_strategy="steps",
        save_total_limit=10,
        remove_unused_columns=False,  # keep env_id + instance_seed
        gradient_checkpointing=False,
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    print(f"Starting training (max_steps={args.max_steps})...", flush=True)
    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0
    print(f"\nTRAIN DONE: wall={wall:.1f}s  global_step={trainer.state.global_step}")
    save_tracker_states(trackers, tracker_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
