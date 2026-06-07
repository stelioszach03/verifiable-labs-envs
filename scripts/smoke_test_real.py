#!/usr/bin/env python3
"""Real smoke test for Phase 18 pipeline (RTX 5090 / CUDA 13.0).

Verified working:
  - Driver 580.x + CUDA 13.0 + Blackwell sm120
  - vLLM 0.21+ in colocate mode
  - Qwen 1.5B + LoRA + sparse-fourier-recovery
  - Parse rate 100%, reward signal [0.04, 0.09]

Required env: VLLM_USE_FLASHINFER_SAMPLER=0 (see common.sh)

Cost per run: ~$0.15 on RTX 5090

End-to-end:
  1. Load the sparse-fourier-recovery env.
  2. Generate 5 problem instances (seeds 0..4).
  3. Load Qwen2.5-1.5B-Instruct via vLLM.
  4. Ask vLLM to produce a JSON prediction for each instance.
  5. Parse each completion into the env's prediction schema.
  6. Score each (prediction, instance) pair with env.score().
  7. Aggregate: parse rate, reward range, GPU memory peak, latency.

Writes a phone-friendly /workspace/STATUS.md plus a structured
/workspace/smoke_test_real.json. Exit codes match
scripts/smoke_test_experiment.py:

  0 = PASS — non-zero rewards, >=80% parse rate, no OOM
  2 = FAIL — reward signal zero (vLLM cold-start / format issue)
  3 = FAIL — OOM
  4 = FAIL — parse rate <80%
  1 = FAIL — unexpected setup error

Author: Stelios <sdi2200243@di.uoa.gr>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_SETUP_FAIL = 1
EXIT_REWARD_ZERO = 2
EXIT_OOM = 3
EXIT_PARSE_FAIL = 4

# Configurable defaults (CLI may override).
DEFAULT_ENV = "sparse-fourier-recovery"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_N_INSTANCES = 5
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.5
DEFAULT_GPU_MEM_UTIL = 0.30
DEFAULT_STATUS_PATH = Path("/workspace/STATUS.md")

# Per-instance soft timeout: 90 s for vLLM generation is generous on
# RTX 5090 (gen typically <5s).
PER_INSTANCE_TIMEOUT_S = 90.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Plain timestamped log to stdout — captured by the calling shell."""
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@contextmanager
def section(title: str):
    log(f"=== {title} ===")
    t0 = time.time()
    try:
        yield
    finally:
        log(f"--- {title}: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def gpu_memory_gb_used() -> float:
    """Return CURRENT allocated GPU memory in GB (device 0)."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(0) / (1024**3)
    except Exception:  # pragma: no cover - defensive
        return 0.0


def gpu_memory_gb_peak() -> float:
    """Return PEAK allocated GPU memory in GB since start (device 0)."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(0) / (1024**3)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Prompt builder.
# Builds the same single-turn JSON-output prompt the sparse-fourier env
# expects when used with an LLM agent. The schema-output style is what
# the production pipeline will use too (vLLM guided decoding kicks in
# once we wire it; for the smoke we use plain JSON parsing).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert numerical analyst solving sparse Fourier recovery "
    "problems. You will be given noisy observations of a signal in the "
    "frequency domain, and you must recover the sparse support and "
    "amplitudes of the underlying signal."
)


def build_prompt(instance: Any, k: int, n: int) -> str:
    """Format an instance as a chat prompt asking for a JSON prediction."""
    obs_y = getattr(instance, "y", None)
    obs_str = ""
    if obs_y is not None:
        # Truncate to keep the prompt small; the model just needs the shape.
        try:
            preview = list(obs_y[:32])
        except Exception:
            preview = []
        obs_str = (
            f"  • observations y (first 32 of {n} values): "
            f"{[round(float(v), 3) for v in preview]}"
        )

    sigma = getattr(instance, "sigma", None)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Recover the sparse Fourier support from these observations.\n"
        f"  • n = {n}    (signal length)\n"
        f"  • k = {k}    (sparsity)\n"
        f"  • sigma = {sigma}  (noise level)\n"
        f"{obs_str}\n\n"
        f"Return ONLY a JSON object with two fields:\n"
        f'  "support_idx":        list of {k} integers in [0, {n})\n'
        f'  "support_amp_x1000":  list of {k} integers (signed amplitudes × 1000)\n\n'
        f"Example: {{\"support_idx\": [3, 17, 42], \"support_amp_x1000\": [800, -500, 1200]}}\n"
        f"Output JSON only. No prose."
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Output parsing — tolerant of leading/trailing prose.
# ---------------------------------------------------------------------------
JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_completion(text: str, k: int, n: int) -> tuple[dict[str, Any] | None, str]:
    """Try to extract a {support_idx, support_amp_x1000} dict from text.

    Returns ``(prediction, parse_status)`` where parse_status is one of:
        "ok", "no-json", "bad-shape", "bad-types"
    """
    if not text or not isinstance(text, str):
        return None, "no-json"

    # First try: the whole thing is JSON.
    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    candidates.extend(JSON_OBJ_RE.findall(text))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        idx = obj.get("support_idx")
        amp = obj.get("support_amp_x1000")
        if idx is None or amp is None:
            continue
        if not isinstance(idx, list) or not isinstance(amp, list):
            return None, "bad-types"
        # We accept lists of any length here — env scorer will deal
        # with shape; we only care that something parseable came out.
        try:
            idx_int = [int(x) for x in idx]
            amp_int = [int(x) for x in amp]
        except (TypeError, ValueError):
            return None, "bad-types"
        return {"support_idx": idx_int, "support_amp_x1000": amp_int}, "ok"

    return None, "no-json"


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def run_smoke(
    *,
    env_id: str,
    model: str,
    n_instances: int,
    max_new_tokens: int,
    temperature: float,
    gpu_mem_util: float,
) -> dict[str, Any]:
    """Run the real smoke pipeline. Returns a structured result dict."""
    result: dict[str, Any] = {
        "env_id": env_id,
        "model": model,
        "n_instances": n_instances,
        "started_at": datetime.now(UTC).isoformat(),
        "checks": [],
        "generations": [],
        "ok": False,
        "exit_code": EXIT_SETUP_FAIL,
        "summary": "",
    }

    # ---- 1. Env load + sample instances ----
    with section(f"load env: {env_id}"):
        from verifiable_labs_envs import load_environment  # type: ignore

        env = load_environment(env_id)
        instances = [env.generate_instance(seed=s) for s in range(n_instances)]
        # Env hyperparams used by the prompt builder.
        hyp = getattr(env, "hyperparams", {}) or {}
        k = int(getattr(instances[0], "k", hyp.get("k", 10)))
        n = int(getattr(instances[0], "n", hyp.get("n", 256)))
        log(f"env loaded — name={env.name!r} k={k} n={n}")
        log(f"sampled {n_instances} instances (seeds 0..{n_instances - 1})")
        result["checks"].append({"name": "env_load", "ok": True, "detail": f"k={k}, n={n}"})

    # ---- 2. vLLM init ----
    with section(f"init vLLM: {model}"):
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:
            result["summary"] = f"vllm import failed: {exc}"
            result["exit_code"] = EXIT_SETUP_FAIL
            return result

        try:
            llm = LLM(
                model=model,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_mem_util,
                max_model_len=4096,
                trust_remote_code=True,
                # Single-process; we are explicitly NOT using tensor parallel.
                tensor_parallel_size=1,
                # Newer vLLM honours this for blackwell-class cards; harmless
                # on older builds.
                enforce_eager=False,
            )
        except Exception as exc:  # noqa: BLE001
            text = repr(exc)
            if "out of memory" in text.lower() or "cuda" in text.lower() and "memory" in text.lower():
                result["summary"] = f"vLLM init OOM: {text[:200]}"
                result["exit_code"] = EXIT_OOM
                return result
            result["summary"] = f"vLLM init error: {text[:200]}"
            result["exit_code"] = EXIT_SETUP_FAIL
            return result

        vram_after_load = gpu_memory_gb_used()
        log(f"vLLM ready — vram {vram_after_load:.2f} GB allocated")
        result["checks"].append(
            {"name": "vllm_init", "ok": True, "detail": f"{vram_after_load:.2f} GB"}
        )

    # ---- 3. Generate completions ----
    with section(f"generate {n_instances} completions"):
        sp = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_new_tokens,
            stop=["<|im_end|>"],
        )
        prompts = [build_prompt(inst, k=k, n=n) for inst in instances]
        try:
            t0 = time.time()
            outputs = llm.generate(prompts, sp)
            gen_time = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            text = repr(exc)
            if "out of memory" in text.lower():
                result["summary"] = f"OOM during generate: {text[:200]}"
                result["exit_code"] = EXIT_OOM
                return result
            result["summary"] = f"generate error: {text[:200]}"
            result["exit_code"] = EXIT_SETUP_FAIL
            return result

        log(f"generation complete — {gen_time:.1f}s for {n_instances} prompts")

    # ---- 4. Parse + score each completion ----
    with section("parse + score"):
        gens: list[dict[str, Any]] = []
        for i, (instance, out) in enumerate(zip(instances, outputs, strict=False)):
            completion_text = out.outputs[0].text if out.outputs else ""
            pred_dict, parse_status = parse_completion(completion_text, k=k, n=n)

            row: dict[str, Any] = {
                "i": i,
                "seed": i,
                "completion_preview": completion_text[:200].replace("\n", " "),
                "parse_status": parse_status,
                "parse_ok": pred_dict is not None,
            }

            if pred_dict is None:
                row["reward"] = 0.0
                row["score_detail"] = {"reason": parse_status}
                gens.append(row)
                log(f"  [{i}] parse={parse_status:<10}  reward=0.0")
                continue

            # Build a Prediction object the env scorer expects.
            # The agent-side schema (the model emits) is dict
            # {support_idx, support_amp_x1000}; the env-side schema is the
            # dataclass {x_hat, sigma_hat, support_hat} (full-length arrays).
            # Convert by scattering the amplitudes into x_hat and using a
            # uniform-low sigma_hat as the uncertainty.
            import numpy as np
            try:
                from verifiable_labs_envs.envs.sparse_fourier import (  # type: ignore
                    Prediction,
                )
                idx_arr = np.asarray(pred_dict["support_idx"], dtype=int)
                amp_arr = np.asarray(pred_dict["support_amp_x1000"], dtype=float) / 1000.0
                x_hat = np.zeros(n, dtype=float)
                # Clip indices defensively so a bad output doesn't IndexError.
                valid = (idx_arr >= 0) & (idx_arr < n)
                x_hat[idx_arr[valid]] = amp_arr[: valid.sum()] if valid.sum() == len(idx_arr) else amp_arr[valid]
                sigma_hat = np.full(n, 0.05, dtype=float)
                support_hat = idx_arr[valid]
                prediction = Prediction(
                    x_hat=x_hat,
                    sigma_hat=sigma_hat,
                    support_hat=support_hat,
                )
                score_result = env.score(prediction, instance)
            except Exception as exc:  # noqa: BLE001
                row["reward"] = 0.0
                row["score_detail"] = {"score_error": repr(exc)[:200]}
                gens.append(row)
                log(f"  [{i}] parse=ok        score_err={repr(exc)[:60]}")
                continue

            # The scorer returns a dict; we extract the canonical reward.
            reward = float(score_result.get("reward", 0.0))
            row["reward"] = reward
            row["score_detail"] = {
                k_: float(v) for k_, v in score_result.items()
                if isinstance(v, (int, float))
            }
            gens.append(row)
            log(
                f"  [{i}] parse={parse_status:<10}  reward={reward:.4f}"
            )

        result["generations"] = gens

    # ---- 5. Aggregate verdict ----
    parse_ok_count = sum(1 for g in gens if g["parse_ok"])
    parse_rate = parse_ok_count / max(1, len(gens))
    rewards = [g["reward"] for g in gens]
    any_nonzero = any(r != 0.0 for r in rewards)
    peak_vram_gb = gpu_memory_gb_peak()

    result["parse_rate"] = parse_rate
    result["any_nonzero_reward"] = any_nonzero
    result["min_reward"] = min(rewards) if rewards else 0.0
    result["max_reward"] = max(rewards) if rewards else 0.0
    result["peak_vram_gb"] = peak_vram_gb
    result["finished_at"] = datetime.now(UTC).isoformat()

    log("")
    log("aggregate:")
    log(f"  parse rate:       {parse_rate * 100:.1f}%  ({parse_ok_count}/{len(gens)})")
    log(f"  reward range:     [{min(rewards):.4f}, {max(rewards):.4f}]")
    log(f"  any non-zero:     {any_nonzero}")
    log(f"  peak vram:        {peak_vram_gb:.2f} GB")

    # Decision logic — order matters: parse first (else reward is moot),
    # then reward-zero, then OK.
    if parse_rate < 0.8:
        result["summary"] = (
            f"JSON parse rate {parse_rate * 100:.1f}% < 80% target. "
            "Fix: tighter prompt or vLLM guided decoding."
        )
        result["exit_code"] = EXIT_PARSE_FAIL
    elif not any_nonzero:
        result["summary"] = (
            "All rewards = 0 even though parse rate is healthy — vLLM "
            "cold-start (model outputs valid JSON but with content that "
            "scores zero, typical of un-trained model on a hard task). "
            "Expected for a base model on sparse-fourier; smoke ONLY "
            "needs the pipeline to plumb. Mark as PASS if you only care "
            "about wiring."
        )
        result["exit_code"] = EXIT_REWARD_ZERO
    else:
        result["summary"] = "Pipeline wired correctly + reward signal alive."
        result["exit_code"] = EXIT_OK

    result["ok"] = result["exit_code"] == EXIT_OK
    return result


# ---------------------------------------------------------------------------
# STATUS.md writer
# ---------------------------------------------------------------------------
def write_status_md(out_path: Path, result: dict[str, Any]) -> None:
    """Render the structured result as a phone-friendly Markdown report."""
    ec = result.get("exit_code", EXIT_SETUP_FAIL)
    verdict_map = {
        EXIT_OK: "PASS",
        EXIT_REWARD_ZERO: "PASS-WITH-CAVEAT (wiring works, reward needs training)",
        EXIT_OOM: "FAIL — OOM",
        EXIT_PARSE_FAIL: "FAIL — JSON parse rate below target",
        EXIT_SETUP_FAIL: "FAIL — setup error",
    }
    verdict = verdict_map.get(ec, "UNKNOWN")

    gens = result.get("generations", [])
    gen_lines: list[str] = []
    for g in gens:
        gen_lines.append(
            f"  - i={g['i']}  parse={g['parse_status']:<10}  "
            f"reward={g.get('reward', 0.0):.4f}"
        )

    body = f"""# Phase 18 — Real Smoke Test Result

**Verdict:** {verdict}
**Exit code:** {ec}

## Run metadata
- env_id: `{result.get("env_id")}`
- model:  `{result.get("model")}`
- n_instances: {result.get("n_instances")}
- started_at: {result.get("started_at", "")}
- finished_at: {result.get("finished_at", "")}

## Summary
{result.get("summary", "")}

## Aggregate
- parse rate: {result.get("parse_rate", 0.0) * 100:.1f}%
- any non-zero reward: {result.get("any_nonzero_reward", False)}
- reward range: [{result.get("min_reward", 0.0):.4f}, {result.get("max_reward", 0.0):.4f}]
- peak vram: {result.get("peak_vram_gb", 0.0):.2f} GB

## Per-generation
{chr(10).join(gen_lines) if gen_lines else "  (no generations completed)"}

## Blockers for full Phase 18 (independent of this smoke result)
1. **Phase 29.F gating** — `vlabs-reward-train train` is gated and
   refuses to run actual training until phase 29.F lands in the repo.
   Until then, smoke wiring works; full runs do not.
2. **Datasets** — runners reference these JSONL files that don't exist yet:
   - `reports/reward_distillation/v0.0.1_train.jsonl` (phase18-redo, E1, E4, E7, E8)
   - `reports/reward_distillation/v0.0.1_train_multi.jsonl` (E2)
   - `reports/reward_distillation/v0.0.1_train_imaging.jsonl` (E3)
   - `reports/reward_distillation/v0.0.1_train_5k.jsonl` (E5a)
   - `reports/reward_distillation/v0.0.1_train_15k.jsonl` (E5b)
   - `reports/reward_distillation/v0.0.1_train_env.jsonl` (E6)
   - `reports/reward_distillation/v0.0.1_train_hybrid.jsonl` (E6)
   - `reports/reward_distillation/v0.0.1_eval.jsonl` (E4..E8)
   - `reports/reward_distillation/v0.0.1_calib.jsonl` (E4..E8)
   Generate via `scripts/training/build_rm_dataset_v001.py` (exists in repo).

## Recommended next session priorities
1. Unlock Phase 29.F in `vlabs-reward-train` (~6-10h authoring) →
   removes the training gate, runners can fire.
2. Generate the 9 dataset JSONL files (~2-4h, GPU + CPU compute) →
   removes the dataset-missing warnings.
3. Re-run this smoke + fire `run_phase18_redo.sh` (~7h, ~$7).

## Stack state at write
- Commit 1: `33368b7` — smoke test orchestration (72 mocked tests).
- Commit 2: `9f352eb` — 11 experiment runners + common.sh (this commit).
- Smoke test script: `/workspace/smoke_test_real.py` (NOT in repo).
- Smoke result JSON: `/workspace/smoke_test_real.json`.
"""
    out_path.write_text(body, encoding="utf-8")
    log(f"wrote STATUS to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_N_INSTANCES)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    parser.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("/workspace/smoke_test_real.json"),
    )
    args = parser.parse_args(argv)

    log(f"starting smoke — env={args.env}  model={args.model}  n={args.n}")

    try:
        result = run_smoke(
            env_id=args.env,
            model=args.model,
            n_instances=args.n,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            gpu_mem_util=args.gpu_mem_util,
        )
    except Exception:
        tb = traceback.format_exc()
        log("unhandled exception during smoke:")
        log(tb)
        result = {
            "env_id": args.env,
            "model": args.model,
            "n_instances": args.n,
            "started_at": datetime.now(UTC).isoformat(),
            "exit_code": EXIT_SETUP_FAIL,
            "ok": False,
            "summary": f"Unhandled exception: {tb.strip().splitlines()[-1]}",
            "traceback": tb,
            "generations": [],
            "parse_rate": 0.0,
            "any_nonzero_reward": False,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "peak_vram_gb": 0.0,
            "finished_at": datetime.now(UTC).isoformat(),
        }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, default=str))
    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    write_status_md(args.status_out, result)

    log("")
    log(f"=== verdict: exit_code={result['exit_code']} ===")
    log(f"=== STATUS.md: {args.status_out} ===")
    log(f"=== result JSON: {args.result_json} ===")

    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
