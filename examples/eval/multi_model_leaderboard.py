"""Multi-model leaderboard eval over Verifiable Labs envs via OpenRouter.

Two modes:
  --estimate-only : run a small sample (5 episodes / model on the cheapest
                    env) to measure typical token counts, query OpenRouter
                    pricing, and project total cost. Prints a per-model +
                    total cost estimate; exits 0 without spending more
                    than ~$0.50.
  --launch        : run the full leaderboard. Background-friendly. Hard
                    kills at ``--budget-usd`` (default $85) regardless
                    of completion progress.

Outputs (Drive):
  /content/drive/MyDrive/verifiable-labs/leaderboard/
    ├── results.jsonl       (per-episode trace)
    ├── stats.json          (aggregated per (model, env))
    ├── calibration.json    (CP coverage validity per cell)
    ├── costs.jsonl         (cumulative cost log every N calls)
    └── leaderboard.csv     (tidy CSV for HF Space)

Hard rules
----------
* API key read from ``./.env`` (gitignored). Never echoed; never written
  back to a script-output file.
* Cost monitor: hard kill at ``--budget-usd`` cap.
* All raw results saved to Drive *before* any post-processing.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

# Drive HF cache (envs may import from there).
DEFAULT_CACHE = "/content/drive/MyDrive/verifiable-labs/hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(DEFAULT_CACHE) / "hub"))


# ── .env loader ──────────────────────────────────────────────────────


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            # Always overwrite (the .env file is the source of truth here);
            # avoids "empty env var present" surprises with setdefault.
            os.environ[k.strip()] = v.strip()


_REPO = Path(__file__).resolve().parents[2]
_load_dotenv(_REPO / ".env")


# ── Models + envs ────────────────────────────────────────────────────


MODELS_TIER1 = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
]
MODELS_TIER2 = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-4o",
]
MODELS_TIER3 = [
    "anthropic/claude-opus-4.6",
]
ALL_MODELS = MODELS_TIER1 + MODELS_TIER2 + MODELS_TIER3  # Option B: 6 models


# Map the "ct-reconstruction" alias requested by the user to the actual
# registered env id.
ENV_ALIASES = {
    "ct-reconstruction": "lodopab-ct-simplified",
}

# Default leaderboard env list (priority order).
DEFAULT_ENVS = [
    "sparse-fourier-recovery",
    "phase-retrieval",
    "super-resolution-div2k-x4",
    "mri-knee-reconstruction",
    "lodopab-ct-simplified",  # the actual id for "ct-reconstruction"
]


# Logic-RL tagged system prompt (Phase C.3) — applied across all envs +
# all models so the format gate is comparable.
TAGGED_SYSTEM_PROMPT = (
    "You are an expert in inverse problems for compressed measurements.\n\n"
    "Given a forward operator and noisy measurements, recover the "
    "underlying signal subject to the env-specific structure.\n\n"
    "Think step-by-step. Place your reasoning inside <think>...</think>. "
    "Place your final answer inside <answer>...</answer> as a JSON object "
    "matching the schema described in the user message."
)


OUT_DIR = Path("/content/drive/MyDrive/verifiable-labs/leaderboard")


# ── OpenRouter client ────────────────────────────────────────────────


def _openrouter_client():
    import openai
    return openai.OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def fetch_pricing(client) -> dict[str, dict[str, float]]:
    """Pull /models endpoint and build a {model_id: {prompt, completion}} pricing map."""
    import urllib.request

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {client.api_key}"},
    )
    with urllib.request.urlopen(req, timeout=20) as f:
        data = json.loads(f.read())
    out: dict[str, dict[str, float]] = {}
    for m in data.get("data", []):
        mid = m.get("id")
        p = m.get("pricing", {})
        try:
            out[mid] = {
                "prompt": float(p.get("prompt", 0.0)),       # USD per token
                "completion": float(p.get("completion", 0.0)),
            }
        except (TypeError, ValueError):
            continue
    return out


def call_model(
    client, *, model: str, system_prompt: str, user_prompt: str,
    temperature: float, max_tokens: int, pricing: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """One chat completion. Returns dict with text + tokens + cost + latency."""
    t0 = time.perf_counter()
    last_err: Exception | None = None
    for attempt in range(3):  # retry on transient errors
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if "429" in msg or "rate" in msg.lower() or "overload" in msg.lower():
                time.sleep(2.0 * (2 ** attempt))
                continue
            else:
                raise
    else:
        raise last_err or RuntimeError("unreachable")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    text = resp.choices[0].message.content or ""
    usage = resp.usage
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    pr = pricing.get(model, {})
    cost_usd = pt * pr.get("prompt", 0.0) + ct * pr.get("completion", 0.0)

    return {
        "model": model,
        "text": text,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost_usd": float(cost_usd),
        "latency_ms": latency_ms,
    }


# ── Episode runner ───────────────────────────────────────────────────


def run_episode(
    *, env, adapter, env_id: str, seed: int, sample_idx: int, model: str,
    client, pricing: dict[str, dict[str, float]],
    temperature: float, max_tokens: int,
) -> dict[str, Any]:
    """One (model, env, seed, sample_idx) episode. Always returns a dict;
    never raises."""
    from verifiable_labs_envs.solvers.adapters._common import extract_json_block
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError
    from verifiable_labs_envs.training import parse_with_tags

    instance = env.generate_instance(seed=seed)
    user_prompt = adapter.build_user_prompt(instance)

    try:
        api_result = call_model(
            client, model=model,
            system_prompt=TAGGED_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature, max_tokens=max_tokens,
            pricing=pricing,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "env_id": env_id, "seed": seed, "sample_idx": sample_idx,
            "model": model,
            "reward": 0.0, "parse_valid": 0, "format_valid": 0,
            "components": {}, "failure_type": "api_error",
            "error_message": f"{type(e).__name__}: {str(e)[:200]}",
            "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
            "latency_ms": 0.0,
            "completion": "",
        }

    text = api_result["text"]
    parse_valid = 0
    format_valid = 0
    components: dict[str, float] = {}
    reward = 0.0
    failure_type: str | None = None

    # Stage 1: parse_valid — JSON object somewhere in the text.
    try:
        extract_json_block(text)
        parse_valid = 1
    except LLMSolverError:
        failure_type = "parse_error"

    if parse_valid:
        # Stage 2: format_valid — strict tag-aware parser.
        try:
            prediction = parse_with_tags(text, instance, adapter)
            format_valid = 1
        except LLMSolverError:
            failure_type = "format_error"
            prediction = None
        except Exception:  # noqa: BLE001
            failure_type = "adapter_exception"
            prediction = None

        if format_valid and prediction is not None:
            try:
                score = env.score(prediction, instance)
                reward = float(score.get("reward", 0.0))
                comps = score.get("components", {}) or {}
                meta = score.get("meta", {}) or {}
                components = {
                    **{k: float(v) for k, v in comps.items()},
                    **{f"meta.{k}": float(v) for k, v in meta.items()
                       if isinstance(v, (int, float))},
                }
            except Exception:  # noqa: BLE001
                failure_type = "score_error"

    return {
        "env_id": env_id, "seed": seed, "sample_idx": sample_idx,
        "model": model,
        "reward": reward, "parse_valid": parse_valid, "format_valid": format_valid,
        "components": components, "failure_type": failure_type,
        "prompt_tokens": api_result["prompt_tokens"],
        "completion_tokens": api_result["completion_tokens"],
        "cost_usd": api_result["cost_usd"],
        "latency_ms": api_result["latency_ms"],
        "completion": text,
    }


# ── Pre-flight cost estimate ─────────────────────────────────────────


def estimate_cost(
    *, models: list[str], envs: list[str], n_seeds: int, n_samples: int,
    probe_seeds: int, max_tokens: int, temperature: float,
    out_path: Path,
) -> dict[str, Any]:
    """Run ``probe_seeds`` episodes per model on the cheapest env, then
    extrapolate to the full grid."""
    from verifiable_labs_envs import load_environment
    from verifiable_labs_envs.solvers.llm_solver import get_adapter

    client = _openrouter_client()
    pricing = fetch_pricing(client)
    print(f"Loaded pricing for {len(pricing)} OR models.\n")

    cheap_env = "sparse-fourier-recovery"
    print(f"=== Pre-flight: {probe_seeds} probe episodes per model on "
          f"{cheap_env} ===\n")
    env = load_environment(cheap_env, calibration_quantile=2.0)
    adapter = get_adapter(cheap_env)

    per_model_avg: dict[str, dict[str, float]] = {}
    probe_total_usd = 0.0
    probe_records: list[dict[str, Any]] = []
    for m in models:
        rows = []
        for s in range(2000, 2000 + probe_seeds):
            try:
                rec = run_episode(
                    env=env, adapter=adapter, env_id=cheap_env,
                    seed=s, sample_idx=0, model=m,
                    client=client, pricing=pricing,
                    temperature=temperature, max_tokens=max_tokens,
                )
                rows.append(rec)
                probe_total_usd += rec["cost_usd"]
                probe_records.append(rec)
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️  probe failed: {m} seed={s} {type(e).__name__}: {e}")
        if not rows:
            per_model_avg[m] = {
                "avg_prompt_tokens": 0, "avg_completion_tokens": 0,
                "avg_cost_usd": 0, "n": 0, "fail": True,
            }
            continue
        avg_pt = statistics.fmean(r["prompt_tokens"] for r in rows)
        avg_ct = statistics.fmean(r["completion_tokens"] for r in rows)
        avg_cost = statistics.fmean(r["cost_usd"] for r in rows)
        per_model_avg[m] = {
            "avg_prompt_tokens": avg_pt, "avg_completion_tokens": avg_ct,
            "avg_cost_usd": avg_cost, "n": len(rows), "fail": False,
            "format_valid_rate": statistics.fmean(r["format_valid"] for r in rows),
        }
        print(f"  {m:<35} n={len(rows)}  pt={avg_pt:.0f} ct={avg_ct:.0f}  "
              f"avg=${avg_cost:.4f}  fmt_ok={per_model_avg[m]['format_valid_rate']*100:.0f}%")
    print(f"\nProbe spent: ${probe_total_usd:.4f}\n")

    # Extrapolation: assume sparse-fourier token counts as pessimistic
    # baseline; multi-image envs may use more, but the order of magnitude
    # is similar within a 2× factor. The estimate prints "min/likely/max".
    n_episodes_per_cell = n_seeds * n_samples
    n_cells_per_model = len(envs)
    print(f"=== Extrapolation: {n_episodes_per_cell} episodes/cell × "
          f"{n_cells_per_model} envs/model = "
          f"{n_episodes_per_cell * n_cells_per_model} episodes/model ===\n")
    print(f"{'model':<35}  {'$/episode':>10}  {'×# total':>9}  "
          f"{'subtotal $ (likely)':>22}  {'(min)':>9}  {'(max 2×)':>9}")
    print("-" * 105)
    total_likely = 0.0
    total_min = 0.0
    total_max = 0.0
    breakdown = {}
    for m in models:
        avg_cost = per_model_avg[m].get("avg_cost_usd", 0.0)
        total = n_episodes_per_cell * n_cells_per_model
        sub = avg_cost * total
        sub_min = sub * 0.7
        sub_max = sub * 2.0
        total_likely += sub
        total_min += sub_min
        total_max += sub_max
        breakdown[m] = {"avg_cost": avg_cost, "n_total": total,
                        "subtotal_usd": sub, "min_usd": sub_min, "max_usd": sub_max}
        print(f"{m:<35}  ${avg_cost:>9.4f}  {total:>9}  "
              f"${sub:>20.2f}  ${sub_min:>7.2f}  ${sub_max:>7.2f}")

    print("-" * 105)
    print(f"{'TOTAL':<35}  {'':>10}  {'':>9}  ${total_likely:>20.2f}  "
          f"${total_min:>7.2f}  ${total_max:>7.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "probe_total_usd": probe_total_usd,
        "per_model_avg": per_model_avg,
        "extrapolation": {"n_episodes_per_cell": n_episodes_per_cell,
                          "n_envs": n_cells_per_model,
                          "total_likely_usd": total_likely,
                          "total_min_usd": total_min,
                          "total_max_usd": total_max,
                          "per_model": breakdown},
    }, indent=2))
    print(f"\n✅ wrote {out_path}")
    return {"likely": total_likely, "min": total_min, "max": total_max,
            "probe_spent": probe_total_usd}


# ── Full launch ──────────────────────────────────────────────────────


def run_full_eval(
    *, models: list[str], envs: list[str], n_seeds: int, n_samples: int,
    seeds_start: int, max_tokens: int, temperature: float,
    out_dir: Path, budget_usd: float, cost_log_every: int = 10,
) -> dict[str, Any]:
    from verifiable_labs_envs import load_environment
    from verifiable_labs_envs.solvers.llm_solver import get_adapter

    client = _openrouter_client()
    pricing = fetch_pricing(client)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    costs_path = out_dir / "costs.jsonl"
    if results_path.exists():
        # Resume-friendly: append by default. If you want a clean run, rename.
        print(f"[resume] {results_path} exists; appending.")
    f_results = results_path.open("a")
    f_costs = costs_path.open("a")

    cumulative_usd = 0.0
    n_done = 0
    n_total = len(models) * len(envs) * n_seeds * n_samples
    t0 = time.perf_counter()
    aborted = False

    try:
        for env_id in envs:
            try:
                env = load_environment(env_id, calibration_quantile=2.0)
                adapter = get_adapter(env_id)
            except Exception as e:  # noqa: BLE001
                print(f"[ENV LOAD FAILED] {env_id}: {type(e).__name__}: {e}")
                continue

            for model in models:
                for seed in range(seeds_start, seeds_start + n_seeds):
                    for sample_idx in range(n_samples):
                        rec = run_episode(
                            env=env, adapter=adapter, env_id=env_id,
                            seed=seed, sample_idx=sample_idx, model=model,
                            client=client, pricing=pricing,
                            temperature=temperature, max_tokens=max_tokens,
                        )
                        rec["timestamp"] = time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        )
                        f_results.write(json.dumps(rec) + "\n")
                        f_results.flush()
                        cumulative_usd += rec["cost_usd"]
                        n_done += 1

                        if n_done % cost_log_every == 0 or n_done == n_total:
                            elapsed = time.perf_counter() - t0
                            eta = elapsed * (n_total - n_done) / max(n_done, 1)
                            f_costs.write(json.dumps({
                                "n_done": n_done, "n_total": n_total,
                                "cumulative_usd": cumulative_usd,
                                "elapsed_sec": elapsed,
                                "eta_sec": eta,
                                "timestamp": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                ),
                            }) + "\n")
                            f_costs.flush()
                            print(f"  [{n_done:>4}/{n_total}] "
                                  f"env={env_id} model={model.split('/')[-1]} "
                                  f"reward={rec['reward']:.3f} fmt={rec['format_valid']} "
                                  f"$cum=${cumulative_usd:.3f} "
                                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                                  flush=True)

                        # Hard kill: budget exceeded.
                        if cumulative_usd > budget_usd:
                            print(f"\n🚨 BUDGET EXCEEDED: "
                                  f"${cumulative_usd:.3f} > ${budget_usd:.3f}; "
                                  f"aborting after {n_done} episodes.")
                            aborted = True
                            break
                    if aborted:
                        break
                if aborted:
                    break
            if aborted:
                break
    finally:
        f_results.close()
        f_costs.close()

    return {
        "n_done": n_done, "n_total": n_total,
        "cumulative_usd": cumulative_usd,
        "aborted": aborted,
        "elapsed_sec": time.perf_counter() - t0,
    }


# ── main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimate-only", action="store_true",
                    help="run a small probe to estimate total cost; do NOT launch full eval")
    ap.add_argument("--launch", action="store_true",
                    help="run the full eval (gated by --budget-usd)")
    ap.add_argument("--budget-usd", type=float, default=85.0,
                    help="hard cap on cumulative spend (default $85)")
    ap.add_argument("--probe-seeds", type=int, default=5,
                    help="probe episodes per model in --estimate-only mode")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--seeds-start", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help="comma-separated model_ids; defaults to ALL_MODELS")
    ap.add_argument("--envs", default=",".join(DEFAULT_ENVS),
                    help="comma-separated env_ids; defaults to DEFAULT_ENVS")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    if args.estimate_only and args.launch:
        ap.error("--estimate-only and --launch are mutually exclusive")
    if not args.estimate_only and not args.launch:
        ap.error("Pass --estimate-only or --launch")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    envs = [ENV_ALIASES.get(e.strip(), e.strip())
            for e in args.envs.split(",") if e.strip()]

    out_dir = Path(args.out_dir)

    if args.estimate_only:
        result = estimate_cost(
            models=models, envs=envs,
            n_seeds=args.n_seeds, n_samples=args.n_samples,
            probe_seeds=args.probe_seeds,
            max_tokens=args.max_tokens, temperature=args.temperature,
            out_path=out_dir / "preflight_estimate.json",
        )
        print("\n=== Estimate vs budget ===")
        print(f"  likely: ${result['likely']:.2f}")
        print(f"  min:    ${result['min']:.2f}")
        print(f"  max:    ${result['max']:.2f}")
        print(f"  budget: ${args.budget_usd:.2f}")
        if result["max"] > args.budget_usd:
            print(f"  ⚠️  worst-case ${result['max']:.2f} EXCEEDS budget ${args.budget_usd:.2f}")
        if result["likely"] > args.budget_usd:
            print("  ⚠️  LIKELY exceeds budget; reduce --n-seeds or drop a Tier-3 model")
        return 0

    # Launch path.
    print(f"=== LAUNCH: {len(models)} models × {len(envs)} envs × "
          f"{args.n_seeds} seeds × {args.n_samples} samples = "
          f"{len(models) * len(envs) * args.n_seeds * args.n_samples} episodes ===")
    print(f"Budget cap: ${args.budget_usd:.2f}")

    final = run_full_eval(
        models=models, envs=envs, n_seeds=args.n_seeds, n_samples=args.n_samples,
        seeds_start=args.seeds_start, max_tokens=args.max_tokens,
        temperature=args.temperature, out_dir=out_dir,
        budget_usd=args.budget_usd,
    )
    print("\n=== DONE ===")
    print(f"  episodes_completed: {final['n_done']} / {final['n_total']}")
    print(f"  cumulative_cost:    ${final['cumulative_usd']:.3f}")
    print(f"  aborted_on_budget:  {final['aborted']}")
    print(f"  wall_time:          {final['elapsed_sec']:.1f}s")
    return 1 if final["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
