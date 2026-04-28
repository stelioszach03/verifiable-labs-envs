"""User-facing ``verifiable`` CLI.

Friendly surface promised by the website:

    verifiable list
    verifiable info --env sparse-fourier-recovery
    verifiable run --env sparse-fourier-recovery --model claude-haiku-4.5 \\
        --episodes 10 --seed 42
    verifiable login

For everything not in that surface (``compare``, ``report``, ``init-env``,
``validate-env``, or the legacy ``run --agent FILE --out F.jsonl``) we
delegate to :mod:`verifiable_labs_envs.cli` so existing developer
workflows keep working unchanged.

Provider routing for ``--model``:

* ``claude-*``   → Anthropic OpenAI-compatible endpoint, ``ANTHROPIC_API_KEY``
* ``gpt-*`` / ``o1`` / ``o3`` / ``o4`` → OpenAI, ``OPENAI_API_KEY``
* ``gemini-*``  → Google OpenAI-compatible endpoint, ``GOOGLE_API_KEY``
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verifiable_labs import __version__

# ── Provider routing ──────────────────────────────────────────────

_PROVIDER_ROUTES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    # OpenRouter — recognised by `provider/model` slug (e.g. openai/gpt-4o-mini,
    # anthropic/claude-haiku-4.5, google/gemini-2.5-flash). Single key, all
    # providers, useful for testing without paying each provider directly.
    (("openrouter/", "anthropic/", "openai/", "google/", "meta-llama/", "qwen/", "deepseek/"),
     "https://openrouter.ai/api/v1/", "OPENROUTER_API_KEY"),
    (("claude-",), "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    (("gpt-", "o1", "o3", "o4"), "https://api.openai.com/v1/", "OPENAI_API_KEY"),
    (("gemini-",), "https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY"),
)

# Per-model pricing in USD per million tokens — (input, output).
# Public list prices as of early 2026; update when providers change rates.
# Unknown models fall back to (0, 0) and the CLI shows "—" for cost.
_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4.5": (1.0, 5.0),
    "claude-haiku-4": (0.80, 4.0),
    "claude-sonnet-4.6": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-5": (5.0, 20.0),
    "gpt-5-mini": (1.0, 4.0),
    "o1": (15.0, 60.0),
    "o3": (10.0, 40.0),
    "o4-mini": (1.0, 4.0),
    # Google
    "gemini-2.5-pro": (1.25, 5.0),
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort cost estimate. Falls back to 0 on unknown models."""
    # Strip any "provider/" prefix used by OpenRouter (e.g. "openai/gpt-4o-mini").
    bare = model.rsplit("/", 1)[-1] if "/" in model else model
    rates = _PRICING.get(bare)
    if not rates:
        # Try a prefix match — covers e.g. "gpt-5-2026-01-01" style versioned ids.
        for known, r in _PRICING.items():
            if bare.startswith(known):
                rates = r
                break
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate

# Friendly subcommands handled here; everything else falls through to dev CLI.
_FRIENDLY = {"list", "info", "run", "login"}
# Subcommands the dev CLI owns — `verifiable envs` etc. Forwarded as-is.
_LEGACY = {"envs", "compare", "report", "init-env", "validate-env"}

# ── Helpers ───────────────────────────────────────────────────────


def _resolve_provider(model: str) -> tuple[str, str]:
    """Return (base_url, api_key_env_var) for a friendly model name.

    Raises SystemExit with a helpful message when the prefix isn't recognised.
    """
    for prefixes, base_url, env_var in _PROVIDER_ROUTES:
        if any(model.startswith(p) for p in prefixes):
            return base_url, env_var
    raise SystemExit(
        f"\n❌ Don't know how to route model {model!r}.\n"
        f"\nSupported prefixes:\n"
        f"  claude-*           → Anthropic   (set ANTHROPIC_API_KEY)\n"
        f"  gpt-* / o1 / o3 / o4 → OpenAI     (set OPENAI_API_KEY)\n"
        f"  gemini-*           → Google      (set GOOGLE_API_KEY)\n"
        f"  <provider>/<model> → OpenRouter  (set OPENROUTER_API_KEY)\n"
        f"    (e.g. openai/gpt-4o-mini, anthropic/claude-haiku-4.5)\n"
    )


def _require_api_key(env_var: str, model: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    raise SystemExit(
        f"\n❌ {env_var} not set (required for model {model!r}).\n"
        f"\nFix:\n"
        f"  export {env_var}=your_key_here\n"
        f"\nOr run `verifiable login` for interactive setup.\n"
    )


def _list_envs() -> list[str]:
    """List env IDs from the envs package registry."""
    try:
        from verifiable_labs_envs import list_environments
    except ImportError as e:
        raise SystemExit(
            "verifiable-labs-envs is not installed. "
            "Reinstall verifiable-labs to pull it in:\n  pip install verifiable-labs\n"
        ) from e
    return list_environments()


def _runs_dir() -> Path:
    d = Path.home() / ".verifiable" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── verifiable list ───────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    envs = _list_envs()
    print(f"{len(envs)} environments available:\n")
    for env in envs:
        print(f"  {env}")
    return 0


# ── verifiable info --env X ───────────────────────────────────────


def cmd_info(args: argparse.Namespace) -> int:
    import warnings

    envs = _list_envs()
    if args.env not in envs:
        print(f"Error: unknown environment {args.env!r}.", file=sys.stderr)
        print(f"Run `verifiable list` to see all {len(envs)} envs.", file=sys.stderr)
        return 2

    from verifiable_labs_envs import load_environment

    # Calibration warm-up triggers numpy edge-case warnings (divide-by-zero
    # on degenerate seeds); they don't affect info output, suppress them.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            env = load_environment(args.env, calibration_quantile=2.0)
        except (TypeError, KeyError):
            try:
                env = load_environment(args.env)
            except Exception as e:  # noqa: BLE001
                print(f"Error loading {args.env!r}: {e}", file=sys.stderr)
                return 2
        except Exception as e:  # noqa: BLE001
            print(f"Error loading {args.env!r}: {e}", file=sys.stderr)
            return 2

    name = getattr(env, "name", args.env)
    domain = getattr(env, "domain", "—")
    description = getattr(env, "description", "")
    target_coverage = getattr(env, "target_coverage", 0.90)

    print(f"{args.env}")
    print(f"  name:      {name}")
    print(f"  domain:    {domain}")
    print(f"  coverage:  target {target_coverage}")
    if description:
        print(f"  about:     {description}")
    return 0


# ── verifiable login (Phase 5 stub) ───────────────────────────────


def cmd_login(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Interactive API-key setup → ~/.verifiable/config.toml.

    Implementation lives in verifiable_labs.config so it can be reused
    by both the CLI and Python users who call ``configure()`` directly.
    """
    from verifiable_labs.config import interactive_setup

    return interactive_setup()


# ── verifiable run --env X --model Y --episodes N --seed N ────────


def cmd_run(args: argparse.Namespace) -> int:
    import warnings

    envs = _list_envs()
    if args.env not in envs:
        print(f"Error: unknown environment {args.env!r}.", file=sys.stderr)
        print(f"Run `verifiable list` to see all {len(envs)} envs.", file=sys.stderr)
        return 2

    base_url, api_key_var = _resolve_provider(args.model)
    api_key = _require_api_key(api_key_var, args.model)

    # The OpenAICompatibleAgent reads OPENAI_API_KEY / OPENAI_BASE_URL from env,
    # so we route the user's provider-specific key into those names just for
    # this process.
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url

    # Lazy-import the heavy deps only when actually running.
    from verifiable_labs_envs import load_environment
    from verifiable_labs_envs.agents import OpenAICompatibleAgent
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError, get_adapter
    from verifiable_labs_envs.traces import FailureType, Trace, hash_payload, write_jsonl

    # Numpy edge-case warnings (degenerate seeds during calibration / scoring)
    # don't affect the friendly output; users shouldn't see them.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(f"✓ Loading environment: {args.env}")
    try:
        env = load_environment(args.env, calibration_quantile=2.0)
    except (TypeError, KeyError):
        env = load_environment(args.env)

    target_coverage = getattr(env, "target_coverage", 0.90)
    print(f"✓ Calibrating conformal interval (target {target_coverage:.2f})...")
    print(f"✓ Running {args.episodes} episodes...")
    print()

    adapter = get_adapter(args.env)
    agent = OpenAICompatibleAgent.from_env(model=args.model)

    rewards: list[float] = []
    conformal_hits: list[float] = []  # per-episode conformal coverage component
    traces: list[Trace] = []
    cost_usd = 0.0
    t0 = time.perf_counter()

    for i in range(args.episodes):
        seed = args.seed + i
        instance = env.generate_instance(seed=seed)
        observation = {
            "env_id": args.env,
            "seed": int(seed),
            "env_kwargs": {},
            "system_prompt": getattr(adapter, "system_prompt", "") or "",
            "prompt_text": adapter.build_user_prompt(instance),
            "inputs": _safe_inputs(instance),
        }
        obs_hash = hash_payload(observation.get("inputs", {}))

        ep_t0 = time.perf_counter()
        ftype = FailureType.NONE
        reward = 0.0
        coverage: float | None = None
        components: dict[str, float] = {}

        try:
            prediction = agent.solve(observation)
        except TimeoutError as e:
            ftype = FailureType.TIMEOUT
            traces.append(_failure_trace(args.env, agent.name, seed, ep_t0, obs_hash, ftype, str(e)))
            continue
        except Exception as e:  # noqa: BLE001
            ftype = FailureType.UNKNOWN
            traces.append(_failure_trace(args.env, agent.name, seed, ep_t0, obs_hash, ftype, str(e)))
            continue

        latency_ms = (time.perf_counter() - ep_t0) * 1000.0
        if not isinstance(prediction, dict):
            traces.append(
                _failure_trace(
                    args.env, agent.name, seed, ep_t0, obs_hash,
                    FailureType.INVALID_SHAPE,
                    f"agent returned {type(prediction).__name__}",
                )
            )
            continue

        # Cost: prefer agent-supplied estimate, else compute from tokens.
        ep_cost = float(prediction.get("_cost_usd", 0.0) or 0.0)
        if ep_cost == 0.0:
            pt = int(prediction.get("_prompt_tokens", 0) or 0)
            ct = int(prediction.get("_completion_tokens", 0) or 0)
            ep_cost = _estimate_cost(args.model, pt, ct)
        cost_usd += ep_cost

        try:
            answer_text = prediction.get("answer_text") or json.dumps(_strip_internals(prediction))
            parsed_pred = adapter.parse_response(answer_text, instance)
        except LLMSolverError as e:
            ftype = FailureType.PARSE_ERROR
            traces.append(_failure_trace(args.env, agent.name, seed, ep_t0, obs_hash, ftype, str(e)[:300]))
            continue
        except Exception as e:  # noqa: BLE001
            ftype = FailureType.SCORING_ERROR
            traces.append(_failure_trace(args.env, agent.name, seed, ep_t0, obs_hash, ftype, str(e)[:300]))
            continue

        try:
            score = env.score(parsed_pred, instance)
        except Exception as e:  # noqa: BLE001
            ftype = FailureType.SCORING_ERROR
            traces.append(_failure_trace(args.env, agent.name, seed, ep_t0, obs_hash, ftype, str(e)[:300]))
            continue

        reward = float(score.get("reward", 0.0))
        components = _to_float_dict(score.get("components"))
        cov = score.get("coverage")
        if cov is not None:
            try:
                coverage = float(cov)
            except (TypeError, ValueError):
                coverage = None
        # Empirical coverage on this episode = conformal component (1.0 if the
        # ground truth is inside the interval, 0.0 otherwise). Mean across
        # episodes is the headline coverage figure on the website.
        if "conformal" in components:
            conformal_hits.append(components["conformal"])

        rewards.append(reward)

        traces.append(
            Trace.new(
                env_name=args.env,
                agent_name=agent.name,
                reward=reward,
                parse_success=True,
                seed=seed,
                model_name=args.model,
                observation_hash=obs_hash,
                prediction_hash=hash_payload(_strip_internals(prediction)),
                reward_components=components,
                coverage=coverage,
                latency_ms=latency_ms,
                failure_type=ftype,
                metadata={"cli": "verifiable-labs", "model": args.model},
            )
        )

    elapsed = time.perf_counter() - t0

    # ── Friendly summary ──
    if rewards:
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    else:
        mean = std = 0.0

    if conformal_hits:
        coverage_avg = statistics.fmean(conformal_hits)
        cov_check = "✓" if abs(coverage_avg - target_coverage) <= 0.05 else "✗"
        cov_line = f"Coverage: {coverage_avg:.3f} (target {target_coverage:.2f}) {cov_check}"
    else:
        cov_line = f"Coverage: — (target {target_coverage:.2f})"

    mins, secs = divmod(int(elapsed), 60)
    if cost_usd <= 0:
        cost_str = "—"
    elif cost_usd < 0.01:
        cost_str = f"${cost_usd:.4f}"
    else:
        cost_str = f"${cost_usd:.2f}"

    print(f"Mean reward: {mean:.3f} ± {std:.3f}")
    print(cov_line)
    print(f"Time: {mins}m {secs:02d}s · Cost: {cost_str}")

    # ── Persist trace ──
    # Replace path separators in the model name (OpenRouter uses provider/model)
    # so the filename can't accidentally create a subdirectory.
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = _runs_dir() / f"{args.env}_{safe_model}_{ts}.jsonl"
    write_jsonl(traces, out_path)
    print()
    print(f"Trace saved to {out_path}")

    failed = [t for t in traces if not t.parse_success]
    if failed and not rewards:
        # Every episode failed — exit non-zero so CI catches it.
        print(f"\n⚠ All {len(traces)} episodes failed.", file=sys.stderr)
        return 1
    return 0


# ── Failure helpers (kept private to this module) ─────────────────


def _failure_trace(
    env_id: str,
    agent_name: str,
    seed: int,
    started_at: float,
    obs_hash: str,
    ftype: Any,
    msg: str,
) -> Any:
    from verifiable_labs_envs.traces import Trace

    return Trace.new(
        env_name=env_id,
        agent_name=agent_name,
        reward=0.0,
        parse_success=False,
        seed=seed,
        observation_hash=obs_hash,
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
        failure_type=ftype,
        metadata={"error": msg[:300]},
    )


def _safe_inputs(instance: Any) -> dict[str, Any]:
    """JSON-safe view of ``instance.as_inputs()`` — mirrors the dev CLI."""
    if not hasattr(instance, "as_inputs"):
        return {}
    raw = instance.as_inputs()
    out: dict[str, Any] = {k: _to_json_safe(v) for k, v in raw.items()}
    shape = out.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        out["h"] = int(shape[0])
        out["w"] = int(shape[1])
    return out


def _to_json_safe(v: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            return {"_complex": True, "re": v.real.tolist(), "im": v.imag.tolist()}
        return v.tolist()
    if isinstance(v, complex):
        return {"_complex": True, "re": v.real, "im": v.imag}
    if isinstance(v, (list, tuple)):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_json_safe(x) for k, x in v.items()}
    return v


def _to_float_dict(d: Any) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _strip_internals(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_") and k != "answer_text"}


# ── Argparse wiring (friendly surface) ────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verifiable",
        description="Verifiable Labs CLI — evaluate frontier LLMs on conformal-calibrated scientific RL environments.",
    )
    p.add_argument("--version", action="version", version=f"verifiable-labs {__version__}")
    sub = p.add_subparsers(dest="command", required=False)

    p_list = sub.add_parser("list", help="list available environments")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="show environment details")
    p_info.add_argument("--env", required=True, help="environment slug (see `verifiable list`)")
    p_info.set_defaults(func=cmd_info)

    p_run = sub.add_parser("run", help="run a benchmark on an environment")
    p_run.add_argument("--env", required=True, help="environment slug")
    p_run.add_argument(
        "--model",
        required=True,
        help="model name, e.g. claude-haiku-4.5, gpt-4o-mini, gemini-2.5-pro",
    )
    p_run.add_argument("--episodes", type=int, default=10, help="number of episodes (default 10)")
    p_run.add_argument("--seed", type=int, default=42, help="first seed (default 42)")
    p_run.set_defaults(func=cmd_run)

    p_login = sub.add_parser("login", help="interactive API key setup")
    p_login.set_defaults(func=cmd_login)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `verifiable`. Friendly surface here, dev surface
    delegated to ``verifiable_labs_envs.cli.main``."""
    raw = list(sys.argv[1:] if argv is None else argv)

    # No args → show friendly help.
    if not raw:
        _build_parser().print_help()
        return 0

    head = raw[0]

    # Pass-through for dev/legacy subcommands so existing workflows still work.
    if head in _LEGACY:
        from verifiable_labs_envs.cli import main as dev_main

        return int(dev_main(raw))

    # `run --agent FILE --out F.jsonl ...` → legacy path; delegate.
    if head == "run" and ("--agent" in raw or "--out" in raw):
        from verifiable_labs_envs.cli import main as dev_main

        return int(dev_main(raw))

    # Friendly subcommands handled here.
    if head in _FRIENDLY or head.startswith("-"):
        parser = _build_parser()
        args = parser.parse_args(raw)
        if args.command is None:
            parser.print_help()
            return 0
        return int(args.func(args))

    # Unknown subcommand → try dev CLI as final fallback.
    try:
        from verifiable_labs_envs.cli import main as dev_main

        return int(dev_main(raw))
    except SystemExit as e:
        return int(e.code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
