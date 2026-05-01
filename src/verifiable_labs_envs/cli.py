"""``verifiable`` — single-binary developer CLI for Verifiable Labs.

Subcommands:
  envs           list available envs
  run            drive an agent through N episodes; write JSONL traces
  compare        side-by-side comparison of two run files
  report         render a Markdown evaluation report from a JSONL run
  init-env       scaffold a new custom env (wraps scripts/create_env.py)
  validate-env   validate a custom env (wraps scripts/validate_env.py)

The CLI is deliberately argparse-only — zero new deps on Click / Typer
so the install path stays minimal. Subcommand functions live in this
file and call into:

- :mod:`verifiable_labs_envs.traces` — JSONL trace schema
- :mod:`verifiable_labs_envs.agents` — agent loaders
- :mod:`verifiable_labs_envs.reporting` — Markdown report writer (P4)
- ``scripts/create_env.py``, ``scripts/validate_env.py`` — existing
  Tier-1 scaffold utilities

Entry point in ``pyproject.toml::[project.scripts]`` is
``verifiable = "verifiable_labs_envs.cli:main"``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from verifiable_labs_envs import __version__
from verifiable_labs_envs.agents import Agent, load_agent
from verifiable_labs_envs.repro import config_hash, instance_hash, reward_hash
from verifiable_labs_envs.traces import (
    FailureType,
    Trace,
    hash_payload,
    read_jsonl,
    write_jsonl,
)

# ── shared helpers ─────────────────────────────────────────────


def _list_envs() -> list[str]:
    """List env IDs from the central registry. Importing from
    ``verifiable_labs_envs`` is the single source of truth for what's
    available."""
    from verifiable_labs_envs import list_environments
    return list_environments()


def _load_env(env_id: str, env_kwargs: dict[str, Any]):
    """Wrap ``load_environment`` with env-specific defaults so the CLI
    works out of the box for every env. ``calibration_quantile=2.0``
    is the cheap calibration shortcut used by the integration tests."""
    from verifiable_labs_envs import load_environment
    kwargs = dict(env_kwargs or {})
    if "calibration_quantile" not in kwargs:
        # Sparse-Fourier-style envs accept this kwarg; image envs ignore unknowns.
        kwargs["calibration_quantile"] = 2.0
    try:
        return load_environment(env_id, **kwargs)
    except TypeError:
        # Env factory doesn't accept calibration_quantile; retry without it.
        kwargs.pop("calibration_quantile", None)
        return load_environment(env_id, **kwargs)


def _get_adapter(env_id: str):
    from verifiable_labs_envs.solvers.llm_solver import get_adapter
    return get_adapter(env_id)


def _safe_inputs(instance: Any) -> dict[str, Any]:
    """Convert ``instance.as_inputs()`` into a JSON-safe dict.

    NumPy arrays → lists (potentially nested). Complex numbers split
    into ``{re, im}`` pairs. Tuples preserved as lists. Includes
    ``h`` / ``w`` derived from ``shape`` for image envs.
    """
    if not hasattr(instance, "as_inputs"):
        return {}
    raw = instance.as_inputs()
    out: dict[str, Any] = {}
    for k, v in raw.items():
        out[k] = _to_json_safe(v)
    # Image envs commonly carry ``shape=(h, w)`` — surface dimensions
    # so simple agents (zero, random) can branch on shape.
    if "shape" in out and isinstance(out["shape"], (list, tuple)) and len(out["shape"]) == 2:
        out["h"] = int(out["shape"][0])
        out["w"] = int(out["shape"][1])
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


def _serialise_components(components: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not components:
        return out
    if isinstance(components, dict):
        for k, v in components.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


# ── timeout + retry + redaction helpers (M4) ───────────────────


class _AgentTimeout(Exception):
    """Raised when an agent's solve() exceeds the per-episode wall-clock budget."""


@contextlib.contextmanager
def _alarm_timeout(seconds: int) -> Iterator[None]:
    """Wrap a block in a SIGALRM-based wall-clock timeout.

    NOTE: signal.SIGALRM-based timeout. Single-threaded only.
    If verifiable run is parallelized in the future (e.g. --workers N),
    switch to subprocess-based timeout or threading.Timer + check pattern.
    """
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise _AgentTimeout(f"agent exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


_REDACT_KEY_RE = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|AUTH)", re.IGNORECASE)
_REDACT_INLINE_RE = re.compile(
    r"(\b\w*(?:API_KEY|TOKEN|SECRET|PASSWORD|AUTH)\w*\s*[:=]\s*)"
    r"([^\s\"',]+)",
    re.IGNORECASE,
)


def _redact_secrets(text: str) -> str:
    """Strip secret-shaped values from a single-line message.

    1. Replace any value of an env var whose key matches
       ``API_KEY|TOKEN|SECRET|PASSWORD|AUTH`` (case-insensitive) and
       has length >= 8 with ``[REDACTED]``.
    2. Replace inline ``API_KEY=xxx`` / ``token: xxx`` patterns.

    The function never raises; on any unexpected input it returns the
    original text unchanged.
    """
    if not text:
        return text
    try:
        result = text
        for key, val in os.environ.items():
            if not val or len(val) < 8:
                continue
            if _REDACT_KEY_RE.search(key):
                result = result.replace(val, "[REDACTED]")
        result = _REDACT_INLINE_RE.sub(r"\1[REDACTED]", result)
        return result
    except Exception:  # noqa: BLE001 — redaction must never break the run
        return text


def _summarise_error(exc: BaseException) -> str:
    """Single-line, redacted, length-capped error message for trace metadata.

    Stack traces NEVER end up in JSONL — they belong on stderr only.
    """
    msg = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ")
    return _redact_secrets(msg)[:200]


# ── subcommand: envs ────────────────────────────────────────────


def cmd_envs(args: argparse.Namespace) -> int:
    envs = _list_envs()
    if args.format == "json":
        print(json.dumps(envs, indent=2))
        return 0
    print(f"{len(envs)} environments available:\n")
    for env_id in envs:
        print(f"  {env_id}")
    return 0


# ── subcommand: run ─────────────────────────────────────────────


def _parse_env_kwargs(raw: list[str] | None) -> dict[str, Any]:
    """Parse ``--env-kwarg key=value`` pairs. Values are JSON-decoded
    when possible (so ``--env-kwarg max_turns=3`` becomes int 3)."""
    out: dict[str, Any] = {}
    for entry in raw or []:
        if "=" not in entry:
            raise SystemExit(f"--env-kwarg requires key=value, got {entry!r}")
        k, _, v = entry.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _build_observation(
    env_id: str, seed: int, env_kwargs: dict[str, Any], adapter: Any, instance: Any
) -> dict[str, Any]:
    return {
        "env_id": env_id,
        "seed": int(seed),
        "env_kwargs": dict(env_kwargs),
        "system_prompt": getattr(adapter, "system_prompt", "") or "",
        "prompt_text": adapter.build_user_prompt(instance),
        "inputs": _safe_inputs(instance),
    }


def _score_episode(
    *,
    env: Any,
    adapter: Any,
    instance: Any,
    prediction: dict[str, Any],
    seed: int,
    env_kwargs: dict[str, Any],
) -> tuple[float, dict[str, float], float | None, FailureType, dict[str, Any]]:
    """Score one (env, instance, prediction) tuple. Returns
    ``(reward, components, coverage, failure_type, extras)``."""
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError

    # Sentinel: simple-baseline agent → run the env's classical baseline.
    if prediction.get("__classical_baseline__"):
        score = env.run_baseline(seed=seed, **env_kwargs) if env_kwargs else env.run_baseline(seed=seed)
        return (
            float(score.get("reward", 0.0)),
            _serialise_components(score.get("components")),
            _opt_float(score.get("coverage")),
            FailureType.NONE,
            {"_classical_baseline": True},
        )

    answer_text = prediction.get("answer_text") or json.dumps(_strip_internals(prediction))
    try:
        pred = adapter.parse_response(answer_text, instance)
    except LLMSolverError as e:
        return 0.0, {}, None, FailureType.PARSE_ERROR, {"parse_error": str(e)[:300]}
    except Exception as e:  # noqa: BLE001 — best-effort scoring boundary
        return 0.0, {}, None, FailureType.SCORING_ERROR, {"error": str(e)[:300]}

    try:
        score = env.score(pred, instance)
    except Exception as e:  # noqa: BLE001
        return 0.0, {}, None, FailureType.SCORING_ERROR, {"error": str(e)[:300]}

    return (
        float(score.get("reward", 0.0)),
        _serialise_components(score.get("components")),
        _opt_float(score.get("coverage")),
        FailureType.NONE,
        {},
    )


def _strip_internals(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys agents use internally (``_latency_ms``, ``_fake``, etc.)."""
    return {k: v for k, v in d.items() if not k.startswith("_") and k != "answer_text"}


def _opt_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def cmd_run(args: argparse.Namespace) -> int:
    env_kwargs = _parse_env_kwargs(args.env_kwarg)
    out_path = Path(args.out)

    # M3 reproducibility metadata: config_hash is per-run; instance_hash is per-episode.
    _env_version = __version__
    _cfg_hash = config_hash({
        "env_id": args.env,
        "agent_id": args.agent,
        "n_episodes": int(args.n),
        "start_seed": int(args.start_seed),
        "env_kwargs": env_kwargs,
        "with_baseline": bool(args.with_baseline),
    })

    try:
        env = _load_env(args.env, env_kwargs)
    except KeyError:
        print(f"ERROR: unknown env {args.env!r}. Available: {_list_envs()}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not load env {args.env!r}: {e}", file=sys.stderr)
        return 2

    try:
        adapter = _get_adapter(args.env)
    except KeyError:
        print(f"ERROR: no adapter registered for env {args.env!r}", file=sys.stderr)
        return 2

    try:
        agent: Agent = load_agent(args.agent)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: could not load agent {args.agent!r}: {e}", file=sys.stderr)
        return 2

    seeds = list(range(args.start_seed, args.start_seed + args.n))
    traces: list[Trace] = []

    if not args.quiet:
        print(f"verifiable run · env={args.env} agent={agent.name} n={args.n} → {out_path}",
              flush=True)

    for seed in seeds:
        instance = env.generate_instance(seed=seed)
        observation = _build_observation(args.env, seed, env_kwargs, adapter, instance)
        obs_hash = hash_payload(observation.get("inputs", {}))
        _inst_hash = instance_hash(args.env, _env_version, seed, env_kwargs)

        # Retry loop wraps agent.solve only — timeouts/exceptions are
        # non-deterministic and worth retrying. parse_error / invalid_schema
        # come AFTER solve and are deterministic; they don't retry.
        prediction: Any = None
        last_exc: BaseException | None = None
        last_ftype = FailureType.NONE
        retries_done = 0
        t0 = time.perf_counter()
        for attempt in range(int(args.max_retries) + 1):
            retries_done = attempt
            try:
                with _alarm_timeout(int(args.timeout_seconds)):
                    prediction = agent.solve(observation)
                break
            except _AgentTimeout as e:
                last_exc = e
                last_ftype = FailureType.TIMEOUT
            except Exception as e:  # noqa: BLE001 — agent boundary
                last_exc = e
                last_ftype = FailureType.UNKNOWN
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if prediction is None:
            trace = _failure_trace(
                env_id=args.env, agent_name=agent.name, seed=seed,
                latency_ms=latency_ms, obs_hash=obs_hash,
                ftype=last_ftype,
                msg=_summarise_error(last_exc) if last_exc else last_ftype.value,
                retries=retries_done,
                cfg_hash=_cfg_hash, inst_hash=_inst_hash,
            )
        elif not isinstance(prediction, dict):
            trace = _failure_trace(
                env_id=args.env, agent_name=agent.name, seed=seed,
                latency_ms=latency_ms, obs_hash=obs_hash,
                ftype=FailureType.INVALID_SHAPE,
                msg=f"agent returned {type(prediction).__name__}, expected dict",
                retries=retries_done,
                cfg_hash=_cfg_hash, inst_hash=_inst_hash,
            )
        else:
            baseline_reward = None
            if args.with_baseline:
                try:
                    base_score = (
                        env.run_baseline(seed=seed, **env_kwargs)
                        if env_kwargs else env.run_baseline(seed=seed)
                    )
                    baseline_reward = float(base_score.get("reward", 0.0))
                except Exception:  # noqa: BLE001
                    baseline_reward = None

            reward, components, coverage, ftype, extras = _score_episode(
                env=env, adapter=adapter, instance=instance, prediction=prediction,
                seed=seed, env_kwargs=env_kwargs,
            )

            # Pull agent-side latency override if present (e.g. OpenAICompatibleAgent).
            if "_latency_ms" in prediction:
                with contextlib.suppress(TypeError, ValueError):
                    latency_ms = float(prediction["_latency_ms"])

            gap = (reward - baseline_reward) if baseline_reward is not None else None
            success = ftype == FailureType.NONE
            metadata: dict[str, Any] = {
                "status": "ok" if success else "failed",
                "retries": int(retries_done),
                "config_hash": _cfg_hash,
                "instance_hash": _inst_hash,
                "reward_hash": reward_hash(reward),
            }
            if not success:
                err = (extras or {}).get("parse_error") or (extras or {}).get("error") or ""
                metadata["error_message"] = _redact_secrets(str(err))[:200]
            if env_kwargs:
                metadata["env_kwargs"] = env_kwargs
            for k, v in (extras or {}).items():
                if k not in metadata:
                    metadata[k] = v

            trace = Trace.new(
                env_name=args.env,
                agent_name=agent.name,
                reward=reward,
                parse_success=success,
                seed=seed,
                model_name=getattr(agent, "model", None),
                observation_hash=obs_hash,
                prediction_hash=hash_payload(_strip_internals(prediction)),
                reward_components=components,
                classical_baseline_reward=baseline_reward,
                gap_to_classical=gap,
                coverage=coverage,
                latency_ms=latency_ms,
                failure_type=ftype,
                metadata=metadata,
            )

        traces.append(trace)
        if not args.quiet:
            if trace.failure_type == FailureType.NONE:
                gap_s = (
                    f" Δ={trace.gap_to_classical:+.3f}"
                    if trace.gap_to_classical is not None else ""
                )
                print(
                    f"  seed={seed:>5d} reward={trace.reward:.3f} ok{gap_s}",
                    flush=True,
                )
            else:
                tag = trace.failure_type.value.upper()
                rs = f" retries={retries_done}" if retries_done else ""
                print(f"  seed={seed:>5d} {tag}{rs}", flush=True)

        if args.fail_fast and trace.failure_type != FailureType.NONE:
            n_written = write_jsonl(traces, out_path)
            if not args.quiet:
                print(
                    f"\n[--fail-fast] aborted after seed={seed} "
                    f"({trace.failure_type.value}); wrote {n_written} traces",
                    file=sys.stderr,
                )
            return 1

    n_written = write_jsonl(traces, out_path)
    if not args.quiet:
        rewards = [t.reward for t in traces if t.parse_success]
        mean = statistics.fmean(rewards) if rewards else 0.0
        ok = sum(1 for t in traces if t.parse_success)
        print(f"\nwrote {n_written} traces → {out_path}  (parse_ok={ok}/{n_written}  mean={mean:.3f})",
              flush=True)
    return 0


def _failure_trace(
    *,
    env_id: str,
    agent_name: str,
    seed: int,
    latency_ms: float,
    obs_hash: str,
    ftype: FailureType,
    msg: str,
    retries: int = 0,
    cfg_hash: str | None = None,
    inst_hash: str | None = None,
) -> Trace:
    """Build a failure-shaped Trace with uniform M3+M4 metadata.

    Schema parity with the success path: every trace carries
    ``status``, ``retries``, ``config_hash``, ``instance_hash``,
    ``reward_hash`` in its ``metadata`` dict. ``error_message`` is
    redacted (env-var-shaped secrets stripped) and capped at 200 chars.
    Stack traces never appear here — they belong on stderr only.
    """
    metadata: dict[str, Any] = {
        "status": "failed",
        "retries": int(retries),
        "error_message": _redact_secrets(msg or "")[:200],
        "reward_hash": reward_hash(0.0),
    }
    if cfg_hash is not None:
        metadata["config_hash"] = cfg_hash
    if inst_hash is not None:
        metadata["instance_hash"] = inst_hash
    return Trace.new(
        env_name=env_id,
        agent_name=agent_name,
        reward=0.0,
        parse_success=False,
        seed=seed,
        observation_hash=obs_hash,
        latency_ms=latency_ms,
        failure_type=ftype,
        metadata=metadata,
    )


# ── subcommand: report ──────────────────────────────────────────


def cmd_report(args: argparse.Namespace) -> int:
    from verifiable_labs_envs.reporting import render_run_report
    traces = read_jsonl(args.run)
    if not traces:
        print(f"ERROR: {args.run} contained no traces", file=sys.stderr)
        return 2
    out_path = Path(args.out)
    render_run_report(traces, out_path)
    if not args.quiet:
        print(f"wrote report → {out_path}")
    return 0


# ── subcommand: compare ─────────────────────────────────────────


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    for path in args.runs:
        traces = read_jsonl(path)
        if not traces:
            print(f"ERROR: {path} contained no traces", file=sys.stderr)
            return 2
        rows.append(_summary_row(path, traces))

    # Pretty-print table.
    headers = ["run", "env", "agent", "n", "mean", "std", "parse_ok%", "Δ_classical", "latency_ms"]
    widths = [max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers]
    sep = "  ".join("-" * w for w in widths)
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print(sep)
    for r in rows:
        cells = [str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths, strict=True)]
        print("  ".join(cells))
    return 0


def _summary_row(path: str | Path, traces: list[Trace]) -> dict[str, Any]:
    rewards = [t.reward for t in traces if t.parse_success]
    parse_ok = sum(1 for t in traces if t.parse_success)
    gaps = [t.gap_to_classical for t in traces if t.gap_to_classical is not None]
    latencies = [t.latency_ms for t in traces if t.latency_ms is not None]
    return {
        "run": Path(path).name,
        "env": traces[0].env_name,
        "agent": traces[0].agent_name,
        "n": len(traces),
        "mean": f"{statistics.fmean(rewards):.3f}" if rewards else "—",
        "std": f"{statistics.pstdev(rewards):.3f}" if len(rewards) > 1 else "—",
        "parse_ok%": f"{parse_ok / len(traces) * 100:.0f}",
        "Δ_classical": f"{statistics.fmean(gaps):+.3f}" if gaps else "—",
        "latency_ms": f"{statistics.fmean(latencies):.0f}" if latencies else "—",
    }


# ── subcommand: init-env ────────────────────────────────────────


def cmd_init_env(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "create_env.py"
    if not script.exists():
        print(f"ERROR: scaffold script not found: {script}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script), args.env_id, "--domain", args.domain]
    if args.target:
        cmd.extend(["--target", args.target])
    if args.force:
        cmd.append("--force")
    return subprocess.run(cmd, check=False).returncode


# ── subcommand: validate-env ────────────────────────────────────


def cmd_validate_env(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "validate_env.py"
    if not script.exists():
        print(f"ERROR: validator script not found: {script}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script), args.path]
    if args.skip_adapter_check:
        cmd.append("--skip-adapter-check")
    if args.n_cal is not None:
        cmd.extend(["--n-cal", str(args.n_cal)])
    if args.tolerance is not None:
        cmd.extend(["--tolerance", str(args.tolerance)])
    return subprocess.run(cmd, check=False).returncode


# ── argparse wiring ─────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verifiable",
        description="Developer CLI for evaluating scientific AI agents on verifiable RL environments.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # envs
    p_envs = sub.add_parser("envs", help="list available environments")
    p_envs.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="output format (default: text)",
    )
    p_envs.set_defaults(func=cmd_envs)

    # run
    p_run = sub.add_parser("run", help="run an agent against an env, write JSONL traces")
    p_run.add_argument("--env", required=True, help="env id (see `verifiable envs`)")
    p_run.add_argument(
        "--agent", required=True,
        help="path/to/agent.py | cmd:<command> | openai:<model>",
    )
    p_run.add_argument("--n", type=int, default=5, help="number of episodes (default 5)")
    p_run.add_argument(
        "--start-seed", type=int, default=0,
        help="first seed; episodes use start..start+n-1 (default 0)",
    )
    p_run.add_argument("--out", required=True, help="output JSONL path")
    p_run.add_argument(
        "--env-kwarg", action="append", default=[],
        help="extra env kwarg as key=value (JSON-decoded if possible); repeatable",
    )
    p_run.add_argument(
        "--with-baseline", action="store_true",
        help="also run env.run_baseline(seed) per episode to record gap_to_classical",
    )
    p_run.add_argument(
        "--timeout-seconds", type=int, default=60,
        help="kill the agent after N seconds; failed episode is recorded as "
             "failure_type=timeout. Set 0 to disable. (default: 60)",
    )
    p_run.add_argument(
        "--max-retries", type=int, default=0,
        help="retry timeouts and exceptions up to N times before giving up. "
             "parse_error / invalid_schema do NOT retry (deterministic). (default: 0)",
    )
    p_run_fail = p_run.add_mutually_exclusive_group()
    p_run_fail.add_argument(
        "--continue-on-error", action="store_true",
        help="continue to the next episode after a failure (default behavior)",
    )
    p_run_fail.add_argument(
        "--fail-fast", action="store_true",
        help="abort the entire run on the first unrecoverable failure (exit code 1)",
    )
    p_run.add_argument("--quiet", action="store_true", help="suppress per-episode log lines")
    p_run.set_defaults(func=cmd_run)

    # compare
    p_cmp = sub.add_parser("compare", help="side-by-side compare of two or more runs")
    p_cmp.add_argument("--runs", nargs="+", required=True, help="JSONL run files")
    p_cmp.set_defaults(func=cmd_compare)

    # report
    p_rep = sub.add_parser("report", help="render a Markdown evaluation report")
    p_rep.add_argument("--run", required=True, help="JSONL run file")
    p_rep.add_argument("--out", required=True, help="output Markdown path")
    p_rep.add_argument("--quiet", action="store_true")
    p_rep.set_defaults(func=cmd_report)

    # init-env
    p_init = sub.add_parser(
        "init-env", help="scaffold a new custom env (wraps scripts/create_env.py)",
    )
    p_init.add_argument("env_id", help="kebab-case env id, e.g. seismic-fwi")
    p_init.add_argument(
        "--domain", default="custom-domain",
        help="short domain label written into the env metadata",
    )
    p_init.add_argument(
        "--target", default=None,
        help="output directory (default environments/<env_id>)",
    )
    p_init.add_argument("--force", action="store_true", help="overwrite existing target")
    p_init.set_defaults(func=cmd_init_env)

    # validate-env
    p_val = sub.add_parser(
        "validate-env", help="run the four-check validator (wraps scripts/validate_env.py)",
    )
    p_val.add_argument("path", help="path to env package")
    p_val.add_argument("--skip-adapter-check", action="store_true")
    p_val.add_argument(
        "--n-cal", type=int, default=None, dest="n_cal",
        help="number of fresh seeds for the calibration check (default 50)",
    )
    p_val.add_argument(
        "--tolerance", type=float, default=None,
        help="±tolerance on empirical coverage vs target (default 0.05)",
    )
    p_val.set_defaults(func=cmd_validate_env)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
