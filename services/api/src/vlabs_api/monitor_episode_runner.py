"""Episode runner adapter for monitor runs (Phase 28.C).

PHASE_28_PLAN.md §9: rather than shelling out to ``verifiable run``
the way ``vlabs_audit.runner.default_episode_run`` does, the monitor
worker calls the customer's LLM endpoint in-process via
:func:`vlabs_api.llm_client.call_llm` and scores via
:func:`vlabs_api.scoring.score_completion`.

This keeps **all** scoring through the existing
``env.score(prediction, instance)`` kernel — there is no new reward
path, no drift between monitor runs and the synchronous ``/v1/score``
endpoint. The monitor run's per-episode reward distribution is
exactly what the customer would see if they hit ``/v1/score`` for
each (env, seed) pair manually.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MonitorEpisodeResult:
    """Outcome of a single (env_id, seed) episode call."""

    env_id: str
    seed: int
    reward: float
    components: dict[str, float]
    coverage: bool
    cost_usd_estimate: float
    success: bool
    error: str | None = None


def _build_prompt(env_id: str, instance: Any) -> tuple[str, str]:
    """Resolve env adapter, render system + user prompt strings."""
    from verifiable_labs_envs.solvers import adapters  # noqa: F401  (registers)
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS, get_adapter

    if env_id not in _ADAPTERS:
        return "", str(getattr(instance, "prompt", ""))
    adapter = get_adapter(env_id)
    return adapter.system_prompt, adapter.build_user_prompt(instance)


def _parse_completion(env_id: str, instance: Any, completion_text: str) -> Any:
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS, get_adapter

    if env_id not in _ADAPTERS:
        return None
    adapter = get_adapter(env_id)
    return adapter.parse_response(completion_text, instance)


async def run_monitor_episode(
    *,
    env_id: str,
    seed: int,
    endpoint_url: str,
    api_key: str,
    model: str,
    http_client: httpx.AsyncClient | None = None,
) -> MonitorEpisodeResult:
    """Drive one (env_id, seed) episode against the customer endpoint."""
    from verifiable_labs_envs import load_environment

    from vlabs_api.llm_client import call_llm

    try:
        env = load_environment(env_id)
        instance = env.generate_instance(seed=seed)
    except Exception as exc:  # noqa: BLE001
        return MonitorEpisodeResult(
            env_id=env_id,
            seed=seed,
            reward=0.0,
            components={},
            coverage=False,
            cost_usd_estimate=0.0,
            success=False,
            error=f"instance: {type(exc).__name__}: {exc}",
        )

    system_prompt, user_prompt = _build_prompt(env_id, instance)
    llm_result = await call_llm(
        endpoint_url=endpoint_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        client=http_client,
    )

    if not llm_result.success:
        return MonitorEpisodeResult(
            env_id=env_id,
            seed=seed,
            reward=0.0,
            components={},
            coverage=False,
            cost_usd_estimate=float(llm_result.cost_usd_estimate),
            success=False,
            error=llm_result.error,
        )

    try:
        prediction = _parse_completion(
            env_id, instance, llm_result.completion_text
        )
        scored = env.score(prediction, instance)
    except Exception as exc:  # noqa: BLE001
        return MonitorEpisodeResult(
            env_id=env_id,
            seed=seed,
            reward=0.0,
            components={},
            coverage=False,
            cost_usd_estimate=float(llm_result.cost_usd_estimate),
            success=False,
            error=f"score: {type(exc).__name__}: {exc}",
        )

    components = {
        k: float(v) for k, v in (scored.get("components") or {}).items()
    }
    meta = scored.get("meta") or {}
    coverage_flag = bool(meta.get("covered", False))
    return MonitorEpisodeResult(
        env_id=env_id,
        seed=seed,
        reward=float(scored.get("reward", 0.0)),
        components=components,
        coverage=coverage_flag,
        cost_usd_estimate=float(llm_result.cost_usd_estimate),
        success=True,
    )


async def run_monitor_episodes(
    *,
    env_subset: list[str],
    episodes_per_env: int,
    endpoint_url: str,
    api_key: str,
    model: str,
    http_client: httpx.AsyncClient | None = None,
    seed_start: int = 0,
) -> list[MonitorEpisodeResult]:
    """Drive a full env-subset × episodes_per_env audit batch.

    Episodes within a single env run **sequentially** (per-env
    semaphore reuse keeps the imaging envs from saturating the
    machine); envs run sequentially in the registered order so the
    reward distribution per env is contiguous in the output list.
    """
    results: list[MonitorEpisodeResult] = []
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient(timeout=60.0)
    try:
        for env_id in env_subset:
            for episode_idx in range(int(episodes_per_env)):
                seed = int(seed_start) + episode_idx
                outcome = await run_monitor_episode(
                    env_id=env_id,
                    seed=seed,
                    endpoint_url=endpoint_url,
                    api_key=api_key,
                    model=model,
                    http_client=http_client,
                )
                results.append(outcome)
                # Tiny yield so a hot loop doesn't starve the scheduler tick.
                await asyncio.sleep(0)
    finally:
        if own_client and http_client is not None:
            await http_client.aclose()
    return results


def compute_run_summary(results: list[MonitorEpisodeResult]) -> dict[str, Any]:
    """Aggregate per-episode results into the per-env summary dict.

    Shape used by the regression module (Phase 28.D) and by the PDF
    renderer (:mod:`vlabs_api.monitor_pdf`)::

        {
          "per_env": {
            env_id: {"n", "mean_reward", "std_reward", "coverage",
                     "rewards": [...]}, ...
          },
          "overall_mean_reward": float,
          "overall_coverage": float,
          "n_total": int,
          "n_success": int,
          "cost_usd_estimate": float,
        }
    """
    if not results:
        return {
            "per_env": {},
            "overall_mean_reward": None,
            "overall_coverage": None,
            "n_total": 0,
            "n_success": 0,
            "cost_usd_estimate": 0.0,
        }

    per_env: dict[str, dict[str, Any]] = {}
    cost_total = 0.0
    n_success = 0
    all_rewards: list[float] = []
    all_coverage: list[float] = []
    for r in results:
        cost_total += r.cost_usd_estimate
        if r.success:
            n_success += 1
        bucket = per_env.setdefault(
            r.env_id, {"rewards": [], "coverage_flags": []}
        )
        bucket["rewards"].append(r.reward)
        bucket["coverage_flags"].append(1.0 if r.coverage else 0.0)
        all_rewards.append(r.reward)
        all_coverage.append(1.0 if r.coverage else 0.0)

    summary_per_env: dict[str, dict[str, Any]] = {}
    for env_id, bucket in per_env.items():
        rewards = bucket["rewards"]
        cov_flags = bucket["coverage_flags"]
        n = len(rewards)
        mean = float(sum(rewards) / n) if n else 0.0
        var = float(
            sum((r - mean) ** 2 for r in rewards) / n
        ) if n else 0.0
        std = var ** 0.5
        coverage = float(sum(cov_flags) / n) if n else 0.0
        summary_per_env[env_id] = {
            "n": int(n),
            "mean_reward": mean,
            "std_reward": std,
            "coverage": coverage,
            "rewards": list(rewards),
            "coverage_flags": list(cov_flags),
        }

    overall_mean = (
        float(sum(all_rewards) / len(all_rewards)) if all_rewards else None
    )
    overall_cov = (
        float(sum(all_coverage) / len(all_coverage))
        if all_coverage
        else None
    )

    return {
        "per_env": summary_per_env,
        "overall_mean_reward": overall_mean,
        "overall_coverage": overall_cov,
        "n_total": len(results),
        "n_success": n_success,
        "cost_usd_estimate": float(cost_total),
    }


__all__ = [
    "MonitorEpisodeResult",
    "run_monitor_episode",
    "run_monitor_episodes",
    "compute_run_summary",
]
