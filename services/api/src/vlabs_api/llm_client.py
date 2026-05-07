"""Customer-LLM-endpoint orchestration (Phase 23.C).

PHASE_23_PLAN.md §5.D1 ruling: the customer brings their own LLM
endpoint. This module is the thin HTTP shim that calls into that
endpoint using the OpenAI Chat Completions protocol — the de-facto
standard surface for OpenRouter, Anthropic-compatible proxies,
together.ai, vLLM-served models, and any provider that ships an
``/v1/chat/completions`` route.

Cost estimation for the per-job ``budget_usd_cap`` is **best-effort**:
we infer prompt tokens via a coarse character-count heuristic
(approx 4 chars / token) and look up per-1K-token rates from a small
in-memory price table. Customers who exceed their stated budget
because of a mis-estimated rate get a partial dataset, not an
overrun — the worker stops at the cap.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger(__name__)

# Per-1K-token USD pricing for cost estimation. Conservative estimates
# — better to under-charge the customer than to surprise them. Updated
# manually per provider rate changes; not authoritative billing.
_PRICE_TABLE_USD_PER_1K = {
    # OpenAI
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    # Anthropic via OpenRouter (anthropic/claude-haiku-4-5 etc.)
    "claude-haiku-4.5": {"prompt": 0.0008, "completion": 0.004},
    "claude-haiku-4-5": {"prompt": 0.0008, "completion": 0.004},
    "claude-sonnet-4.6": {"prompt": 0.003, "completion": 0.015},
    "claude-sonnet-4-6": {"prompt": 0.003, "completion": 0.015},
    "claude-opus-4.7": {"prompt": 0.015, "completion": 0.075},
    "claude-opus-4-7": {"prompt": 0.015, "completion": 0.075},
}
# Fallback rate for unknown models: rounds-up $1 / 1M tokens.
_FALLBACK_PRICE = {"prompt": 0.001, "completion": 0.002}

# Heuristic token count (good enough for cost estimation, not billing).
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class LLMResult:
    """Outcome of a single ``call`` to the customer's LLM endpoint."""

    completion_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd_estimate: float
    success: bool
    error: str | None = None


def _model_price(model: str) -> dict[str, float]:
    """Resolve per-1K-token rates for a model name (case-insensitive,
    OpenRouter prefix tolerated)."""
    key = model.lower()
    # Strip OpenRouter-style "anthropic/" / "openai/" prefixes.
    if "/" in key:
        key = key.split("/", 1)[1]
    return _PRICE_TABLE_USD_PER_1K.get(key, _FALLBACK_PRICE)


def _estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    rates = _model_price(model)
    return (
        prompt_tokens / 1000 * rates["prompt"]
        + completion_tokens / 1000 * rates["completion"]
    )


async def call_llm(
    *,
    endpoint_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float = 60.0,
    client: httpx.AsyncClient | None = None,
) -> LLMResult:
    """POST a chat completion to the customer's OpenAI-compatible endpoint.

    Returns an :class:`LLMResult` even on failure (``success=False``);
    the worker treats failed calls as "tuple skipped, do not increment
    counter" rather than raising. This keeps a single failed LLM call
    from killing a 100K-tuple job.

    Rates are best-effort estimates; ``cost_usd_estimate`` is
    accumulated against ``dataset_jobs.budget_usd_spent`` so the
    per-job spend cap kicks in deterministically.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Many providers expect the endpoint URL to be the base, with the
    # client appending /chat/completions; some pass the full path.
    # We accept either: if the URL doesn't already end in
    # /chat/completions, append it.
    url = endpoint_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout_s)

    try:
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            return LLMResult(
                completion_text="",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd_estimate=0.0,
                success=False,
                error=f"transport: {type(exc).__name__}: {exc}",
            )

        if resp.status_code >= 400:
            return LLMResult(
                completion_text="",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd_estimate=0.0,
                success=False,
                error=f"http_{resp.status_code}: {resp.text[:200]}",
            )

        try:
            data = resp.json()
        except ValueError as exc:
            return LLMResult(
                completion_text="",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd_estimate=0.0,
                success=False,
                error=f"json_decode: {exc}",
            )

        choices = data.get("choices") or []
        if not choices:
            return LLMResult(
                completion_text="",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd_estimate=0.0,
                success=False,
                error="no choices in response",
            )

        completion_text = (
            choices[0].get("message", {}).get("content") or ""
        )

        # Prefer provider-reported usage; fall back to char-count estimate.
        usage = data.get("usage") or {}
        prompt_tokens = int(
            usage.get("prompt_tokens")
            or _approx_tokens(system_prompt) + _approx_tokens(user_prompt)
        )
        completion_tokens = int(
            usage.get("completion_tokens") or _approx_tokens(completion_text)
        )

        return LLMResult(
            completion_text=completion_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd_estimate=_estimate_cost(
                model, prompt_tokens, completion_tokens
            ),
            success=True,
        )
    finally:
        if own_client:
            await client.aclose()


__all__ = [
    "LLMResult",
    "call_llm",
]
