"""Frontier-model judgment harness (Phase 29.B, plan §5 D5-D).

The frontier slice supplements env-procedural rewards with a second
opinion on **borderline** rows — those whose env reward sits in the
middle band ``(0.3, 0.7)`` where the env's procedural verifier is
genuinely uncertain (partial-credit syntheses, almost-correct SQL,
near-miss math). High-confidence wins (env_reward ≈ 1.0) and
high-confidence misses (env_reward ≈ 0.0) don't need a second opinion;
the env score is already trustworthy at the tails.

Design points:

- The judge endpoint defaults to OpenRouter
  (`https://openrouter.ai/api/v1/chat/completions`), model
  ``anthropic/claude-sonnet-4.6``. The caller is fully pluggable —
  tests inject a deterministic stub via ``judge_caller``; production
  uses :func:`openrouter_judge_caller` which wraps ``httpx.post``.
- Cost is capped per :doc:`PHASE_29_PLAN.md` §5 D1-D ($7.50 at the
  default 1500-prompt × $0.005 budget; the gate at $30 sits at the
  CLI level).
- Outputs land back as ``frontier_judgment`` / ``frontier_rationale``
  on a *new* :class:`RewardTrainingRow` (rows are immutable); the
  caller decides whether to merge them into the training set.

Privacy posture: only the prompt and completion text are sent to the
judge; nothing else. The judge model id is recorded in row metadata
so the audit trail is reproducible.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verifiable_labs_envs.reward_distillation.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    consensus_reward,
    disagreement,
)
from verifiable_labs_envs.reward_distillation.dataset import RewardTrainingRow

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL: str = "anthropic/claude-sonnet-4.6"
DEFAULT_BORDERLINE_LOW: float = 0.3
DEFAULT_BORDERLINE_HIGH: float = 0.7
DEFAULT_FRACTION: float = 0.10
DEFAULT_OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_S: float = 30.0
DEFAULT_TEMPERATURE: float = 0.0


JUDGE_SYSTEM_PROMPT: str = (
    "You are evaluating a single (prompt, completion) pair for a "
    "reward-distillation dataset. Score the completion's quality on a "
    "0-1 scale considering: factual correctness, instruction-following, "
    "helpfulness, and absence of hallucinations. Return strict JSON with "
    'shape {"score": float in [0,1], "rationale": short string}. Do not '
    "include any text outside the JSON object."
)


@dataclass(frozen=True)
class FrontierJudgeResult:
    """One judgment on one (prompt, completion) pair."""

    row_id: str
    score: float
    rationale: str
    judge_model: str
    raw_response: str
    parsed_ok: bool


JudgeCaller = Callable[[str, str, str, str], dict[str, Any]]
"""Signature: ``(prompt, completion, judge_model, api_key) -> response_dict``.

The response dict must follow the OpenAI/OpenRouter chat-completions
shape: ``{"choices": [{"message": {"content": "..."}}]}``."""


def is_borderline(
    env_reward: float | None,
    *,
    low: float = DEFAULT_BORDERLINE_LOW,
    high: float = DEFAULT_BORDERLINE_HIGH,
) -> bool:
    """Predicate: should this row be sent to the frontier judge?

    True iff ``env_reward`` is non-null and lies in the open interval
    ``(low, high)``. Pure tail rows (env_reward None or outside the
    window) are skipped — the env signal is already informative there.
    """
    if env_reward is None:
        return False
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"require 0 <= low < high <= 1; got low={low}, high={high}")
    return low < float(env_reward) < high


def select_borderline_rows(
    rows: Sequence[RewardTrainingRow],
    *,
    fraction: float = DEFAULT_FRACTION,
    low: float = DEFAULT_BORDERLINE_LOW,
    high: float = DEFAULT_BORDERLINE_HIGH,
    seed: int = 0,
    max_rows: int | None = None,
) -> list[RewardTrainingRow]:
    """Pick the subset of rows to send to the judge.

    Filters to ``is_borderline`` rows, then samples ``fraction`` of them
    (rounded up to at least 1 if any borderline rows exist) using a
    seeded numpy RNG. ``max_rows`` provides a hard upper bound; the CLI
    uses this to cap spend regardless of dataset size.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}")

    borderline = [r for r in rows if is_borderline(r.env_reward, low=low, high=high)]
    if not borderline:
        return []

    target = max(1, int(round(len(borderline) * fraction)))
    if max_rows is not None:
        target = min(target, int(max_rows))
    target = min(target, len(borderline))

    import numpy as np

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(borderline), size=target, replace=False)
    indices.sort()
    return [borderline[int(i)] for i in indices]


def sample_frontier_judgments(
    rows: Sequence[RewardTrainingRow],
    *,
    fraction: float = DEFAULT_FRACTION,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
    judge_caller: JudgeCaller | None = None,
    seed: int = 0,
    max_rows: int | None = None,
    low: float = DEFAULT_BORDERLINE_LOW,
    high: float = DEFAULT_BORDERLINE_HIGH,
    raise_on_error: bool = False,
) -> list[FrontierJudgeResult]:
    """Sample ``fraction`` of borderline rows and call the judge.

    The function returns a flat list of :class:`FrontierJudgeResult`;
    integrating those results back into rows is :func:`merge_judgments`.
    Splitting the two steps keeps the network-touching path narrow and
    independently testable.

    ``judge_caller`` defaults to :func:`openrouter_judge_caller` when
    ``api_key`` is set; tests pass an explicit deterministic stub.
    Either ``api_key`` or ``judge_caller`` must be supplied — this
    surfaces missing credentials early instead of failing N rows in.
    """
    if judge_caller is None:
        if not api_key:
            raise ValueError(
                "either judge_caller or api_key must be provided; "
                "set OPENROUTER_API_KEY or supply a stub for tests"
            )
        judge_caller = openrouter_judge_caller

    selected = select_borderline_rows(
        rows, fraction=fraction, low=low, high=high, seed=seed, max_rows=max_rows
    )
    results: list[FrontierJudgeResult] = []
    for row in selected:
        try:
            raw = judge_caller(row.prompt, row.completion, judge_model, api_key or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge call failed for %s: %s", row.row_id, exc)
            if raise_on_error:
                raise
            results.append(_failed_result(row.row_id, judge_model, str(exc)))
            continue
        results.append(_parse_response(row.row_id, judge_model, raw))
    return results


def merge_judgments(
    rows: Sequence[RewardTrainingRow],
    judgments: Sequence[FrontierJudgeResult],
    *,
    env_weight: float = DEFAULT_ENV_WEIGHT,
    frontier_weight: float = DEFAULT_FRONTIER_WEIGHT,
) -> list[RewardTrainingRow]:
    """Apply judgments back onto rows.

    Returns a NEW list with the same length as ``rows``. Rows that have
    a matching judgment carry an updated ``frontier_judgment`` /
    ``frontier_rationale`` / ``consensus_reward`` / ``disagreement`` /
    ``source="judgment"``. Rows without a matching judgment pass through
    unchanged.
    """
    by_id = {j.row_id: j for j in judgments if j.parsed_ok}
    out: list[RewardTrainingRow] = []
    for row in rows:
        match = by_id.get(row.row_id)
        if match is None:
            out.append(row)
            continue
        new_meta = dict(row.metadata)
        new_meta["judge_model"] = match.judge_model
        new_meta["judge_raw"] = match.raw_response[:512]
        consensus = consensus_reward(
            env_reward=row.env_reward,
            frontier_reward=match.score,
            env_weight=env_weight,
            frontier_weight=frontier_weight,
        )
        d = (
            disagreement(row.env_reward, match.score) if row.env_reward is not None else None
        )
        out.append(
            dataclasses.replace(
                row,
                frontier_judgment=match.score,
                frontier_rationale=match.rationale,
                consensus_reward=consensus,
                disagreement=d,
                source="judgment",
                metadata=new_meta,
            )
        )
    return out


# ── caller implementations ───────────────────────────────────────────


def openrouter_judge_caller(
    prompt: str,
    completion: str,
    judge_model: str,
    api_key: str,
    *,
    url: str = DEFAULT_OPENROUTER_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Default judge caller — POSTs to the OpenRouter chat endpoint.

    ``httpx`` is imported lazily so users without it (or running in
    offline CI) don't pay the import cost. Tests cover this function
    via ``unittest.mock.patch`` on ``httpx.post``.
    """
    if not api_key:
        raise ValueError("api_key is required for openrouter_judge_caller")

    import httpx  # noqa: PLC0415 — lazy

    body = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _format_judge_user_message(prompt, completion),
            },
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://verifiable-labs.dev",
        "X-Title": "Verifiable Labs Reward Distillation",
    }
    response = httpx.post(url, json=body, headers=headers, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected OpenRouter payload type: {type(payload)!r}")
    return payload


def stub_judge_caller(
    prompt: str,
    completion: str,
    judge_model: str,
    api_key: str,
) -> dict[str, Any]:
    """Deterministic offline stub — used by CI when no real key is set.

    Returns a uniform 0.5 score so downstream code paths exercise without
    network. Marks the response with ``"<stub>"`` so audit scans flag any
    accidental prod use.
    """
    del prompt, completion, judge_model, api_key
    body = json.dumps({"score": 0.5, "rationale": "<stub deterministic 0.5>"})
    return {
        "choices": [
            {"message": {"content": body}},
        ],
        "model": "<stub>",
    }


def _format_judge_user_message(prompt: str, completion: str) -> str:
    return (
        "PROMPT:\n"
        f"{prompt}\n\n"
        "COMPLETION:\n"
        f"{completion}\n\n"
        'Return strict JSON: {"score": float in [0,1], "rationale": str}.'
    )


# ── response parsing ─────────────────────────────────────────────────


def _parse_response(
    row_id: str, judge_model: str, payload: dict[str, Any]
) -> FrontierJudgeResult:
    raw = _extract_message_text(payload)
    score, rationale, parsed_ok = _extract_score_rationale(raw)
    return FrontierJudgeResult(
        row_id=row_id,
        score=score,
        rationale=rationale,
        judge_model=judge_model,
        raw_response=raw,
        parsed_ok=parsed_ok,
    )


def _failed_result(row_id: str, judge_model: str, error: str) -> FrontierJudgeResult:
    return FrontierJudgeResult(
        row_id=row_id,
        score=0.5,
        rationale=f"<judge call failed: {error}>",
        judge_model=judge_model,
        raw_response="",
        parsed_ok=False,
    )


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_score_rationale(raw: str) -> tuple[float, str, bool]:
    if not raw:
        return 0.5, "<empty judge response>", False

    candidates: list[str] = [raw.strip()]
    candidates.extend(_JSON_BLOCK_RE.findall(raw))
    seen: set[str] = set()
    for cand in candidates:
        cleaned = cand.strip()
        if cleaned in seen:
            continue
        seen.add(cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        score = obj.get("score")
        rationale = obj.get("rationale", "")
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        score_f = _clip01(score_f)
        rationale_str = str(rationale) if rationale is not None else ""
        return score_f, rationale_str, True
    return 0.5, "<failed to parse score>", False


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ── env-aware convenience helpers ────────────────────────────────────


def resolve_api_key(env_var: str = "OPENROUTER_API_KEY") -> str | None:
    """Read the OpenRouter key from the env, returning ``None`` if
    absent. The CLI uses this to decide whether to use the real or stub
    caller without forcing tests to clear env vars."""
    value = os.environ.get(env_var, "").strip()
    return value or None


def estimate_judge_cost(n_rows: int, *, per_row_usd: float = 0.005) -> float:
    """Rough USD cost for a judge slice — used by the CLI to honour the
    $30 cap from :doc:`PHASE_29_PLAN.md` §5 D1-D."""
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative; got {n_rows}")
    return float(n_rows) * float(per_row_usd)


__all__ = [
    "DEFAULT_BORDERLINE_HIGH",
    "DEFAULT_BORDERLINE_LOW",
    "DEFAULT_FRACTION",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_OPENROUTER_URL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "FrontierJudgeResult",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeCaller",
    "estimate_judge_cost",
    "is_borderline",
    "merge_judgments",
    "openrouter_judge_caller",
    "resolve_api_key",
    "sample_frontier_judgments",
    "select_borderline_rows",
    "stub_judge_caller",
]
