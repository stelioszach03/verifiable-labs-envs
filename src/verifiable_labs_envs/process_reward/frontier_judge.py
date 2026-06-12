"""Per-step frontier-model judgment harness (Phase 30.B, D2-C backup).

The frontier slice supplements the per-step env-procedural rewards
with a second opinion on **borderline steps** — those whose env per-step
reward sits in the middle band ``(0.3, 0.7)`` where the per-step
verifier signal is genuinely uncertain.

Design points (mirror Phase 29 frontier_judge.py):

- Default judge model: ``anthropic/claude-sonnet-4.6`` via OpenRouter.
- Caller is fully pluggable — tests inject a deterministic stub via
  ``judge_caller``; production uses :func:`openrouter_step_judge_caller`.
- Cost cap (D8 / §19): $50 per slice (vs $30 in Phase 29 outcome
  judging — per-step prompts are denser).
- Outputs land back as
  :class:`PerStepFrontierResult`; merging into rows is the caller's
  job (``merge_per_step_judgments``).

Privacy posture: only the prompt + the specific step text are sent
to the judge. Trace-level audit metadata records the judge model id.
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

from verifiable_labs_envs.process_reward.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    consensus_reward,
    per_step_disagreement,
)
from verifiable_labs_envs.process_reward.dataset import ProcessRewardTraceRow

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL: str = "anthropic/claude-sonnet-4.6"
DEFAULT_BORDERLINE_LOW: float = 0.3
DEFAULT_BORDERLINE_HIGH: float = 0.7
DEFAULT_FRACTION: float = 0.10
DEFAULT_OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_S: float = 30.0
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_COST_CAP_USD: float = 50.0
"""Locked per :doc:`PHASE_30_PLAN.md` §19 — $50 per slice (vs $30 for
Phase 29 outcome judge)."""


JUDGE_SYSTEM_PROMPT: str = (
    "You are evaluating a single reasoning step from a multi-step "
    "reasoning trace. Score the step on a 0-1 scale considering: "
    "logical correctness, factual accuracy, contribution to the final "
    "answer. Return strict JSON with shape "
    '{"score": float in [0,1], "rationale": short string}. Do not '
    "include any text outside the JSON object."
)


@dataclass(frozen=True)
class PerStepFrontierResult:
    """One judgment on one (prompt, step) pair within a trace."""

    row_id: str
    step_index: int
    score: float
    rationale: str
    judge_model: str
    raw_response: str
    parsed_ok: bool


PerStepJudgeCaller = Callable[[str, str, str, str, str], dict[str, Any]]
"""Signature: ``(prompt, prefix, step, judge_model, api_key) ->
response_dict``. ``prefix`` is the concatenation of all prior steps
(empty string for step 0); judges should consider the step in
context of what came before. Response shape mirrors OpenRouter chat-
completions: ``{"choices": [{"message": {"content": "..."}}]}``."""


def is_borderline_step(
    step_reward: float | None,
    *,
    low: float = DEFAULT_BORDERLINE_LOW,
    high: float = DEFAULT_BORDERLINE_HIGH,
) -> bool:
    """Predicate: should this *step* be sent to the frontier judge?

    True iff the env-procedural step reward is non-null and lies in
    the open interval ``(low, high)``.
    """
    if step_reward is None:
        return False
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f"require 0 <= low < high <= 1; got low={low}, high={high}"
        )
    return low < float(step_reward) < high


def select_borderline_step_targets(
    rows: Sequence[ProcessRewardTraceRow],
    *,
    fraction: float = DEFAULT_FRACTION,
    low: float = DEFAULT_BORDERLINE_LOW,
    high: float = DEFAULT_BORDERLINE_HIGH,
    seed: int = 0,
    max_steps: int | None = None,
) -> list[tuple[ProcessRewardTraceRow, int]]:
    """Pick ``(row, step_index)`` pairs to send to the judge.

    Filters to borderline steps across all rows, then samples
    ``fraction`` of them (rounded up to at least 1 if any borderline
    steps exist) using a seeded numpy RNG. ``max_steps`` provides a
    hard upper bound — the CLI uses this to cap spend regardless of
    dataset size.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}")

    targets: list[tuple[ProcessRewardTraceRow, int]] = []
    for row in rows:
        for i, r in enumerate(row.step_rewards):
            if is_borderline_step(r, low=low, high=high):
                targets.append((row, i))
    if not targets:
        return []

    desired = max(1, int(round(len(targets) * fraction)))
    if max_steps is not None:
        desired = min(desired, int(max_steps))
    desired = min(desired, len(targets))

    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(targets), size=desired, replace=False)
    indices.sort()
    return [targets[int(i)] for i in indices]


def sample_per_step_judgments(
    rows: Sequence[ProcessRewardTraceRow],
    *,
    fraction: float = DEFAULT_FRACTION,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
    judge_caller: PerStepJudgeCaller | None = None,
    seed: int = 0,
    max_steps: int | None = None,
    low: float = DEFAULT_BORDERLINE_LOW,
    high: float = DEFAULT_BORDERLINE_HIGH,
    raise_on_error: bool = False,
) -> list[PerStepFrontierResult]:
    """Sample borderline steps and call the judge.

    The function returns a flat list of :class:`PerStepFrontierResult`;
    integrating those results back into rows is
    :func:`merge_per_step_judgments`. Splitting the two steps keeps
    the network-touching path narrow and independently testable.

    ``judge_caller`` defaults to :func:`openrouter_step_judge_caller`
    when ``api_key`` is set; tests pass an explicit deterministic stub.
    Either ``api_key`` or ``judge_caller`` must be supplied.
    """
    if judge_caller is None:
        if not api_key:
            raise ValueError(
                "either judge_caller or api_key must be provided; "
                "set OPENROUTER_API_KEY or supply a stub for tests"
            )
        judge_caller = openrouter_step_judge_caller

    selected = select_borderline_step_targets(
        rows,
        fraction=fraction,
        low=low,
        high=high,
        seed=seed,
        max_steps=max_steps,
    )
    results: list[PerStepFrontierResult] = []
    for row, step_idx in selected:
        prefix = "\n".join(row.steps[:step_idx])
        step_text = row.steps[step_idx]
        try:
            raw = judge_caller(
                row.prompt, prefix, step_text, judge_model, api_key or ""
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "step-judge call failed for %s step %d: %s",
                row.row_id,
                step_idx,
                exc,
            )
            if raise_on_error:
                raise
            results.append(
                _failed_step_result(row.row_id, step_idx, judge_model, str(exc))
            )
            continue
        results.append(_parse_step_response(row.row_id, step_idx, judge_model, raw))
    return results


def merge_per_step_judgments(
    rows: Sequence[ProcessRewardTraceRow],
    judgments: Sequence[PerStepFrontierResult],
    *,
    env_weight: float = DEFAULT_ENV_WEIGHT,
    frontier_weight: float = DEFAULT_FRONTIER_WEIGHT,
) -> list[ProcessRewardTraceRow]:
    """Apply judgments back onto rows.

    Returns a NEW list with the same length as ``rows``. Rows that
    have at least one matching judgment have their per-step
    ``frontier_judgments``, ``frontier_rationales``,
    ``consensus_rewards``, ``disagreements`` updated; ``source`` flips
    to ``"judgment"``. Rows without any matching judgment pass
    through unchanged.
    """
    by_row: dict[str, dict[int, PerStepFrontierResult]] = {}
    for j in judgments:
        if not j.parsed_ok:
            continue
        by_row.setdefault(j.row_id, {})[j.step_index] = j

    out: list[ProcessRewardTraceRow] = []
    for row in rows:
        per_step = by_row.get(row.row_id, {})
        if not per_step:
            out.append(row)
            continue
        new_judgments = list(row.step_frontier_judgments)
        new_rationales = list(row.step_frontier_rationales)
        for step_idx, j in per_step.items():
            if 0 <= step_idx < row.step_count:
                new_judgments[step_idx] = j.score
                new_rationales[step_idx] = j.rationale
        new_consensus: list[float] = []
        for env_r, front_r in zip(row.step_rewards, new_judgments, strict=True):
            if env_r is None and front_r is None:
                new_consensus.append(0.5)
                continue
            new_consensus.append(
                consensus_reward(
                    env_r,
                    front_r,
                    env_weight=env_weight,
                    frontier_weight=frontier_weight,
                )
            )
        new_disagreements = per_step_disagreement(
            row.step_rewards, tuple(new_judgments)
        )
        new_aggregate = (
            sum(new_consensus) / len(new_consensus) if new_consensus else 0.5
        )
        new_meta = dict(row.metadata)
        new_meta["judge_model"] = next(iter(per_step.values())).judge_model
        new_meta["judged_step_count"] = len(per_step)
        out.append(
            dataclasses.replace(
                row,
                step_frontier_judgments=tuple(new_judgments),
                step_frontier_rationales=tuple(new_rationales),
                step_consensus_rewards=tuple(new_consensus),
                step_disagreements=tuple(new_disagreements),
                aggregate_reward=new_aggregate,
                source="judgment",
                metadata=new_meta,
            )
        )
    return out


# ── caller implementations ──────────────────────────────────────────


def openrouter_step_judge_caller(
    prompt: str,
    prefix: str,
    step: str,
    judge_model: str,
    api_key: str,
    *,
    url: str = DEFAULT_OPENROUTER_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Default caller — POSTs to the OpenRouter chat endpoint with a
    per-step judge prompt.

    Lazy ``httpx`` import so users without it pay no cost. Tests
    mock this via ``unittest.mock.patch`` on ``httpx.post``.
    """
    if not api_key:
        raise ValueError("api_key is required for openrouter_step_judge_caller")

    import httpx  # noqa: PLC0415

    body = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _format_step_user_message(prompt, prefix, step)},
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://verifiable-labs.dev",
        "X-Title": "Verifiable Labs Process Reward",
    }
    response = httpx.post(url, json=body, headers=headers, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected OpenRouter payload type: {type(payload)!r}")
    return payload


def stub_step_judge_caller(
    prompt: str,
    prefix: str,
    step: str,
    judge_model: str,
    api_key: str,
) -> dict[str, Any]:
    """Deterministic offline stub — uniform 0.5 score."""
    del prompt, prefix, step, judge_model, api_key
    body = json.dumps({"score": 0.5, "rationale": "<stub deterministic 0.5>"})
    return {
        "choices": [{"message": {"content": body}}],
        "model": "<stub>",
    }


def _format_step_user_message(prompt: str, prefix: str, step: str) -> str:
    return (
        "PROMPT:\n"
        f"{prompt}\n\n"
        "PRIOR STEPS:\n"
        f"{prefix or '<none>'}\n\n"
        "CURRENT STEP:\n"
        f"{step}\n\n"
        'Return strict JSON: {"score": float in [0,1], "rationale": str}.'
    )


# ── response parsing ────────────────────────────────────────────────


def _parse_step_response(
    row_id: str, step_index: int, judge_model: str, payload: dict[str, Any]
) -> PerStepFrontierResult:
    raw = _extract_message_text(payload)
    score, rationale, parsed_ok = _extract_score_rationale(raw)
    return PerStepFrontierResult(
        row_id=row_id,
        step_index=step_index,
        score=score,
        rationale=rationale,
        judge_model=judge_model,
        raw_response=raw,
        parsed_ok=parsed_ok,
    )


def _failed_step_result(
    row_id: str, step_index: int, judge_model: str, error: str
) -> PerStepFrontierResult:
    return PerStepFrontierResult(
        row_id=row_id,
        step_index=step_index,
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


# ── env helpers ─────────────────────────────────────────────────────


def resolve_api_key(env_var: str = "OPENROUTER_API_KEY") -> str | None:
    value = os.environ.get(env_var, "").strip()
    return value or None


def estimate_step_judge_cost(n_steps: int, *, per_step_usd: float = 0.005) -> float:
    """USD cost estimate for a per-step judge slice.

    ``per_step_usd`` defaults to $0.005 (Phase 29's outcome judge
    rate). Per-step slices typically cost ~10 % more than outcome
    slices because the judge prompt includes the prefix; the CLI
    bumps this to $0.0055 in the cost-cap calculation when needed.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative; got {n_steps}")
    return float(n_steps) * float(per_step_usd)


__all__ = [
    "DEFAULT_BORDERLINE_HIGH",
    "DEFAULT_BORDERLINE_LOW",
    "DEFAULT_COST_CAP_USD",
    "DEFAULT_FRACTION",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_OPENROUTER_URL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "JUDGE_SYSTEM_PROMPT",
    "PerStepFrontierResult",
    "PerStepJudgeCaller",
    "estimate_step_judge_cost",
    "is_borderline_step",
    "merge_per_step_judgments",
    "openrouter_step_judge_caller",
    "resolve_api_key",
    "sample_per_step_judgments",
    "select_borderline_step_targets",
    "stub_step_judge_caller",
]
