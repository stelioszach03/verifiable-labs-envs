"""External-data slice — UltraFeedback subset (Phase 29.B, plan §5 D4-C).

Per :doc:`PHASE_29_PLAN.md`:

- 3 000 rows from the HuggingFace ``openbmb/UltraFeedback`` corpus
  contribute breadth on instruction-following / helpfulness, prevent
  the student collapsing onto the env-specific reward shape (R4).
- The slice carries ``source="external"`` and ``env_reward=None``;
  the consensus reward falls back to the frontier model's score
  (UltraFeedback's per-completion ``overall_score``, normalised to
  ``[0, 1]``).
- The HuggingFace ``datasets`` library is **optional** — the harness
  works without it via :func:`build_synthetic_subset`, which yields a
  deterministic offline-friendly proxy of the same shape. CI uses the
  synthetic path; production uses the real loader.

The shape of each emitted row is identical to the env-procedural rows
so downstream training consumes a single uniform JSONL stream.
"""
from __future__ import annotations

import hashlib
import importlib
import logging
from typing import Any

from verifiable_labs_envs.reward_distillation.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    consensus_reward,
)
from verifiable_labs_envs.reward_distillation.dataset import (
    SCHEMA_VERSION,
    RewardTrainingRow,
    make_row_id,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME: str = "openbmb/UltraFeedback"
DEFAULT_SPLIT: str = "train"
DEFAULT_OVERALL_SCALE: float = 10.0
"""UltraFeedback overall_score is in ``[1, 10]`` per its dataset card.
We rescale to ``[0, 1]`` by dividing by 10."""


def collect_ultrafeedback_subset(
    n: int,
    *,
    seed: int = 0,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    fallback_to_synthetic: bool = True,
) -> list[RewardTrainingRow]:
    """Sample ``n`` UltraFeedback rows and project them into
    :class:`RewardTrainingRow` shape.

    Behaviour:

    - If the HuggingFace ``datasets`` library is importable AND the
      dataset name resolves, sample ``n`` rows deterministically using
      ``seed`` (numpy default RNG) from the chosen split.
    - Else, if ``fallback_to_synthetic``, return :func:`build_synthetic_subset`
      so CI / offline runs still produce a row stream of the right shape.
    - Else raise :class:`RuntimeError`.

    The synthetic path is **clearly marked** in row metadata
    (``metadata["synthetic"] = True``) so downstream training pipelines
    can filter it out for production runs.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    if n == 0:
        return []

    try:
        records = _load_real_dataset(n, seed=seed, dataset_name=dataset_name, split=split)
    except _UltraFeedbackUnavailable as exc:
        logger.info("UltraFeedback real loader unavailable (%s); using synthetic", exc)
        if not fallback_to_synthetic:
            raise RuntimeError(f"UltraFeedback unavailable: {exc}") from exc
        return build_synthetic_subset(n, seed=seed)

    rows: list[RewardTrainingRow] = []
    for record in records:
        try:
            row = _record_to_row(record)
        except _UltraFeedbackBadRow as exc:
            logger.debug("dropping malformed UltraFeedback row: %s", exc)
            continue
        rows.append(row)
    return rows


def build_synthetic_subset(n: int, *, seed: int = 0) -> list[RewardTrainingRow]:
    """Deterministic offline proxy for the real UltraFeedback subset.

    The text is procedurally generated from ``(seed, index)`` so the
    test surface is bit-stable, and the ``overall_score`` is sampled
    uniformly from a coarse 1-10 grid that mirrors the real
    distribution. ``metadata["synthetic"] = True`` flags the row.

    This intentionally does NOT depend on ``datasets`` or the network;
    it's the harness path. The 29.G real-training run uses
    :func:`collect_ultrafeedback_subset` with ``fallback_to_synthetic=False``.
    """
    import numpy as np

    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")

    rng = np.random.default_rng(seed)
    rows: list[RewardTrainingRow] = []
    for i in range(n):
        prompt_seed = int(rng.integers(0, 2**31 - 1))
        prompt_kind = rng.choice(["explain", "summarise", "classify", "translate", "decide"])
        prompt = (
            f"[synthetic-uf #{i:05d}] Please {prompt_kind} the following "
            f"piece of content (token id {prompt_seed:>10}). Be concise."
        )
        completion = (
            f"[synthetic completion #{i:05d}] Stand-in response generated "
            f"deterministically from (seed={seed}, index={i}); used in "
            f"CI when the HuggingFace datasets path is offline."
        )
        score = float(rng.choice([2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0])) / DEFAULT_OVERALL_SCALE
        rows.append(_build_row(prompt, completion, score, synthetic=True, source_index=i))
    return rows


# ── internals ────────────────────────────────────────────────────────


class _UltraFeedbackUnavailable(RuntimeError):
    """Raised when the real loader can't run — missing dep, missing
    network, or dataset name unresolved."""


class _UltraFeedbackBadRow(ValueError):
    """Raised when a single row is malformed (missing fields). The
    caller drops the row and continues."""


def _load_real_dataset(
    n: int, *, seed: int, dataset_name: str, split: str
) -> list[dict[str, Any]]:
    try:
        datasets_mod = importlib.import_module("datasets")
    except (ImportError, AttributeError) as exc:  # pragma: no cover
        # AttributeError covers the partially-initialised-module case
        # that arises when test runners poison ``sys.modules['datasets']``
        # via monkeypatch and a later test triggers a circular re-import.
        raise _UltraFeedbackUnavailable("datasets library not installed") from exc

    load_fn = getattr(datasets_mod, "load_dataset", None)
    if load_fn is None:
        raise _UltraFeedbackUnavailable(
            "datasets.load_dataset is not available (partial import?)"
        )

    try:
        ds = load_fn(dataset_name, split=split)
    except Exception as exc:  # noqa: BLE001 — HF surfaces many error types
        raise _UltraFeedbackUnavailable(
            f"failed to load {dataset_name!r} split={split!r}: {exc}"
        ) from exc

    try:
        total = len(ds)
    except Exception as exc:  # noqa: BLE001
        raise _UltraFeedbackUnavailable(
            f"{dataset_name!r} returned non-sized dataset: {exc}"
        ) from exc
    if total == 0:
        raise _UltraFeedbackUnavailable(f"{dataset_name!r} split={split!r} is empty")

    import numpy as np

    rng = np.random.default_rng(seed)
    take = min(n, total)
    indices = rng.choice(total, size=take, replace=False)
    indices.sort()
    try:
        records = [ds[int(i)] for i in indices]
    except Exception as exc:  # noqa: BLE001
        raise _UltraFeedbackUnavailable(
            f"failed to read records from {dataset_name!r}: {exc}"
        ) from exc
    return records


def _record_to_row(record: dict[str, Any]) -> RewardTrainingRow:
    """Project an UltraFeedback record into our row shape.

    UltraFeedback columns: ``instruction``, ``completions`` (list of
    ``{model, principle, custom_system_prompt, response,
    annotations: {"overall-rating": str|float, ...}}``). We project a
    single (instruction, completion) pair per record by picking the
    *first* completion deterministically — this is a 29.B harness shape;
    real training in 29.F-G can sample multiple completions per record.
    """
    prompt = record.get("instruction") or record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise _UltraFeedbackBadRow("missing/empty instruction")
    completions = record.get("completions") or record.get("completion")
    if isinstance(completions, list) and completions:
        first = completions[0]
        completion = _extract_completion_text(first)
        score = _extract_overall_score(first)
    elif isinstance(completions, str):
        completion = completions
        score = _extract_overall_score(record)
    else:
        raise _UltraFeedbackBadRow("no completions in record")
    if not completion:
        raise _UltraFeedbackBadRow("empty completion text")
    return _build_row(prompt, completion, score, synthetic=False, source_index=None)


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("response", "completion", "text"):
            value = completion.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _extract_overall_score(record: Any) -> float:
    """Pull an ``overall-rating`` (or fallback) from the record and
    rescale to ``[0, 1]``. Missing → 0.5 (neutral).

    The dataset stores rating either as a string ("1"-"10") or float
    depending on subset; both are tolerated.
    """
    if not isinstance(record, dict):
        return 0.5
    annotations = record.get("annotations") or {}
    candidates = (
        annotations.get("overall-rating"),
        annotations.get("overall_rating"),
        record.get("overall-rating"),
        record.get("overall_rating"),
    )
    for cand in candidates:
        if cand is None:
            continue
        try:
            value = float(cand)
        except (TypeError, ValueError):
            continue
        return _clip01(value / DEFAULT_OVERALL_SCALE)
    return 0.5


def _build_row(
    prompt: str,
    completion: str,
    overall_score: float,
    *,
    synthetic: bool,
    source_index: int | None,
) -> RewardTrainingRow:
    overall_score = _clip01(float(overall_score))
    consensus = consensus_reward(
        env_reward=None,
        frontier_reward=overall_score,
        env_weight=DEFAULT_ENV_WEIGHT,
        frontier_weight=DEFAULT_FRONTIER_WEIGHT,
    )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "external_dataset": DEFAULT_DATASET_NAME,
        "external_split": DEFAULT_SPLIT,
        "synthetic": bool(synthetic),
    }
    if source_index is not None:
        metadata["external_index"] = int(source_index)
    return RewardTrainingRow(
        row_id=make_row_id(None, prompt, completion, seed=None),
        env_id=None,
        prompt=prompt,
        completion=completion,
        env_reward=None,
        env_components=None,
        conformal_interval=None,
        frontier_judgment=overall_score,
        frontier_rationale=None,
        consensus_reward=consensus,
        disagreement=None,
        source="external",
        metadata=metadata,
    )


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def deterministic_subset_indices(total: int, n: int, *, seed: int) -> list[int]:
    """Public helper exposed for testing — return the indices the real
    loader would pick. Useful for asserting reproducibility without
    running the network call."""
    if total <= 0:
        return []
    import numpy as np

    take = min(n, total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=take, replace=False)
    return sorted(int(i) for i in indices)


def fingerprint_text(text: str) -> str:
    """SHA-256 fingerprint of an UltraFeedback prompt/completion. Used
    in audit metadata for the live slice; the synthetic path doesn't
    use this (those rows are deterministic from seed+index already)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Helpful aliases — the 29.B-E test surface uses the underscore-stripped
# names because they read more naturally inside test files.
collect_external_subset = collect_ultrafeedback_subset


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_OVERALL_SCALE",
    "DEFAULT_SPLIT",
    "build_synthetic_subset",
    "collect_external_subset",
    "collect_ultrafeedback_subset",
    "deterministic_subset_indices",
    "fingerprint_text",
]
