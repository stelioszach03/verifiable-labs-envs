"""Process-reward dataset construction (Phase 30.B, plan §6 / §15).

Public surface:

- :class:`ProcessRewardTraceRow` — frozen dataclass holding one
  ``(prompt, steps, per_step_rewards, aggregate)`` tuple in canonical
  JSONL shape.
- :func:`collect_env_traces` — extracts trace rows from the locked
  25-env catalogue + the existing Phase 29 dataset infrastructure.
- :func:`extend_from_phase29_rows` — augments a Phase 29
  :class:`~verifiable_labs_envs.reward_distillation.dataset.RewardTrainingRow`
  list into PRM-ready trace rows by running the segmenter +
  per-step labeler on each completion.
- :func:`write_jsonl` / :func:`read_jsonl` — round-trippable JSONL
  IO mirroring the Phase 29 pattern.

D5-D primary slice flows through this module: rows carry the env's
per-step procedural rewards (D2-D), optional per-step frontier
judgments (D2-C, gated), per-step consensus reward (D5-D blend), per-step
conformal intervals (placeholder until 30.D), aggregate score, and
audit metadata. The trained student arrives in 30.G; 30.B builds the
harness.

Module is **CPU-only** by contract — no torch, no transformers, no
GPU. The downstream training pipeline lives in 30.C.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from verifiable_labs_envs import _REGISTRY, load_environment
from verifiable_labs_envs.process_reward.consensus import (
    per_step_consensus,
    trace_aggregate_consensus,
)
from verifiable_labs_envs.process_reward.segmentation import (
    DEFAULT_MAX_STEPS,
    is_pre_segmented,
    segment_trace,
)
from verifiable_labs_envs.process_reward.step_labeling import (
    StepLabelOutcome,
    label_steps,
)
from verifiable_labs_envs.reward_distillation.dataset import (
    DEFAULT_HELD_OUT_ENVS as PHASE29_HELD_OUT_ENVS,
)
from verifiable_labs_envs.reward_distillation.dataset import (
    DEFAULT_TRAINING_ENVS as PHASE29_TRAINING_ENVS,
)
from verifiable_labs_envs.reward_distillation.dataset import (
    RewardTrainingRow,
    baseline_completion_source,
)

# ── locked constants ────────────────────────────────────────────────

DEFAULT_HELD_OUT_ENVS: tuple[str, ...] = PHASE29_HELD_OUT_ENVS
"""D7-A held-out test envs — same as Phase 29 (long-context-synthesis,
sql-multiturn, code-mini-repo)."""

DEFAULT_TRAINING_ENVS: tuple[str, ...] = PHASE29_TRAINING_ENVS
"""22 training envs (25 total minus 3 held-out)."""

ROW_ID_PREFIX: str = "prw_"
ROW_ID_HASH_LEN: int = 16
SCHEMA_VERSION: str = "v0.1.0"
DEFAULT_AGGREGATION_METHOD: str = "mean"


SourceLiteral = Literal["env", "external", "judgment"]


# ── canonical row dataclass ─────────────────────────────────────────


@dataclass(frozen=True)
class ProcessRewardTraceRow:
    """One row of process-reward training data.

    Field semantics match :doc:`PHASE_30_PLAN.md` §7
    `ProcessRewardTraceRow` schema:

    - ``steps`` is the post-segmentation step list (immutable tuple);
      length matches every per-step list below.
    - ``step_rewards`` is the per-step env-procedural reward (D2-D).
      ``None`` entries mark steps with no env signal (handled by the
      consensus blender).
    - ``step_frontier_judgments`` is the optional per-step frontier
      judgment (D2-C); ``None`` until the frontier slice runs.
    - ``step_consensus_rewards`` is the actual MSE distillation target
      per step, computed via the 70/30 D5-D blend.
    - ``step_conformal_intervals`` is reserved for the post-30.D
      calibration step; ``None`` for every step in the 30.B harness
      output.
    - ``aggregate_reward`` is the trace-level score (mean over per-step
      consensus by default; ``method="env_blend"`` available).
    """

    row_id: str
    env_id: str | None
    prompt: str
    steps: tuple[str, ...]
    step_rewards: tuple[float | None, ...]
    step_components: tuple[dict[str, float] | None, ...]
    step_conformal_intervals: tuple[tuple[float, float] | None, ...]
    step_frontier_judgments: tuple[float | None, ...]
    step_frontier_rationales: tuple[str | None, ...]
    step_consensus_rewards: tuple[float, ...]
    step_disagreements: tuple[float | None, ...]
    aggregate_reward: float
    aggregate_conformal_interval: tuple[float, float] | None
    decomposition: str
    segmentation_strategy: str
    segmentation_confidence: float
    truncated: bool
    source: SourceLiteral
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # Normalise tuple-of-tuple CIs into list-of-list for JSON.
        d["step_conformal_intervals"] = [
            ([float(ci[0]), float(ci[1])] if ci is not None else None)
            for ci in self.step_conformal_intervals
        ]
        if self.aggregate_conformal_interval is not None:
            d["aggregate_conformal_interval"] = [
                float(self.aggregate_conformal_interval[0]),
                float(self.aggregate_conformal_interval[1]),
            ]
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProcessRewardTraceRow:
        steps = tuple(str(s) for s in payload["steps"])
        n = len(steps)

        def _fetch_seq(key: str, default=None) -> tuple:
            seq = payload.get(key)
            if seq is None:
                return tuple(default for _ in range(n))
            return tuple(seq)

        step_cis_raw = payload.get("step_conformal_intervals") or [None] * n
        step_cis: list[tuple[float, float] | None] = []
        for entry in step_cis_raw:
            if entry is None:
                step_cis.append(None)
            else:
                step_cis.append((float(entry[0]), float(entry[1])))

        agg_ci_raw = payload.get("aggregate_conformal_interval")
        agg_ci: tuple[float, float] | None = (
            None
            if agg_ci_raw is None
            else (float(agg_ci_raw[0]), float(agg_ci_raw[1]))
        )

        return cls(
            row_id=str(payload["row_id"]),
            env_id=payload.get("env_id"),
            prompt=str(payload["prompt"]),
            steps=steps,
            step_rewards=tuple(_maybe_float(v) for v in _fetch_seq("step_rewards")),
            step_components=tuple(
                (dict(v) if isinstance(v, dict) else None)
                for v in _fetch_seq("step_components")
            ),
            step_conformal_intervals=tuple(step_cis),
            step_frontier_judgments=tuple(
                _maybe_float(v) for v in _fetch_seq("step_frontier_judgments")
            ),
            step_frontier_rationales=tuple(
                (str(v) if v is not None else None)
                for v in _fetch_seq("step_frontier_rationales")
            ),
            step_consensus_rewards=tuple(
                float(v) for v in _fetch_seq("step_consensus_rewards", default=0.5)
            ),
            step_disagreements=tuple(
                _maybe_float(v) for v in _fetch_seq("step_disagreements")
            ),
            aggregate_reward=float(payload.get("aggregate_reward", 0.5)),
            aggregate_conformal_interval=agg_ci,
            decomposition=str(payload.get("decomposition", "terminal_uniform")),
            segmentation_strategy=str(payload.get("segmentation_strategy", "single_step")),
            segmentation_confidence=float(payload.get("segmentation_confidence", 0.0)),
            truncated=bool(payload.get("truncated", False)),
            source=payload.get("source", "env"),
            metadata=payload.get("metadata", {}) or {},
        )


def make_row_id(
    env_id: str | None,
    prompt: str,
    steps: Sequence[str],
    seed: int | None = None,
) -> str:
    """Deterministic row id ``prw_<sha256[:16]>`` over the canonical
    ``(env_id, seed, prompt, joined_steps)`` tuple. Re-extraction at
    the same seed produces a bit-identical id (R10 invariant)."""
    h = hashlib.sha256()
    h.update((env_id or "<external>").encode("utf-8"))
    h.update(b"|")
    h.update(str(seed if seed is not None else "<n/a>").encode("utf-8"))
    h.update(b"|")
    h.update(prompt.encode("utf-8"))
    h.update(b"|")
    h.update("".join(s for s in steps).encode("utf-8"))
    return f"{ROW_ID_PREFIX}{h.hexdigest()[:ROW_ID_HASH_LEN]}"


# ── completion + trace source abstraction ───────────────────────────


def baseline_trace_source(
    env_id: str, env: Any, instance: Any, seed: int
) -> tuple[str, str | list[str], dict[str, Any]]:
    """Default trace source — uses each env's reference baseline fn
    (the same path :func:`baseline_completion_source` from Phase 29
    uses) to produce a (prompt, completion-as-trace, score) triple.

    The completion text is treated as the reasoning trace; the
    segmenter splits it into steps. Deterministic, API-free, suitable
    for CI. Matches the Phase 29 pattern verbatim — only the return
    type differs (the completion is interpreted as a trace string,
    which the segmenter then segments).
    """
    prompt, completion, score = baseline_completion_source(
        env_id, env, instance, seed
    )
    return prompt, completion, score


# ── env-trace extraction ────────────────────────────────────────────


def collect_env_traces(
    env_ids: Sequence[str],
    n_per_env: int,
    *,
    seed_start: int = 0,
    trace_source: Callable[
        [str, Any, Any, int], tuple[str, str | list[str], dict[str, Any]]
    ] = baseline_trace_source,
    env_loader: Callable[[str], Any] = load_environment,
    max_steps: int = DEFAULT_MAX_STEPS,
    aggregation_method: str = DEFAULT_AGGREGATION_METHOD,
    fail_fast: bool = False,
    on_error: Callable[[str, int, Exception], None] | None = None,
) -> list[ProcessRewardTraceRow]:
    """Procedural extraction with step segmentation + per-step labels.

    For each env in ``env_ids`` and each of ``n_per_env`` consecutive
    seeds starting at ``seed_start``:

    1. ``env = env_loader(env_id)``
    2. ``instance = env.generate_instance(seed)``
    3. ``prompt, trace, score = trace_source(env_id, env, instance, seed)``
    4. ``segmentation = segment_trace(trace, max_steps=max_steps)``
    5. ``labels = label_steps(env_id=env_id, steps=segmentation.steps,
       outcome_reward=score['reward'], components=score['components'])``
    6. Build a :class:`ProcessRewardTraceRow` with ``source="env"``,
       per-step env rewards from ``labels``, no frontier judgments
       yet, per-step consensus = env-only blend.

    Errors per (env_id, seed) are routed to ``on_error`` if supplied,
    otherwise either raised (when ``fail_fast``) or silently dropped.
    """
    if n_per_env < 0:
        raise ValueError(f"n_per_env must be non-negative; got {n_per_env}")

    rows: list[ProcessRewardTraceRow] = []
    for env_id in env_ids:
        env = env_loader(env_id)
        for offset in range(n_per_env):
            seed = int(seed_start) + offset
            try:
                instance = env.generate_instance(seed)
                prompt, trace, score = trace_source(env_id, env, instance, seed)
                row = _build_env_trace_row(
                    env_id=env_id,
                    seed=seed,
                    prompt=prompt,
                    trace=trace,
                    score=score,
                    max_steps=max_steps,
                    aggregation_method=aggregation_method,
                )
            except Exception as exc:  # noqa: BLE001
                if on_error is not None:
                    on_error(env_id, seed, exc)
                if fail_fast:
                    raise
                continue
            rows.append(row)
    return rows


def extend_from_phase29_rows(
    rows: Iterable[RewardTrainingRow],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    aggregation_method: str = DEFAULT_AGGREGATION_METHOD,
) -> list[ProcessRewardTraceRow]:
    """Augment Phase 29 :class:`RewardTrainingRow` rows into
    :class:`ProcessRewardTraceRow` rows by segmenting the completion
    + relabeling per step.

    Each resulting row inherits the source env id, the prompt, and
    the metadata of the input row; the completion text becomes the
    raw trace input to the segmenter. ``env_reward`` from the Phase
    29 row is the outcome reward fed into :func:`label_steps`.
    """
    out: list[ProcessRewardTraceRow] = []
    for row in rows:
        seed_meta = row.metadata.get("seed") if row.metadata else None
        seed = int(seed_meta) if isinstance(seed_meta, int) else None
        # Reconstruct a synthetic score dict that label_steps can use.
        score = {
            "reward": float(
                row.env_reward
                if row.env_reward is not None
                else row.consensus_reward
            ),
            "components": dict(row.env_components or {}),
            "meta": {},
        }
        try:
            new_row = _build_env_trace_row(
                env_id=row.env_id,
                seed=seed,
                prompt=row.prompt,
                trace=row.completion,
                score=score,
                max_steps=max_steps,
                aggregation_method=aggregation_method,
                source=row.source,
                inherited_metadata=row.metadata,
            )
        except Exception:  # noqa: BLE001
            continue
        out.append(new_row)
    return out


def _build_env_trace_row(
    *,
    env_id: str | None,
    seed: int | None,
    prompt: str,
    trace: str | Sequence[str],
    score: dict[str, Any],
    max_steps: int,
    aggregation_method: str,
    source: SourceLiteral = "env",
    inherited_metadata: dict[str, Any] | None = None,
) -> ProcessRewardTraceRow:
    segmentation = segment_trace(trace, max_steps=max_steps)
    outcome_reward = float(score.get("reward", 0.0))
    components = score.get("components") or {}
    label_outcome: StepLabelOutcome = label_steps(
        env_id=env_id,
        steps=segmentation.steps,
        outcome_reward=outcome_reward,
        components=components if components else None,
    )

    n = segmentation.step_count
    step_consensus = per_step_consensus(label_outcome.step_rewards, None)
    aggregate = trace_aggregate_consensus(
        step_consensus,
        env_outcome=outcome_reward,
        method=aggregation_method,
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "raw_trace_chars": len(
            "".join(segmentation.steps)
            if is_pre_segmented(trace)
            else (trace if isinstance(trace, str) else "")
        ),
        "outcome_reward": outcome_reward,
        "outcome_components": dict(components) if components else None,
    }
    if segmentation.warning is not None:
        metadata["segmentation_warning"] = segmentation.warning
    if inherited_metadata:
        metadata = {**inherited_metadata, **metadata}

    row_id = make_row_id(env_id, prompt, segmentation.steps, seed)
    return ProcessRewardTraceRow(
        row_id=row_id,
        env_id=env_id,
        prompt=prompt,
        steps=segmentation.steps,
        step_rewards=tuple(label_outcome.step_rewards),
        step_components=tuple(label_outcome.step_components),
        step_conformal_intervals=tuple(None for _ in range(n)),
        step_frontier_judgments=tuple(None for _ in range(n)),
        step_frontier_rationales=tuple(None for _ in range(n)),
        step_consensus_rewards=step_consensus,
        step_disagreements=tuple(None for _ in range(n)),
        aggregate_reward=aggregate,
        aggregate_conformal_interval=None,
        decomposition=label_outcome.decomposition,
        segmentation_strategy=segmentation.strategy,
        segmentation_confidence=segmentation.confidence,
        truncated=segmentation.truncated,
        source=source,
        metadata=metadata,
    )


# ── JSONL IO ────────────────────────────────────────────────────────


def write_jsonl(
    rows: Iterable[ProcessRewardTraceRow], path: Path | str
) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[ProcessRewardTraceRow]:
    p = Path(path)
    rows: list[ProcessRewardTraceRow] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            rows.append(ProcessRewardTraceRow.from_dict(payload))
    return rows


def merge_jsonl(paths: Sequence[Path | str]) -> list[ProcessRewardTraceRow]:
    out: list[ProcessRewardTraceRow] = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


def trace_dataset_summary(
    rows: Sequence[ProcessRewardTraceRow],
) -> dict[str, Any]:
    """Aggregate stats — used by the CLI summary command + run audit."""
    if not rows:
        return {
            "n_traces": 0,
            "n_steps_total": 0,
            "by_env": {},
            "by_decomposition": {},
            "by_segmentation_strategy": {},
            "aggregate_reward_mean": 0.0,
            "step_count_mean": 0.0,
            "schema_version": SCHEMA_VERSION,
        }

    by_env: dict[str, int] = {}
    by_decomposition: dict[str, int] = {}
    by_strategy: dict[str, int] = {}
    by_source: dict[str, int] = {}
    n_steps = 0
    aggregate_sum = 0.0
    truncated = 0
    for row in rows:
        env_key = row.env_id or "<external>"
        by_env[env_key] = by_env.get(env_key, 0) + 1
        by_decomposition[row.decomposition] = (
            by_decomposition.get(row.decomposition, 0) + 1
        )
        by_strategy[row.segmentation_strategy] = (
            by_strategy.get(row.segmentation_strategy, 0) + 1
        )
        by_source[row.source] = by_source.get(row.source, 0) + 1
        n_steps += row.step_count
        aggregate_sum += float(row.aggregate_reward)
        if row.truncated:
            truncated += 1

    return {
        "n_traces": len(rows),
        "n_steps_total": int(n_steps),
        "step_count_mean": float(n_steps) / len(rows),
        "by_env": dict(sorted(by_env.items())),
        "by_decomposition": dict(sorted(by_decomposition.items())),
        "by_segmentation_strategy": dict(sorted(by_strategy.items())),
        "by_source": dict(sorted(by_source.items())),
        "aggregate_reward_mean": aggregate_sum / len(rows),
        "n_truncated": truncated,
        "schema_version": SCHEMA_VERSION,
    }


def is_held_out(env_id: str | None) -> bool:
    return env_id is not None and env_id in set(DEFAULT_HELD_OUT_ENVS)


def env_loader_safe(env_id: str) -> Any:
    if env_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown env {env_id!r}; available: {available}")
    return load_environment(env_id)


def default_train_envs() -> list[str]:
    return list(DEFAULT_TRAINING_ENVS)


def output_path_default() -> Path:
    return Path("reports") / "process_reward" / "v0.0.1_train.jsonl"


def is_phase30_collect_frontier_enabled() -> bool:
    """Optional gate for the per-step frontier judge slice (D2-C).

    Set ``VLABS_PHASE30_COLLECT_FRONTIER=1`` *and* have
    ``OPENROUTER_API_KEY`` in env to enable the live judge slice.
    The default behaviour is the no-API harness path; this lets CI
    stay offline while the maintainer can run the live slice on
    demand. Mirrors :doc:`PHASE_30_PLAN.md` §19 cost gating.
    """
    flag = os.environ.get("VLABS_PHASE30_COLLECT_FRONTIER", "").strip().lower()
    has_key = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    return flag in {"1", "true", "yes", "on"} and has_key


def _maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


__all__ = [
    "DEFAULT_AGGREGATION_METHOD",
    "DEFAULT_HELD_OUT_ENVS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_TRAINING_ENVS",
    "ROW_ID_HASH_LEN",
    "ROW_ID_PREFIX",
    "SCHEMA_VERSION",
    "ProcessRewardTraceRow",
    "baseline_trace_source",
    "collect_env_traces",
    "default_train_envs",
    "env_loader_safe",
    "extend_from_phase29_rows",
    "is_held_out",
    "is_phase30_collect_frontier_enabled",
    "make_row_id",
    "merge_jsonl",
    "output_path_default",
    "read_jsonl",
    "trace_dataset_summary",
    "write_jsonl",
]
