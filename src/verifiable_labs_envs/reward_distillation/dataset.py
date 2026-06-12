"""Reward-distillation dataset construction (Phase 29.B, plan §7).

Public surface:

- :class:`RewardTrainingRow` — frozen dataclass holding one
  (prompt, completion, target) tuple in canonical JSONL shape.
- :func:`collect_env_rows` — procedurally extracts rows from the locked
  25-env catalogue using a pluggable :class:`CompletionSource`. Default
  source is the env's own ``baseline_predict`` function which is
  deterministic and API-free, suitable for CI.
- :func:`write_jsonl` / :func:`read_jsonl` — round-trippable JSONL IO.

D4-A *procedural extraction* primary slice flows through this module:
rows carry the env's procedural reward + Phase 22 conformal interval
on the env's residual, both of which feed the D6-A MSE distillation
target via :func:`consensus_reward`.

The module is **CPU-only** by contract — no torch, no transformers,
no GPU. The trained student arrives in 29.G; 29.B builds the harness.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from verifiable_labs_envs import _REGISTRY, load_environment
from verifiable_labs_envs.reward_distillation.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    consensus_reward,
)

# ── locked constants ─────────────────────────────────────────────────

DEFAULT_HELD_OUT_ENVS: tuple[str, ...] = (
    "long-context-synthesis",
    "sql-multiturn",
    "code-mini-repo",
)
"""D4 held-out test envs — reserved from training, used as the
in-distribution generalization probe in :doc:`PHASE_29_PLAN.md` §9."""

DEFAULT_TRAINING_ENVS: tuple[str, ...] = tuple(
    env_id for env_id in sorted(_REGISTRY) if env_id not in set(DEFAULT_HELD_OUT_ENVS)
)
"""22 envs used for procedural-row extraction (25 total minus 3 held-out)."""

ROW_ID_PREFIX: str = "rwd_"
ROW_ID_HASH_LEN: int = 16
SCHEMA_VERSION: str = "v0.1.0"

# ── canonical row dataclass ──────────────────────────────────────────


SourceLiteral = Literal["env", "external", "judgment"]


@dataclass(frozen=True)
class RewardTrainingRow:
    """One row of distillation training data.

    Field semantics match :doc:`PHASE_29_PLAN.md` §5 D5-D row schema:

    - ``env_reward`` is the primary supervision signal from the Layer 1
      conformal moat. ``None`` for ``source="external"`` rows.
    - ``frontier_judgment`` is the optional second opinion from a
      frontier model (D5-C). ``None`` until 29.B's frontier slice runs.
    - ``consensus_reward`` is the actual MSE distillation target,
      computed via the 70/30 D5-D blend.
    - ``conformal_interval`` is the Phase 22 conformal CI on the env
      reward; persisted so 29.D can compute calibration drift.
    """

    row_id: str
    env_id: str | None
    prompt: str
    completion: str
    env_reward: float | None
    env_components: dict[str, float] | None
    conformal_interval: tuple[float, float] | None
    frontier_judgment: float | None
    frontier_rationale: str | None
    consensus_reward: float
    disagreement: float | None
    source: SourceLiteral
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-serializable representation."""
        d = dataclasses.asdict(self)
        if self.conformal_interval is not None:
            d["conformal_interval"] = [
                float(self.conformal_interval[0]),
                float(self.conformal_interval[1]),
            ]
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RewardTrainingRow:
        """Inverse of :meth:`to_dict`. Tolerates list-shaped CI."""
        ci = payload.get("conformal_interval")
        if ci is not None:
            ci = (float(ci[0]), float(ci[1]))
        return cls(
            row_id=str(payload["row_id"]),
            env_id=payload.get("env_id"),
            prompt=str(payload["prompt"]),
            completion=str(payload["completion"]),
            env_reward=_maybe_float(payload.get("env_reward")),
            env_components=payload.get("env_components"),
            conformal_interval=ci,
            frontier_judgment=_maybe_float(payload.get("frontier_judgment")),
            frontier_rationale=payload.get("frontier_rationale"),
            consensus_reward=float(payload["consensus_reward"]),
            disagreement=_maybe_float(payload.get("disagreement")),
            source=payload.get("source", "env"),
            metadata=payload.get("metadata", {}) or {},
        )


def make_row_id(env_id: str | None, prompt: str, completion: str, seed: int | None = None) -> str:
    """Deterministic row id ``rwd_<sha256[:16]>`` over the canonical
    ``(env_id, seed, prompt, completion)`` tuple. Re-extracting at the
    same seed produces a bit-identical id."""
    h = hashlib.sha256()
    h.update((env_id or "<external>").encode("utf-8"))
    h.update(b"|")
    h.update(str(seed if seed is not None else "<n/a>").encode("utf-8"))
    h.update(b"|")
    h.update(prompt.encode("utf-8"))
    h.update(b"|")
    h.update(completion.encode("utf-8"))
    return f"{ROW_ID_PREFIX}{h.hexdigest()[:ROW_ID_HASH_LEN]}"


# ── completion source abstraction ────────────────────────────────────


class CompletionSource(Protocol):
    """Produces (prompt_text, completion_text, env_score_dict) for one
    instance.

    The ``env_score_dict`` must follow the env-scorer convention
    (``{"reward": float, "components": dict, "meta": dict}``); the
    components/meta are passed straight through into the row.
    """

    def __call__(
        self, env_id: str, env: Any, instance: Any, seed: int
    ) -> tuple[str, str, dict[str, Any]]: ...


_BASELINE_FN_CANDIDATES: tuple[str, ...] = (
    "baseline_predict",
    "zero_baseline",
    "ista_baseline",
)


def baseline_completion_source(
    env_id: str, env: Any, instance: Any, seed: int
) -> tuple[str, str, dict[str, Any]]:
    """Default completion source — uses each env's reference baseline fn.

    Deterministic, API-free, suitable for CI. The extracted prompt is
    ``instance.prompt`` if the instance carries one (text envs); else
    a JSON dump of ``instance.as_inputs()`` with numpy arrays converted
    to nested lists. The completion is a JSON dump of the prediction's
    serialisable fields.

    Looks up the baseline fn by name, trying ``baseline_predict``,
    ``zero_baseline``, and ``ista_baseline`` in order. The first two
    accept ``(instance,)`` directly; ``ista_baseline``-style fns take
    keyword arguments derived from ``instance.as_inputs()``. Envs that
    expose none of those raise :class:`RuntimeError`.

    29.F-G replace this with an OpenRouter/Haiku source that calls the
    Phase 28 inference path; the row shape is identical.
    """
    module = importlib.import_module(_REGISTRY[env_id])
    prediction = _call_baseline(module, instance)
    if prediction is None:
        raise RuntimeError(
            f"env {env_id!r} module {module.__name__} exposes no baseline fn "
            f"(tried {_BASELINE_FN_CANDIDATES}); supply an explicit CompletionSource."
        )
    score = env.score(prediction, instance)
    prompt = _instance_to_prompt_text(env_id, instance, seed)
    completion = _prediction_to_completion_text(prediction)
    return prompt, completion, score


def _call_baseline(module: Any, instance: Any) -> Any | None:
    """Try the candidate baseline functions in order. The signature
    ``fn(instance)`` is the common case; for envs whose baselines take
    raw kwargs (sparse-fourier's ``zero_baseline``), we fall through to
    ``fn(**instance.as_inputs())``.
    """
    for name in _BASELINE_FN_CANDIDATES:
        fn = getattr(module, name, None)
        if fn is None:
            continue
        try:
            return fn(instance)
        except TypeError:
            pass
        if hasattr(instance, "as_inputs"):
            try:
                return fn(**instance.as_inputs())
            except TypeError:
                continue
    return None


def _instance_to_prompt_text(env_id: str, instance: Any, seed: int) -> str:
    if hasattr(instance, "prompt") and isinstance(instance.prompt, str):
        return instance.prompt
    if hasattr(instance, "as_inputs"):
        try:
            inputs = instance.as_inputs()
        except Exception:  # noqa: BLE001 — fall through to skeleton tag below
            inputs = {}
    else:
        inputs = {}
    payload = {"env_id": env_id, "seed": int(seed), "inputs": _jsonable(inputs)}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _prediction_to_completion_text(prediction: Any) -> str:
    if dataclasses.is_dataclass(prediction):
        d = dataclasses.asdict(prediction)
    elif isinstance(prediction, dict):
        d = dict(prediction)
    else:
        d = {"value": str(prediction)}
    return json.dumps(_jsonable(d), sort_keys=True, ensure_ascii=False)


def _jsonable(obj: Any) -> Any:
    """Coerce numpy arrays / scalars into JSON-serialisable forms.

    Complex numbers (sparse-fourier measurements) become
    ``{"real": ..., "imag": ...}`` dicts so JSON round-trips don't
    lose information.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return [{"real": float(v.real), "imag": float(v.imag)} for v in obj.ravel().tolist()]
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


# ── env-row extraction ───────────────────────────────────────────────


def collect_env_rows(
    env_ids: Sequence[str],
    n_per_env: int,
    *,
    seed_start: int = 0,
    completion_source: CompletionSource = baseline_completion_source,
    env_loader: Callable[[str], Any] = load_environment,
    fail_fast: bool = False,
    on_error: Callable[[str, int, Exception], None] | None = None,
) -> list[RewardTrainingRow]:
    """Procedural extraction from the env catalogue.

    For each env in ``env_ids`` and each of ``n_per_env`` consecutive
    seeds starting at ``seed_start``:

    1. ``env = env_loader(env_id)``
    2. ``instance = env.generate_instance(seed)``
    3. ``prompt, completion, score = completion_source(env_id, env, instance, seed)``
    4. Build a :class:`RewardTrainingRow` with ``source="env"``, the
       env's reward as ``env_reward``, no frontier judgment yet.

    Per :doc:`PHASE_29_PLAN.md` §7 reproducibility contract: re-extraction
    at the same seed produces bit-identical rows (modulo whatever
    determinism the chosen ``completion_source`` provides; the default
    baseline source is fully deterministic).

    Errors per (env_id, seed) are routed to ``on_error`` if supplied,
    otherwise either raised (when ``fail_fast``) or silently dropped
    (the row simply doesn't appear in the output). The default is to
    drop, mirroring the §7 "filter out obviously-broken rows" rule.
    """
    if n_per_env < 0:
        raise ValueError(f"n_per_env must be non-negative; got {n_per_env}")

    rows: list[RewardTrainingRow] = []
    for env_id in env_ids:
        env = env_loader(env_id)
        for offset in range(n_per_env):
            seed = int(seed_start) + offset
            try:
                instance = env.generate_instance(seed)
                prompt, completion, score = completion_source(env_id, env, instance, seed)
            except Exception as exc:  # noqa: BLE001
                if on_error is not None:
                    on_error(env_id, seed, exc)
                if fail_fast:
                    raise
                continue
            rows.append(_row_from_env_score(env_id, seed, prompt, completion, score))
    return rows


def _row_from_env_score(
    env_id: str,
    seed: int,
    prompt: str,
    completion: str,
    score: dict[str, Any],
) -> RewardTrainingRow:
    reward = float(score.get("reward", 0.0))
    components = _coerce_float_dict(score.get("components"))
    meta = score.get("meta", {}) or {}
    ci = _extract_conformal_interval(reward, meta)
    consensus = consensus_reward(
        reward, None, env_weight=DEFAULT_ENV_WEIGHT, frontier_weight=DEFAULT_FRONTIER_WEIGHT
    )
    return RewardTrainingRow(
        row_id=make_row_id(env_id, prompt, completion, seed),
        env_id=env_id,
        prompt=prompt,
        completion=completion,
        env_reward=reward,
        env_components=components,
        conformal_interval=ci,
        frontier_judgment=None,
        frontier_rationale=None,
        consensus_reward=consensus,
        disagreement=None,
        source="env",
        metadata={
            "seed": int(seed),
            "schema_version": SCHEMA_VERSION,
            "env_meta": _jsonable(meta),
        },
    )


def _coerce_float_dict(maybe: Any) -> dict[str, float] | None:
    if maybe is None:
        return None
    if not isinstance(maybe, dict):
        return None
    return {str(k): float(v) for k, v in maybe.items()}


def _extract_conformal_interval(
    reward: float, meta: dict[str, Any]
) -> tuple[float, float] | None:
    """The env's score meta carries a ``conformal_quantile`` and
    ``residual``; the implied finite-sample CI is ``[reward - q, reward + q]``
    clipped to ``[0, 1]``. Returns ``None`` if the env didn't produce a
    quantile (some envs run uncalibrated)."""
    q = meta.get("conformal_quantile")
    if q is None:
        return None
    q_f = float(q)
    low = max(0.0, reward - q_f)
    high = min(1.0, reward + q_f)
    return (low, high)


def _maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


# ── JSONL IO ─────────────────────────────────────────────────────────


def write_jsonl(rows: Iterable[RewardTrainingRow], path: Path | str) -> int:
    """Write rows to a JSONL file (one JSON object per line). Returns
    the number of rows written. The output is byte-identical when
    re-run on the same input due to ``sort_keys=True`` and the
    deterministic row id."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[RewardTrainingRow]:
    """Inverse of :func:`write_jsonl`. Empty / blank lines are skipped."""
    p = Path(path)
    rows: list[RewardTrainingRow] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            rows.append(RewardTrainingRow.from_dict(payload))
    return rows


def merge_jsonl(paths: Sequence[Path | str]) -> list[RewardTrainingRow]:
    """Concatenate multiple JSONL files in order. Used by the CLI to
    merge per-env extraction shards into a single training file."""
    out: list[RewardTrainingRow] = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


def dataset_summary(rows: Sequence[RewardTrainingRow]) -> dict[str, Any]:
    """Quick aggregate stats — emitted by the CLI alongside the JSONL
    so a reviewer can sanity-check the extraction at a glance."""
    if not rows:
        return {
            "n_rows": 0,
            "by_env": {},
            "by_source": {},
            "consensus_mean": 0.0,
            "consensus_min": 0.0,
            "consensus_max": 0.0,
            "schema_version": SCHEMA_VERSION,
        }

    by_env: dict[str, int] = {}
    by_source: dict[str, int] = {}
    consensus_values: list[float] = []
    for row in rows:
        env_key = row.env_id or "<external>"
        by_env[env_key] = by_env.get(env_key, 0) + 1
        by_source[row.source] = by_source.get(row.source, 0) + 1
        consensus_values.append(float(row.consensus_reward))

    consensus_arr = np.asarray(consensus_values, dtype=np.float64)
    return {
        "n_rows": len(rows),
        "by_env": dict(sorted(by_env.items())),
        "by_source": dict(sorted(by_source.items())),
        "consensus_mean": float(consensus_arr.mean()),
        "consensus_min": float(consensus_arr.min()),
        "consensus_max": float(consensus_arr.max()),
        "schema_version": SCHEMA_VERSION,
    }


def is_held_out(env_id: str | None) -> bool:
    """Predicate: would this env be excluded from training under the
    locked D4 split?"""
    return env_id is not None and env_id in set(DEFAULT_HELD_OUT_ENVS)


def env_loader_safe(env_id: str) -> Any:
    """``load_environment`` with a friendlier error path for the CLI.

    Raises :class:`KeyError` (not Python's bare KeyError-from-importlib
    soup) when ``env_id`` is unknown. Used by the CLI; tests bypass
    this and use ``load_environment`` directly.
    """
    if env_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown env {env_id!r}; available: {available}")
    return load_environment(env_id)


def default_train_envs() -> list[str]:
    """The 22 training envs (all registered envs minus the 3 held-out)."""
    return list(DEFAULT_TRAINING_ENVS)


def output_path_default() -> Path:
    """Canonical training-set output path
    ``reports/reward_distillation/v0.0.1_train.jsonl``.

    The repository-relative resolution is left to the caller (CLI uses
    the user's CWD; tests pass an explicit ``tmp_path``)."""
    return Path("reports") / "reward_distillation" / "v0.0.1_train.jsonl"


def env_disk_size_estimate(rows: Sequence[RewardTrainingRow]) -> int:
    """Rough disk footprint estimate (bytes) for a row collection — the
    CLI reports this so the user notices unexpectedly-large dumps. Sums
    UTF-8 lengths of the JSON-encoded rows."""
    total = 0
    for row in rows:
        total += len(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        total += 1  # newline
    return total


def is_phase29_collect_frontier_enabled() -> bool:
    """Optional gate for the OpenRouter frontier slice.

    Set ``VLABS_PHASE29_COLLECT_FRONTIER=1`` *and* have ``OPENROUTER_API_KEY``
    in env to enable the live judge slice. The default behaviour is the
    no-API harness path; this lets CI stay offline while the maintainer
    can run the live slice on demand. Mirrors the gating contract from
    :doc:`PHASE_29_PLAN.md` §19.
    """
    flag = os.environ.get("VLABS_PHASE29_COLLECT_FRONTIER", "").strip().lower()
    has_key = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    return flag in {"1", "true", "yes", "on"} and has_key


__all__ = [
    "DEFAULT_HELD_OUT_ENVS",
    "DEFAULT_TRAINING_ENVS",
    "ROW_ID_PREFIX",
    "ROW_ID_HASH_LEN",
    "SCHEMA_VERSION",
    "RewardTrainingRow",
    "CompletionSource",
    "baseline_completion_source",
    "collect_env_rows",
    "dataset_summary",
    "default_train_envs",
    "env_disk_size_estimate",
    "env_loader_safe",
    "is_held_out",
    "is_phase29_collect_frontier_enabled",
    "make_row_id",
    "merge_jsonl",
    "output_path_default",
    "read_jsonl",
    "write_jsonl",
]
