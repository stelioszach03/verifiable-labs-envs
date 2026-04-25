"""Canonical JSONL trace format for ``verifiable run`` output.

Every CLI run writes one ``Trace`` per episode to a JSONL file. Other
tools (``verifiable report``, ``verifiable compare``, the CI workflow)
read those files back without manual parsing.

Design rules:

- Required fields are minimal — anything that can be missing must be
  ``Optional`` and tolerated on read.
- Every trace carries a ``schema_version``; bump on a *breaking*
  change (renamed / removed field). Adding new optional fields does
  **not** require a bump.
- Reads are forgiving (unknown extra keys ignored), writes are strict
  (only known keys serialised).
- No env-specific shapes leak into the schema. ``reward_components``
  is a free-form ``dict[str, float]`` so different envs can attach
  different decompositions (NMSE / SSIM / conformal / etc.) without
  touching this module.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class FailureType(enum.StrEnum):
    """Coarse classification of why an episode failed (or didn't)."""

    NONE = "none"
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"
    INVALID_SHAPE = "invalid_shape"
    INVALID_JSON = "invalid_json"
    TOOL_ERROR = "tool_error"
    SCORING_ERROR = "scoring_error"
    UNKNOWN = "unknown"


# Required when reading a trace back. Anything else is optional.
_REQUIRED_KEYS = frozenset(
    {"schema_version", "trace_id", "env_name", "agent_name", "reward", "parse_success"}
)


@dataclass
class Trace:
    """One episode of (env, seed) × (agent) → (prediction, reward).

    Construct via :meth:`Trace.new` (auto-fills ``trace_id`` + timestamp)
    or directly when round-tripping through JSON.
    """

    # Required ────────────────────────────────────────────
    env_name: str
    agent_name: str
    reward: float
    parse_success: bool

    # Auto-filled ────────────────────────────────────────
    schema_version: int = SCHEMA_VERSION
    trace_id: str = field(default_factory=lambda: f"t_{uuid.uuid4().hex[:12]}")
    timestamp: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )

    # Optional ───────────────────────────────────────────
    env_version: str | None = None
    seed: int | None = None
    episode_id: str | None = None
    model_name: str | None = None
    observation_hash: str | None = None
    prediction_hash: str | None = None
    prediction_summary: dict[str, Any] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    classical_baseline_reward: float | None = None
    gap_to_classical: float | None = None
    coverage: float | None = None
    latency_ms: float | None = None
    token_input: int | None = None
    token_output: int | None = None
    estimated_cost_usd: float | None = None
    failure_type: FailureType = FailureType.NONE
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Construction helpers ──────────────────────────────────

    @classmethod
    def new(
        cls,
        *,
        env_name: str,
        agent_name: str,
        reward: float,
        parse_success: bool,
        **kwargs: Any,
    ) -> Trace:
        """Construct a fresh trace with a generated id + current timestamp."""
        return cls(
            env_name=env_name,
            agent_name=agent_name,
            reward=reward,
            parse_success=parse_success,
            **kwargs,
        )

    # JSON round-trip ───────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict. Drops ``None``-valued optional fields
        to keep the on-disk representation compact, but never drops required
        keys."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, FailureType):
                v = v.value
            if v is None and f.name not in _REQUIRED_KEYS:
                continue
            if isinstance(v, (list, dict)) and not v and f.name not in _REQUIRED_KEYS:
                continue
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        """Parse a dict back into a Trace, tolerating missing optional fields
        and unknown extra keys."""
        missing = _REQUIRED_KEYS - d.keys()
        if missing:
            raise ValueError(f"Trace is missing required keys: {sorted(missing)}")
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k not in known:
                continue  # forward-compat: ignore extras
            if k == "failure_type":
                v = FailureType(v) if not isinstance(v, FailureType) else v
            kwargs[k] = v
        # Backfill defaults for optional keys absent from the dict.
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


# JSONL helpers ────────────────────────────────────────────────


def write_jsonl(traces: Iterable[Trace], path: str | Path) -> int:
    """Write traces to ``path`` (one JSON object per line). Returns count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for t in traces:
            f.write(t.to_json())
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Trace]:
    """Read a JSONL file of traces. Skips blank lines; raises on malformed
    JSON or missing required keys."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trace file not found: {p}")
    out: list[Trace] = []
    with p.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no} is not valid JSON: {e}") from e
            try:
                out.append(Trace.from_dict(d))
            except ValueError as e:
                raise ValueError(f"{p}:{line_no} {e}") from e
    return out


# Hashing helpers (for observation_hash / prediction_hash) ─────


def hash_payload(payload: Any) -> str:
    """Stable short hash of a JSON-serialisable payload. Used by the CLI to
    record observation / prediction without dumping the full content."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


__all__ = [
    "SCHEMA_VERSION",
    "FailureType",
    "Trace",
    "write_jsonl",
    "read_jsonl",
    "hash_payload",
]
