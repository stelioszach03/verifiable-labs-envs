"""Problem generator for __ENV_ID__.

Each call must produce a fresh problem dict with these keys:

- ``template_name`` — categorical tag for telemetry.
- ``prompt`` — natural-language question the LLM reads.
- ``create_statements`` — tuple of ``CREATE TABLE ...`` strings.
- ``seed_statements`` — tuple of ``INSERT INTO ...`` strings.
- ``table_names`` — tuple of table names (in display order).
- ``column_names`` — dict mapping table → tuple of column names.
- ``gold_query`` — canonical SELECT used as oracle.
- ``gold_query_is_ordered`` — bool; True iff gold has ORDER BY.
- ``gold_result_rows`` — tuple of value tuples (the gold result-set).

Procedural regeneration from a 64-bit seed plus a finite template
pool gives the contamination-resistance guarantee that makes the env
safe to use as an RLVR training signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verifiable_labs_envs.sql_primitives import Schema


@dataclass(frozen=True)
class SqlInstance:
    """One SQL problem draw.

    ``gold_query`` + ``gold_result_rows`` are the oracle fields
    excluded from :meth:`as_inputs`. ``gold_query_is_ordered`` flips
    the comparator's ordered/unordered branch (D4-A).
    """

    prompt: str
    template_name: str
    seed: int
    schema: Schema
    gold_query: str
    gold_query_is_ordered: bool
    gold_result_rows: tuple[tuple[Any, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "schema": {
                "create_statements": list(self.schema.create_statements),
                "tables": list(self.schema.table_names),
                "columns_by_table": {
                    t: list(c) for t, c in self.schema.column_names_by_table.items()
                },
            },
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class SqlPrediction:
    """Solver's answer.

    ``query`` is the SELECT (or WITH / EXPLAIN) the model proposes;
    ``raw`` keeps the LLM's full response for traceability;
    ``confidence`` is a scalar self-report in ``[0, 1]``.
    """

    query: str
    raw: str = ""
    confidence: float = 0.5


def generate_problem(seed: int, **hyperparams: Any) -> dict[str, Any]:
    """Sample a fresh SQL problem from the per-env distribution.

    Determinism: identical ``seed`` returns the byte-identical dict.
    """
    raise NotImplementedError(
        "TODO: implement problem generator for __ENV_ID__. "
        "Use np.random.default_rng(seed) for reproducibility, sample "
        "templates from a pool whose size × seed-space × parameter "
        "range gives EFFECTIVE_INSTANCES > 1e15. See "
        "src/verifiable_labs_envs/sql_primitives/__init__.py for 8 "
        "worked-example templates."
    )


__all__ = ["SqlInstance", "SqlPrediction", "generate_problem"]
