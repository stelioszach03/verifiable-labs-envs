"""LLM adapter for __ENV_ID__.

Permissive prompt rendering + parsing. The expected JSON envelope is::

    {"query": "<SQLite SELECT query>", "confidence": <float in [0, 1]>}

The scorer reads ``query`` field; the JSON envelope is mandatory
(missing or malformed → ``format_valid = 0``).
"""
from __future__ import annotations

import json
import re
from typing import Any

from __ENV_PY__.data import SqlInstance, SqlPrediction

SYSTEM_PROMPT = (
    "You are an expert SQL programmer. Given a natural-language "
    "question and a schema description (CREATE TABLE statements plus "
    "per-table column rosters), return exactly one JSON object of the "
    "form\n\n"
    '    {"query": "<SQLite SELECT query>", "confidence": <float in [0, 1]>}\n\n'
    "Constraints:\n"
    "- SQLite dialect only (no Postgres-specific syntax).\n"
    "- SELECT, WITH, or EXPLAIN only — no INSERT / UPDATE / DELETE.\n"
    "- Avoid RANDOM() and other non-deterministic functions.\n"
    "- Use `LIMIT` only with an explicit `ORDER BY`.\n"
    "- Quote string literals with single quotes.\n\n"
    "No prose, no markdown fences — JSON only."
)


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(instance: SqlInstance) -> str:
    schema_md = instance.schema.render_markdown()
    create_block = "\n".join(instance.schema.create_statements)
    return (
        "PROBLEM:\n"
        f"{instance.prompt}\n\n"
        "SCHEMA (CREATE TABLE statements):\n"
        f"```\n{create_block}\n```\n\n"
        "SCHEMA (markdown):\n"
        f"{schema_md}\n\n"
        "OUTPUT SCHEMA:\n"
        '{"query": "<SQLite SELECT query>", "confidence": <float in [0, 1]>}'
    )


def parse_response(text: str, instance: SqlInstance) -> SqlPrediction:
    """Parse the LLM's text into a :class:`SqlPrediction`."""
    del instance
    cleaned = text.strip()
    candidates: list[str] = list(_FENCED_RE.findall(cleaned))
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))
    for c in candidates:
        try:
            data: Any = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        query = str(data.get("query", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return SqlPrediction(query=query, raw=text, confidence=confidence)
    return SqlPrediction(query="", raw=text, confidence=0.0)


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "parse_response"]
