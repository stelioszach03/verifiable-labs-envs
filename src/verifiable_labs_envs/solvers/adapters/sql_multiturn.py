"""LLM adapter for the sql-multiturn environment (Phase 26).

Reuses :class:`SqlSingleTurnAdapter`'s prompt + parse for turn 1
and adds verifier feedback rendering between turns. The gold rows
are NEVER serialised into the feedback (R10 carry-over).
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.sql_multiturn import (
    query_diagnostic,
    render_sql_feedback,
)
from verifiable_labs_envs.envs.sql_single_turn import (
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

SYSTEM_PROMPT_MT = (
    "You are an expert SQL programmer. You have up to 3 turns to "
    "submit a working query.\n\n"
    "On turn 1 you see the natural-language question and the schema "
    "description (CREATE TABLE statements + per-table column rosters).\n"
    "On turns 2 and 3 you receive FEEDBACK on your previous query "
    "(parse status + row-count diagnostics). The gold result-set is "
    "held out — you cannot see it.\n\n"
    "Output exactly one JSON object per turn:\n"
    '    {"query": "<SQLite SELECT query>", "confidence": <float in [0, 1]>}\n\n'
    "Constraints:\n"
    "- SQLite dialect only.\n"
    "- SELECT, WITH, or EXPLAIN only — no INSERT / UPDATE / DELETE.\n"
    "- Avoid RANDOM() and other non-deterministic functions.\n"
    "- Use `LIMIT` only with an explicit `ORDER BY`.\n\n"
    "No prose, no markdown fences — JSON only."
)


class SqlMultiturnAdapter(EnvAdapter):
    env_name: str = "sql-multiturn"
    system_prompt: str = SYSTEM_PROMPT_MT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)

    def build_followup_turn(
        self,
        history: list[dict[str, str]],  # noqa: ARG002
        last_prediction: Any,
        instance: Any,
    ) -> str:
        diag = query_diagnostic(
            instance=instance,
            query=last_prediction.query if last_prediction else "",
        )
        return render_sql_feedback(diag)


__all__ = ["SqlMultiturnAdapter", "SYSTEM_PROMPT_MT"]
