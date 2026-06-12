"""LLM adapter for the sql-single-turn environment (Phase 26)."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.sql_single_turn import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter


class SqlSingleTurnAdapter(EnvAdapter):
    env_name: str = "sql-single-turn"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)


__all__ = ["SqlSingleTurnAdapter"]
