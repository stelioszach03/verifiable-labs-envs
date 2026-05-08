"""LLM adapter for the long-context-needle environment (Phase 27)."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.long_context_needle import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter


class LongContextNeedleAdapter(EnvAdapter):
    env_name: str = "long-context-needle"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)


__all__ = ["LongContextNeedleAdapter"]
