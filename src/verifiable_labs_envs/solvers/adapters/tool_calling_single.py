"""LLM adapter for the tool-calling-single environment (Phase 25)."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.tool_calling_single import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter


class ToolCallingSingleAdapter(EnvAdapter):
    env_name: str = "tool-calling-single"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)


__all__ = ["ToolCallingSingleAdapter"]
