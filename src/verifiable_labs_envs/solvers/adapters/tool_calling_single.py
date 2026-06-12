"""LLM adapter for the tool-calling-single environment (Phase 25)."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.tool_calling_single import (
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter
from verifiable_labs_envs.tool_primitives import schemas_for


class ToolCallingSingleAdapter(EnvAdapter):
    env_name: str = "tool-calling-single"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)

    def get_tools_schema(self, instance: Any) -> list[dict[str, Any]] | None:
        """Forward the env's tool schema (Phase 25 OpenAI function-calling).

        Per-instance: filters ``TOOL_SCHEMAS`` to only the
        ``instance.available_tools`` subset; falls back to the full
        pool if the instance doesn't restrict.
        """
        names = getattr(instance, "available_tools", None)
        if names:
            return schemas_for(names) or list(TOOL_SCHEMAS)
        return list(TOOL_SCHEMAS)


__all__ = ["ToolCallingSingleAdapter"]
