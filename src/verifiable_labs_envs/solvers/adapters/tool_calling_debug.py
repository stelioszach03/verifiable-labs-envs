"""LLM adapter for the tool-calling-debug environment (Phase 25)."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.tool_calling_debug import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.envs.tool_calling_single import TOOL_SCHEMAS
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter
from verifiable_labs_envs.tool_primitives import schemas_for


class ToolCallingDebugAdapter(EnvAdapter):
    env_name: str = "tool-calling-debug"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)

    def get_tools_schema(self, instance: Any) -> list[dict[str, Any]] | None:
        """Forward the env's tool schema (Phase 25 contract).

        The trace-debug env's instance carries an ``available_tools``
        list mirroring the shape of tool-calling-single; restrict the
        forwarded schema to that subset so the model picks from the
        same pool the trace was built against.
        """
        names = getattr(instance, "available_tools", None)
        if names:
            return schemas_for(names) or list(TOOL_SCHEMAS)
        return list(TOOL_SCHEMAS)


__all__ = ["ToolCallingDebugAdapter"]
