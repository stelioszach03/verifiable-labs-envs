"""LLM adapter for the code-humaneval-tools environment (Phase 24).

Drives the read_file / write_file / run_test rollout. The platform's
``LLMSolver`` consumes ``TOOL_SCHEMAS`` directly when running through
``run_rollout``; this adapter handles the prompt/parse layer so
``/v1/score`` (which scores a single completion, not a rollout)
treats the tools-env the same as ``code-humaneval``.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.code_humaneval_tools import (
    SYSTEM_PROMPT_TOOLS,
    TOOL_SCHEMAS,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter


class CodeHumanevalToolsAdapter(EnvAdapter):
    env_name: str = "code-humaneval-tools"
    system_prompt: str = SYSTEM_PROMPT_TOOLS

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)

    def get_tools_schema(self, instance: Any) -> list[dict[str, Any]] | None:
        """Forward the env's 3-primitive read/write/run_test schema."""
        del instance
        return list(TOOL_SCHEMAS)


__all__ = ["CodeHumanevalToolsAdapter"]
