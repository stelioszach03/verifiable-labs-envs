"""LLM adapter for the code-humaneval environment (Phase 24).

Single-turn: solver reads a function signature + docstring + visible
test block and returns a JSON envelope ``{"code": "<source>",
"confidence": <float>}``. The env's own ``parse_response`` handles
fenced blocks and free-form prose; this adapter is a thin
EnvAdapter-shaped wrapper so the platform's ``/v1/score`` endpoint
can drive the env without importing the env module directly.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.code_humaneval import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter


class CodeHumanevalLLMAdapter(EnvAdapter):
    env_name: str = "code-humaneval"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)


__all__ = ["CodeHumanevalLLMAdapter"]
