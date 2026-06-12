"""LLM adapter for the code-mini-repo environment (Phase 24).

Single-turn multi-file edit. The model sees a small repo + spec and
returns a JSON envelope ``{"files": {"<path>": "<content>", ...},
"confidence": <float>}``. The env's reward kernel handles
path-restriction and per-file compileability.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.code_mini_repo import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter


class CodeMiniRepoAdapter(EnvAdapter):
    env_name: str = "code-mini-repo"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)


__all__ = ["CodeMiniRepoAdapter"]
