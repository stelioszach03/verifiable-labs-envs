"""LLM adapter for the tool-calling-multiturn environment (Phase 25)."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.tool_calling_multiturn import (
    render_turn_feedback,
)
from verifiable_labs_envs.envs.tool_calling_single import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

SYSTEM_PROMPT_MT = (
    SYSTEM_PROMPT
    + "\n\nThis is a MULTI-TURN task: between turns you will receive "
    "verifier feedback (tool-call results + remaining budget). Use "
    "the feedback to revise your plan. Each extra assistant turn "
    "incurs a small reward penalty (capped at 10%)."
)


class ToolCallingMultiturnAdapter(EnvAdapter):
    env_name: str = "tool-calling-multiturn"
    system_prompt: str = SYSTEM_PROMPT_MT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)

    def build_followup_turn(
        self,
        history: list[dict[str, str]],  # noqa: ARG002
        last_prediction: Any,  # noqa: ARG002
        instance: Any,  # noqa: ARG002
    ) -> str:
        # Multi-turn rollout machinery in the env module composes the
        # feedback message directly from the workspace state; this
        # method is here only to satisfy the EnvAdapter interface.
        return render_turn_feedback(last_call=None, n_tool_calls=0, budget=0)


__all__ = ["ToolCallingMultiturnAdapter", "SYSTEM_PROMPT_MT"]
