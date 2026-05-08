"""LLM adapter for the code-humaneval-multiturn environment (Phase 24).

Reuses :class:`CodeHumanevalLLMAdapter` for single-turn
prompt/parse, layering ``build_followup_turn`` on top so the
multi-turn env's ``run_rollout`` can produce verifier feedback
between turns. Hidden tests are NEVER referenced in the feedback
(R10).
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.code_humaneval import (
    build_user_prompt,
    parse_response,
)
from verifiable_labs_envs.envs.code_humaneval_multiturn import (
    render_feedback_message,
    visible_test_feedback,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

SYSTEM_PROMPT_MT = (
    "You are an expert Python programmer. You have up to 3 turns to "
    "submit a working implementation.\n\n"
    "On turn 1 you see the function signature, docstring, and a small "
    "visible test block.\n"
    "On turns 2 and 3 you receive FEEDBACK on your previous answer "
    "(visible-test pass count + first-failure excerpt). The hidden "
    "test suite is held out — you cannot see it.\n\n"
    "Output exactly one JSON object per turn:\n"
    '    {"code": "<Python source>", "confidence": <float in [0, 1]>}\n\n'
    "No prose, no markdown fences, JSON only."
)


class CodeHumanevalMultiturnAdapter(EnvAdapter):
    env_name: str = "code-humaneval-multiturn"
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
        feedback = visible_test_feedback(last_prediction, instance)
        return render_feedback_message(feedback)


__all__ = ["CodeHumanevalMultiturnAdapter", "SYSTEM_PROMPT_MT"]
