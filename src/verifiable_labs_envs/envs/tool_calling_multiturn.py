"""tool-calling-multiturn — multi-turn tool-calling RL env (Phase 25.C).

Extends ``tool-calling-single`` with the math-multiturn-style
per-extra-turn penalty (D8-C parity, locked in PHASE_25_PLAN.md
§9.2):

    final_reward = base * (1 - min(0.05 · extra_turns, 0.10))

``extra_turns`` counts ASSISTANT messages produced between the
initial prompt and the final submission. Three rounds (one tool +
one feedback + one final) yields 1 extra turn → 0.95× the
single-turn reward; reaching the budget ceiling caps the penalty
at 0.10.

Verifier-derived feedback messages are rendered between turns:

- Empty trajectory + first error → format guidance.
- Last tool-call returned an error → echo the error message and
  invite the model to retry.
- Otherwise → echo the tool-call summary so far + budget remaining.

Hidden ``gold_spec`` is NEVER serialised into a feedback message
(R10 carry-over).

Problem generation, scoring, calibration, and the verifier all
delegate to :mod:`verifiable_labs_envs.envs.tool_calling_single` —
``tool_calling_multiturn`` is a pure rollout-shape wrapper around it.
"""
from __future__ import annotations

import contextlib
import json
from typing import Any

from verifiable_labs_envs.envs.tool_calling_single import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.tool_calling_single import (
    DEFAULT_MAX_TOOL_CALLS,
    MAX_TOOL_RESULT_BYTES,
    ToolCallingInstance,
    ToolCallingPrediction,
    ToolCallingSingleEnv,
    _cached_quantile,
    compute_reward,
)
from verifiable_labs_envs.tool_primitives import (
    TOOL_SCHEMAS,
    dispatch_tool,
    init_state,
    schemas_for,
)

NAME = "tool-calling-multiturn"
TURN_PENALTY_PER_EXTRA: float = 0.05
TURN_PENALTY_CAP: float = 0.10


def render_turn_feedback(
    *,
    last_call: dict[str, Any] | None,
    n_tool_calls: int,
    budget: int,
) -> str:
    """User-message body shown between turns.

    Verifier-derived only — never references ``gold_spec``. Used when
    the previous assistant turn was a tool call (we surface the
    result + remaining budget) or an empty/non-JSON message (we
    nudge toward the JSON envelope).
    """
    remaining = max(0, budget - n_tool_calls)
    if last_call is None:
        return (
            "FEEDBACK on your previous turn:\n"
            "No tool call was issued and no JSON envelope was produced. "
            "Issue a tool call OR submit a final non-tool message of the "
            'form `{"answer": <result>, "confidence": <float>}`.'
        )
    name = str(last_call.get("name", "?"))
    result = last_call.get("result", {})
    if isinstance(result, dict) and "error" in result:
        return (
            "FEEDBACK on your previous turn:\n"
            f"Tool `{name}` returned an error: {result['error']}. "
            f"You have {remaining} tool call(s) remaining; try a "
            "different tool or fix the arguments."
        )
    summary = json.dumps(result)[:200]
    return (
        "FEEDBACK on your previous turn:\n"
        f"Tool `{name}` succeeded; result preview: {summary}. "
        f"Remaining budget: {remaining} tool call(s)."
    )


class ToolCallingMultiturnEnv(ToolCallingSingleEnv):
    """:class:`ToolCallingSingleEnv` with explicit per-extra-turn penalty.

    The rollout loop is structurally identical to the single-turn
    env's; the only difference is that we apply
    ``_apply_turn_penalty`` to the final reward and inject verifier
    feedback messages between assistant turns.
    """

    name: str = NAME

    def _apply_turn_penalty(
        self,
        scored: dict[str, Any],
        n_assistant_turns: int,
    ) -> dict[str, Any]:
        """Multiply the base reward by ``(1 − penalty)``.

        ``n_assistant_turns`` is the number of assistant messages
        produced during the rollout (every tool-call counts as one,
        plus the final non-tool turn). The first assistant turn is
        free; each additional one accrues
        ``TURN_PENALTY_PER_EXTRA``, capped at ``TURN_PENALTY_CAP``.
        """
        extra = max(0, int(n_assistant_turns) - 1)
        penalty = min(TURN_PENALTY_CAP, TURN_PENALTY_PER_EXTRA * extra)
        base = float(scored["reward"])
        adjusted = max(0.0, base * (1.0 - penalty))
        scored["reward"] = float(adjusted)
        scored["meta"] = {
            **scored.get("meta", {}),
            "base_reward": base,
            "turn_penalty": float(penalty),
            "n_assistant_turns": int(n_assistant_turns),
        }
        return scored

    def run_rollout(
        self,
        solver: Any,
        instance: ToolCallingInstance,
        *,
        adapter: Any = None,
        max_tool_calls: int | None = None,
    ) -> dict[str, Any]:
        """Run the multi-turn tool-calling loop with verifier feedback.

        Returns the standard :meth:`score` dict with these extras in
        ``meta``:

        - ``tool_calls`` / ``n_tool_calls`` — same as single-turn.
        - ``base_reward`` — pre-penalty reward.
        - ``turn_penalty`` — fraction subtracted per extra assistant turn.
        - ``n_assistant_turns`` — total assistant messages emitted.
        - ``state`` — serialised :class:`WorkspaceState` at rollout end.
        """
        from verifiable_labs_envs.solvers.llm_solver import (
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        budget = int(
            max_tool_calls if max_tool_calls is not None else self.max_tool_calls
        )

        state = init_state(seed=instance.seed, initial_files=instance.initial_files)
        history: list[dict[str, Any]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        tool_calls: list[dict[str, Any]] = []
        last_prediction: ToolCallingPrediction | None = None
        final_text = ""
        n_assistant_turns = 0
        tool_schemas = schemas_for(instance.available_tools) or list(TOOL_SCHEMAS)

        for _ in range(budget + 1):
            completion = solver.complete_turns(history, tools=tool_schemas)
            n_assistant_turns += 1
            tool_call = getattr(completion, "tool_call", None)
            if tool_call is not None and len(tool_calls) < budget:
                result = dispatch_tool(tool_call.name, tool_call.arguments, state)
                call_record = {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result,
                }
                tool_calls.append(call_record)
                history.append({"role": "assistant", "content": completion.text or ""})
                history.append({
                    "role": "tool",
                    "name": tool_call.name,
                    "content": json.dumps(result)[:MAX_TOOL_RESULT_BYTES],
                })
                # Verifier feedback before the next assistant turn.
                history.append({
                    "role": "user",
                    "content": render_turn_feedback(
                        last_call=call_record,
                        n_tool_calls=len(tool_calls),
                        budget=budget,
                    ),
                })
                continue
            with contextlib.suppress(LLMSolverError):
                last_prediction = adapter.parse_response(completion.text, instance)
            final_text = completion.text or ""
            break

        prediction = ToolCallingPrediction(
            tool_calls=tuple(tool_calls),
            final_text=final_text,
            final_state=state,
            raw=final_text,
            confidence=last_prediction.confidence if last_prediction is not None else 0.0,
        )

        scored = self.score(prediction, instance)
        scored = self._apply_turn_penalty(scored, n_assistant_turns=n_assistant_turns)
        scored["meta"] = {
            **scored["meta"],
            "tool_calls": list(tool_calls),
            "n_tool_calls": len(tool_calls),
            "max_tool_calls": budget,
            "state": state.to_serialisable(),
        }
        return scored


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> ToolCallingMultiturnEnv:
    """Factory matching the single-turn env. Calibration is reused."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, float(_BASE_DEFAULTS["alpha"]))
    return ToolCallingMultiturnEnv(
        conformal_quantile=q, max_tool_calls=max_tool_calls
    )


# Re-export for convenience — keeps the public surface symmetric with
# single-turn so consumers don't have to know which module owns each
# helper.
def baseline_predict(instance: ToolCallingInstance) -> ToolCallingPrediction:
    from verifiable_labs_envs.envs.tool_calling_single import (
        baseline_predict as _baseline_predict,
    )
    return _baseline_predict(instance)


def generate_instance(seed: int, **kwargs: Any) -> ToolCallingInstance:
    from verifiable_labs_envs.envs.tool_calling_single import (
        generate_instance as _generate_instance,
    )
    return _generate_instance(seed, **kwargs)


__all__ = [
    "NAME",
    "TURN_PENALTY_PER_EXTRA",
    "TURN_PENALTY_CAP",
    "ToolCallingMultiturnEnv",
    "baseline_predict",
    "compute_reward",
    "generate_instance",
    "load_environment",
    "render_turn_feedback",
]
