"""code-humaneval-multiturn — multi-turn code-execution RL environment.

Phase 24.C extends ``code-humaneval`` with a 3-turn rollout (D8-C
locked, matching math-algebra-multiturn verbatim):

* **Turn 1**: the LLM sees the function signature + docstring +
  visible test block and proposes a Python implementation
  (``solution_v1``) plus a self-reported confidence.
* **Turn 2**: the server runs ``solution_v1`` against the
  ``visible_tests`` only (no test source, no oracle) and reports
  passed / failed counts. The model proposes ``solution_v2``.
* **Turn 3**: same — the LLM proposes ``solution_final``. The final
  reward is computed against ``visible_tests ∪ hidden_tests``
  (R10 — hidden tests are NEVER shown to the model).

Reward is computed on the **final** turn's prediction with a
turn-count penalty: ``final = base * (1 − 0.05 · (n_turns − 1))``,
capped at ``0.10``. Three turns scores 0.9× the equivalent
single-turn reward.

Problem generation, scoring, calibration, and the sandbox primitive
all delegate to :mod:`verifiable_labs_envs.envs.code_humaneval` —
``code_humaneval_multiturn`` is a pure conversation wrapper around it.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.code_humaneval import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.code_humaneval import (
    DEFAULT_MEM_BYTES,
    DEFAULT_TIMEOUT_S_PER_CALL,
    CodeHumanevalEnv,
    CodeInstance,
    CodePrediction,
    _cached_quantile,
    score_components,
)
from verifiable_labs_envs.sandbox import (
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

NAME = "code-humaneval-multiturn"
DEFAULT_MAX_TURNS: int = 3
# Per-extra-turn penalty as a fraction of base reward, capped at TURN_PENALTY_CAP.
TURN_PENALTY_PER_EXTRA: float = 0.05
TURN_PENALTY_CAP: float = 0.10


def _format_visible_only_test_module(instance: CodeInstance) -> str:
    """Render a pytest module containing ONLY the visible tests.

    Used by ``visible_test_feedback`` between turns — the model is
    told how many of its visible cases passed, but never sees the
    hidden test set or its results.
    """
    asserts = list(instance.visible_tests)
    lines = [
        "from solution import *  # noqa: F401, F403",
        "",
    ]
    for i, a in enumerate(asserts):
        lines.append(f"def test_visible_{i:03d}():")
        lines.append(f"    assert {a}")
        lines.append("")
    return "\n".join(lines)


def visible_test_feedback(
    prediction: CodePrediction,
    instance: CodeInstance,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_CALL,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> dict[str, Any]:
    """Run the prediction against the visible tests only.

    Returns ``{passed, failed, total, format_valid, parse_valid,
    error_excerpt}``. The ``error_excerpt`` is a single-line summary
    of the pytest stdout — meant to be shown to the model so it can
    correct an obvious bug between turns. The hidden tests + their
    expected outputs are NEVER included (R10).
    """
    components = score_components(
        prediction,
        instance,
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
    )
    feedback: dict[str, Any] = {
        "format_valid": float(components["format_valid"]),
        "parse_valid": float(components["parse_valid"]),
        "passed": 0,
        "failed": 0,
        "total": len(instance.visible_tests),
        "error_excerpt": "",
    }
    if components["format_valid"] == 0.0:
        feedback["error_excerpt"] = (
            "Output was not valid JSON containing a `code` field."
        )
        return feedback
    if components["parse_valid"] == 0.0:
        feedback["error_excerpt"] = (
            "Submitted code did not compile; check for syntax errors."
        )
        return feedback

    code = prediction.code.strip() or prediction.raw
    files = {
        "solution.py": code + "\n",
        "test_solution.py": _format_visible_only_test_module(instance),
    }
    manifest = build_pytest_manifest(["test_solution.py"], timeout_s=timeout_s)
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=manifest,
        mem_bytes=mem_bytes,
    )
    counts = parse_pytest_q_summary(result.stdout)
    feedback["passed"] = int(counts["passed"])
    feedback["failed"] = int(counts["failed"]) + int(counts["error"])
    if feedback["failed"] > 0:
        # First failure-line excerpt — keep it short to avoid drowning
        # the next turn's prompt.
        for line in (result.stdout + "\n" + result.stderr).splitlines():
            if "FAILED" in line or "AssertionError" in line:
                feedback["error_excerpt"] = line.strip()[:240]
                break
    return feedback


def render_feedback_message(feedback: dict[str, Any]) -> str:
    """Render the feedback dict as the user message for the next turn."""
    if feedback["format_valid"] == 0.0:
        return (
            "FEEDBACK on your previous turn:\n"
            "Your output was not valid JSON. Return exactly one JSON object "
            'matching the schema: {"code": "<source>", "confidence": <float>}.'
        )
    if feedback["parse_valid"] == 0.0:
        return (
            "FEEDBACK on your previous turn:\n"
            "Your `code` field did not compile. Check for syntax errors and "
            "make sure the function signature matches the prompt."
        )
    passed = feedback["passed"]
    total = feedback["total"]
    if total > 0 and passed == total:
        return (
            "FEEDBACK on your previous turn:\n"
            f"All {total} visible test(s) passed. The hidden test set may "
            "still trip your solution — review edge cases (empty input, "
            "negative numbers, boundary conditions) before the final turn."
        )
    excerpt = feedback["error_excerpt"]
    body = (
        f"You passed {passed}/{total} visible test case(s)."
    )
    if excerpt:
        body += f"\nFirst failure: {excerpt}"
    return "FEEDBACK on your previous turn:\n" + body


class CodeHumanevalMultiturnEnv(CodeHumanevalEnv):
    """:class:`CodeHumanevalEnv` with a multi-turn rollout entry point."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        super().__init__(conformal_quantile, hyperparams, weights)
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1; got {max_turns}")
        self.max_turns = int(max_turns)

    def _apply_turn_penalty(
        self,
        scored: dict[str, Any],
        n_turns: int,
    ) -> dict[str, Any]:
        """Multiply the base reward by ``(1 − penalty)``.

        Same shape as math_algebra_multiturn._apply_turn_penalty (D8-C).
        """
        penalty = min(
            TURN_PENALTY_CAP,
            TURN_PENALTY_PER_EXTRA * max(0, n_turns - 1),
        )
        base = float(scored["reward"])
        adjusted = max(0.0, base * (1.0 - penalty))
        scored["reward"] = float(adjusted)
        scored["meta"] = {
            **scored.get("meta", {}),
            "base_reward": base,
            "turn_penalty": float(penalty),
        }
        return scored

    def run_rollout(
        self,
        solver: Any,
        instance: CodeInstance,
        *,
        adapter: Any = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        """Run up to ``max_turns`` turns of ``solver`` on ``instance``.

        Returns the final-turn :meth:`score` dict with these extras in
        ``meta``:

        - ``turn_rewards``: list[float], score after each turn.
        - ``turn_components``: list[dict], per-turn component breakdown.
        - ``n_turns``: int, the number of turns actually taken.
        - ``max_turns``: int, the cap for this rollout.
        - ``base_reward``: float, unpenalised final-turn reward.
        - ``turn_penalty``: float, fraction subtracted for extra turns.

        ``adapter`` defaults to looking up
        ``code-humaneval-multiturn`` in the global EnvAdapter
        registry; pass an explicit adapter to bypass the lookup
        (useful before 24.F when the adapter isn't auto-registered).
        """
        from verifiable_labs_envs.solvers.llm_solver import (
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        turns = int(max_turns or self.max_turns)

        history: list[dict[str, str]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        turn_rewards: list[float] = []
        turn_components: list[dict[str, float]] = []
        last_prediction: CodePrediction | None = None

        for turn_idx in range(turns):
            completion = solver.complete_turns(history)
            try:
                prediction = adapter.parse_response(completion.text, instance)
            except LLMSolverError:
                if last_prediction is None:
                    raise
                break

            scored = self.score(prediction, instance)
            turn_rewards.append(float(scored["reward"]))
            turn_components.append(dict(scored["components"]))
            last_prediction = prediction

            if turn_idx + 1 < turns:
                history.append({"role": "assistant", "content": completion.text})
                history.append(
                    {
                        "role": "user",
                        "content": adapter.build_followup_turn(
                            history, prediction, instance
                        ),
                    }
                )

        assert last_prediction is not None
        final = self.score(last_prediction, instance)
        final = self._apply_turn_penalty(final, n_turns=len(turn_rewards))
        final["meta"] = {
            **final["meta"],
            "turn_rewards": turn_rewards,
            "turn_components": turn_components,
            "n_turns": len(turn_rewards),
            "max_turns": turns,
        }
        return final


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> CodeHumanevalMultiturnEnv:
    """Factory matching the single-turn env. Calibration is reused."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, float(_BASE_DEFAULTS["alpha"]))
    return CodeHumanevalMultiturnEnv(conformal_quantile=q, max_turns=max_turns)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TURNS",
    "TURN_PENALTY_PER_EXTRA",
    "TURN_PENALTY_CAP",
    "CodeHumanevalMultiturnEnv",
    "load_environment",
    "visible_test_feedback",
    "render_feedback_message",
]
