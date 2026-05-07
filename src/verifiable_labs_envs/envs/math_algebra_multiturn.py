"""Environment — multi-turn algebraic-simplification (Phase 21.D).

Extends ``math-algebra`` with a 3-turn rollout:

* **Turn 1**: the LLM sees the algebra problem and proposes an answer
  + self-reported confidence.
* **Turn 2**: the server reports whether the previous answer was
  correct, and if not, which component failed (format → parse → equivalence).
  No oracle leakage — the gold expression itself is not revealed.
* **Turn 3**: same — the LLM proposes a final answer.

Reward is computed on the **final** turn's prediction with a
turn-count penalty: ``final = base * (1 − 0.05 · (n_turns − 1))``,
capped at ``0.1``. So three turns scores 0.9× the equivalent
single-turn reward — the penalty is bounded by the spec's "≤ 0.1 of
total reward" constraint.

Problem generation, scoring, calibration, and the verifier all
delegate to :mod:`verifiable_labs_envs.envs.math_algebra` —
``math_algebra_multiturn`` is a pure conversation wrapper around it.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.math_algebra import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.math_algebra import (
    Instance,
    MathAlgebraEnv,
    Prediction,
    _cached_quantile,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolver

NAME = "math-algebra-multiturn"
DEFAULT_MAX_TURNS: int = 3
# Per-extra-turn penalty as a fraction of base reward, capped at TURN_PENALTY_CAP.
TURN_PENALTY_PER_EXTRA: float = 0.05
TURN_PENALTY_CAP: float = 0.10


class MathAlgebraMultiturnEnv(MathAlgebraEnv):
    """:class:`MathAlgebraEnv` with a multi-turn rollout entry point."""

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
        """Multiply the base reward by ``(1 − penalty)`` where penalty
        scales with extra-turn count and caps at ``TURN_PENALTY_CAP``."""
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
        solver: LLMSolver,
        instance: Instance,
        *,
        adapter: EnvAdapter | None = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        """Run up to ``max_turns`` turns of ``solver`` on ``instance``.

        Returns the final-turn :meth:`score` dict with these extras in
        ``meta``:
            - ``turn_rewards``: list[float], the score after each turn.
            - ``turn_components``: list[dict], per-turn component breakdown.
            - ``n_turns``: int, the number of turns actually taken.
            - ``max_turns``: int, the cap for this rollout.
            - ``base_reward``: float, the unpenalised final-turn reward.
            - ``turn_penalty``: float, fraction subtracted for extra turns.

        If a turn returns un-parseable output after at least one good
        turn, the rollout halts and the last good prediction is scored.
        Hard parse failure on turn 1 propagates as
        :class:`LLMSolverError`.
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
        last_prediction: Prediction | None = None

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
                history.append({
                    "role": "user",
                    "content": adapter.build_followup_turn(history, prediction, instance),
                })

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
) -> MathAlgebraMultiturnEnv:
    """Factory matching the single-turn env. Calibration is reused."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 30 if fast else 200
        q = _cached_quantile(n, float(_BASE_DEFAULTS["alpha"]))
    return MathAlgebraMultiturnEnv(conformal_quantile=q, max_turns=max_turns)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TURNS",
    "TURN_PENALTY_PER_EXTRA",
    "TURN_PENALTY_CAP",
    "MathAlgebraMultiturnEnv",
    "load_environment",
]
