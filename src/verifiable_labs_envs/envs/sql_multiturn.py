"""sql-multiturn — multi-turn text-to-SQL RL env (Phase 26.C).

Extends ``sql-single-turn`` with a 3-turn rollout (D6-A locked,
matching math-algebra-multiturn verbatim):

* **Turn 1**: the LLM sees the problem + schema and proposes a query.
* **Turn 2**: the env runs the query and reports parse status / row
  count / row-count match status. The gold result-set itself is
  NEVER serialised into the feedback (R10 carry-over).
* **Turn 3**: same — the LLM proposes the final query.

Reward is computed on the **final** turn's query against the gold
result-set, with the same per-extra-turn penalty math as
math-algebra-multiturn:

    final = base * (1 - min(0.05 · (n_turns - 1), 0.10))

Three turns scores 0.9× the equivalent single-turn reward.

Problem generation, scoring, calibration, and the verifier all
delegate to :mod:`verifiable_labs_envs.envs.sql_single_turn` —
``sql_multiturn`` is a pure conversation wrapper around it.
"""
from __future__ import annotations

import contextlib
from typing import Any

from verifiable_labs_envs.envs.sql_single_turn import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.sql_single_turn import (
    DEFAULT_MAX_QUERY_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    SqlInstance,
    SqlPrediction,
    SqlSingleTurnEnv,
    _cached_quantile,
    score_components,
)
from verifiable_labs_envs.sql_primitives import (
    compare_result_sets,
    execute_query_sync,
    is_read_only_query,
    query_has_order_by,
)

NAME = "sql-multiturn"
DEFAULT_MAX_TURNS: int = 3
TURN_PENALTY_PER_EXTRA: float = 0.05
TURN_PENALTY_CAP: float = 0.10


def query_diagnostic(
    *,
    instance: SqlInstance,
    query: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
) -> dict[str, Any]:
    """Run the candidate query against the sandbox and return a
    diagnostic dict for verifier feedback.

    Returns ``{parse_error, predicted_row_count, gold_row_count,
    correctness_match}``. Result rows are NOT included — only
    counts + parse status. R10-safe.
    """
    diagnostic: dict[str, Any] = {
        "parse_error": None,
        "predicted_row_count": None,
        "gold_row_count": len(instance.gold_result_rows),
        "correctness_match": False,
    }
    if not query or not query.strip():
        diagnostic["parse_error"] = "empty query"
        return diagnostic
    ok, reason = is_read_only_query(query)
    if not ok:
        diagnostic["parse_error"] = reason
        return diagnostic
    schema_sql = list(
        instance.schema.create_statements + instance.schema.seed_statements
    )
    result = execute_query_sync(
        schema_sql=schema_sql,
        query=query,
        max_rows=max_rows,
        timeout_s=timeout_s,
        max_query_bytes=max_query_bytes,
    )
    if not result.success:
        diagnostic["parse_error"] = result.error
        return diagnostic
    diagnostic["predicted_row_count"] = result.rowcount
    ordered = (
        instance.gold_query_is_ordered or query_has_order_by(query)
    ) and instance.gold_query_is_ordered
    diagnostic["correctness_match"] = compare_result_sets(
        instance.gold_result_rows,
        result.rows,
        ordered=ordered,
    )
    return diagnostic


def render_sql_feedback(diagnostic: dict[str, Any]) -> str:
    """User-message body shown between turns.

    Verifier-derived only — never references the gold rows
    (R10 carry-over). The feedback ladder (PHASE_26_PLAN.md §9):

    - parse_error → "Your query did not parse: ..."
    - predicted_row_count == 0 AND gold > 0 → "0 rows; expected M"
    - row counts mismatch → "you returned N; expected M"
    - row counts match but content mismatch → review columns/order
    - correctness_match → "Your previous query was correct."
    """
    parse_error = diagnostic.get("parse_error")
    if parse_error is not None:
        return (
            "FEEDBACK on your previous turn:\n"
            f"Your query did not parse or run: {parse_error}. Re-check "
            "the syntax (SQLite dialect; SELECT-only; quote string "
            "literals with single quotes)."
        )
    pred = int(diagnostic.get("predicted_row_count") or 0)
    gold = int(diagnostic.get("gold_row_count") or 0)
    if diagnostic.get("correctness_match"):
        return (
            "FEEDBACK on your previous turn:\n"
            "Your previous query was correct. You may submit the same "
            "query (or an equivalent form) for the final turn."
        )
    if pred == 0 and gold > 0:
        return (
            "FEEDBACK on your previous turn:\n"
            f"Your query returned 0 rows. The expected result has {gold} "
            "row(s). Review join predicates and filter clauses."
        )
    if pred != gold:
        return (
            "FEEDBACK on your previous turn:\n"
            f"Your query returned {pred} row(s); the expected result has "
            f"{gold} row(s). Review aggregation, filters, and ordering."
        )
    return (
        "FEEDBACK on your previous turn:\n"
        f"Your query returned the correct row count ({gold}) but the "
        "row contents do not match. Review column selection and "
        "ordering."
    )


class SqlMultiturnEnv(SqlSingleTurnEnv):
    """:class:`SqlSingleTurnEnv` with a multi-turn rollout entry point.

    Reward, schema, gold query, and result-set comparator are
    identical to the single-turn env; only the rollout protocol
    differs (verifier feedback between turns + per-extra-turn
    penalty).
    """

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
        """Multiply the base reward by ``(1 − penalty)`` (D6-A parity)."""
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
        instance: SqlInstance,
        *,
        adapter: Any = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        """Run up to ``max_turns`` turns of ``solver`` on ``instance``.

        Returns the standard :meth:`score` dict with these extras in
        ``meta``:

        - ``turn_rewards``: list[float], score after each turn.
        - ``turn_components``: list[dict], per-turn component breakdown.
        - ``n_turns``: int, the number of turns actually taken.
        - ``max_turns``: int, the cap for this rollout.
        - ``base_reward``: float, unpenalised final-turn reward.
        - ``turn_penalty``: float, fraction subtracted for extra turns.
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
        last_prediction: SqlPrediction | None = None

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
                with contextlib.suppress(LLMSolverError):
                    history.append({
                        "role": "user",
                        "content": adapter.build_followup_turn(
                            history, prediction, instance
                        ),
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
) -> SqlMultiturnEnv:
    """Factory matching the single-turn env. Calibration is reused."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, float(_BASE_DEFAULTS["alpha"]))
    return SqlMultiturnEnv(conformal_quantile=q, max_turns=max_turns)


# Re-export for convenience.
def baseline_predict(instance: SqlInstance) -> SqlPrediction:
    from verifiable_labs_envs.envs.sql_single_turn import (
        baseline_predict as _baseline_predict,
    )
    return _baseline_predict(instance)


def generate_instance(seed: int, **kwargs: Any) -> SqlInstance:
    from verifiable_labs_envs.envs.sql_single_turn import (
        generate_instance as _generate_instance,
    )
    return _generate_instance(seed, **kwargs)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TURNS",
    "TURN_PENALTY_PER_EXTRA",
    "TURN_PENALTY_CAP",
    "SqlMultiturnEnv",
    "baseline_predict",
    "generate_instance",
    "load_environment",
    "query_diagnostic",
    "render_sql_feedback",
    "score_components",
]
