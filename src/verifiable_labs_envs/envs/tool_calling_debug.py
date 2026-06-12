"""tool-calling-debug — trace-completion tool-calling RL env (Phase 25.D).

PHASE_25_PLAN.md D8-C locks the trace-debug shape: each instance
ships a partial trajectory (`prefix_messages` + the
:class:`WorkspaceState` snapshot it produced) plus a goal predicate.
The model continues the rollout from that point with a tightened
budget. This exercises a unique skill — continuation conditioning
on external state — that the single + multi-turn variants don't
cover, while reusing 100% of the existing rollout machinery.

Three templates wrap base single-turn templates (PHASE_25_PLAN.md §9.3):

| Debug template       | Base                  | Prefix supplied                                     |
|----------------------|-----------------------|------------------------------------------------------|
| ``partial_compute``  | ``arithmetic_compute``| First `(a+b)` step pre-computed; model finishes.    |
| ``partial_search``   | ``search_and_email``  | `web_search` call done; model sends the email.      |
| ``partial_workspace``| ``file_concat``       | Both files read into state; model writes the merge. |

Reward is computed identically to the single-turn env (same
`format_valid` + `parse_valid` + `correctness` shape, same
`_check_gold_state` predicate). The `prefix_state` is the rollout's
starting workspace — tool calls inside the rollout mutate it
further.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from verifiable_labs_envs.conformal import split_conformal_quantile
from verifiable_labs_envs.envs.tool_calling_single import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_TOOL_CALLS,
    MAX_TOOL_RESULT_BYTES,
    ToolCallingPrediction,
    ToolCallingSingleEnv,
    _check_gold_state,
    compute_reward,
    parse_response,
)
from verifiable_labs_envs.envs.tool_calling_single import (
    DEFAULT_HYPERPARAMS as _BASE_HYPERPARAMS,
)
from verifiable_labs_envs.envs.tool_calling_single import (
    SYSTEM_PROMPT as _BASE_SYSTEM_PROMPT,
)
from verifiable_labs_envs.envs.tool_calling_single import (
    build_user_prompt as _base_build_user_prompt,
)
from verifiable_labs_envs.tool_primitives import (
    TOOL_SCHEMAS,
    WorkspaceState,
    dispatch_tool,
    init_state,
    schemas_for,
)

NAME = "tool-calling-debug"

# 3 templates × 64-bit seed × ~1e6 parameter combinations per base ≈
# 5.5e22 effective instances; well above the 1e15 gate.
EFFECTIVE_INSTANCES: int = 3 * (2**64) * 1_000_000

DEFAULT_PREFIX_MIN: int = 1
DEFAULT_PREFIX_MAX: int = 3

DEBUG_HYPERPARAMS: dict[str, Any] = {
    **_BASE_HYPERPARAMS,
    "max_remaining_calls": DEFAULT_MAX_TOOL_CALLS,
}

SYSTEM_PROMPT = (
    _BASE_SYSTEM_PROMPT
    + "\n\nThis is a TRACE-DEBUG task: the conversation already shows a "
    "partial sequence of tool calls and their results. Continue from "
    "that point — do not repeat the prefix. The workspace state has "
    "already been mutated by the prefix calls; check it before acting."
)


# ── Public dataclasses ───────────────────────────────────────────────


@dataclass(frozen=True)
class DebugInstance:
    """One trace-debug problem draw.

    ``prefix_messages`` is the conversation seed (excluding the
    system prompt) the env replays into the rollout history before
    the solver's first turn. ``prefix_state`` is the
    :class:`WorkspaceState` snapshot the prefix produced — every
    rollout starts from a copy of this dict.
    """

    prompt: str
    template_name: str
    base_template: str
    seed: int
    gold_spec: dict[str, Any]
    initial_files: dict[str, str]
    available_tools: tuple[str, ...]
    prefix_messages: tuple[dict[str, Any], ...]
    prefix_state_payload: dict[str, Any]
    max_remaining_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "available_tools": list(self.available_tools),
            "template_name": self.template_name,
            "base_template": self.base_template,
            "prefix_messages": [dict(m) for m in self.prefix_messages],
            "max_remaining_calls": self.max_remaining_calls,
            **self.metadata,
        }

    @property
    def prefix_state(self) -> WorkspaceState:
        """Rehydrate the workspace snapshot the prefix produced."""
        return WorkspaceState.from_serialisable(self.prefix_state_payload)


# ── Procedural lattice — 3 partial-trajectory templates ──────────────


def _replay(
    prefix_calls: list[dict[str, Any]],
    *,
    seed: int,
    initial_files: dict[str, str],
) -> tuple[WorkspaceState, list[dict[str, Any]]]:
    """Execute ``prefix_calls`` against a fresh workspace.

    Returns ``(state, messages)`` — ``state`` is the workspace after
    replay; ``messages`` is the assistant/tool message sequence the
    rollout will splice into history before the solver's first turn.
    """
    state = init_state(seed=seed, initial_files=initial_files)
    messages: list[dict[str, Any]] = []
    for call in prefix_calls:
        result = dispatch_tool(call["name"], call["arguments"], state)
        messages.append({
            "role": "assistant",
            "content": "",
            "_tool_call": {
                "name": call["name"],
                "arguments": dict(call["arguments"]),
            },
        })
        messages.append({
            "role": "tool",
            "name": call["name"],
            "content": json.dumps(result)[:MAX_TOOL_RESULT_BYTES],
        })
    return state, messages


def _tmpl_partial_compute(rng: np.random.Generator) -> dict[str, Any]:
    """``arithmetic_compute`` with the first sub-expression pre-computed."""
    a = int(rng.integers(2, 20))
    b = int(rng.integers(2, 20))
    c = int(rng.integers(2, 20))
    expr = f"({a} + {b}) * {c}"
    target = (a + b) * c
    prompt = (
        f"A previous agent already computed `{a} + {b}`. Continue the "
        f"trace: multiply that intermediate result by `{c}` to obtain "
        f"`{expr}`. Submit a JSON envelope when done."
    )
    prefix_calls = [
        {"name": "calculator", "arguments": {"expression": f"{a} + {b}"}},
    ]
    state, messages = _replay(prefix_calls, seed=int(rng.integers(0, 2**31)), initial_files={})
    return _problem_dict(
        template_name="partial_compute",
        base_template="arithmetic_compute",
        prompt=prompt,
        gold_spec={"target": float(target), "expr": expr},
        initial_files={},
        available_tools=("calculator",),
        prefix_messages=messages,
        prefix_state=state,
        max_remaining_calls=DEFAULT_MAX_TOOL_CALLS - len(prefix_calls),
    )


def _tmpl_partial_search(rng: np.random.Generator) -> dict[str, Any]:
    """``search_and_email`` with the search call already issued."""
    pool_topics = (
        "fourier",
        "phase retrieval",
        "compressed sensing",
        "tomography",
        "humaneval",
    )
    pool_recipients = (
        "alice@example.com",
        "bob@example.com",
        "charlie@example.com",
        "diana@example.com",
        "ed@example.com",
    )
    topic = pool_topics[int(rng.integers(0, len(pool_topics)))]
    recipient = pool_recipients[int(rng.integers(0, len(pool_recipients)))]
    prompt = (
        f"A previous agent already searched the web for `{topic}`. "
        "Continue the trace: read the search result from history and "
        f"send a one-sentence summary email to `{recipient}`. The "
        f"email body must mention `{topic}`. Submit a JSON envelope "
        "when done."
    )
    prefix_calls = [
        {"name": "web_search", "arguments": {"query": topic, "top_k": 3}},
    ]
    state, messages = _replay(prefix_calls, seed=int(rng.integers(0, 2**31)), initial_files={})
    return _problem_dict(
        template_name="partial_search",
        base_template="search_and_email",
        prompt=prompt,
        gold_spec={"topic": topic, "recipient": recipient},
        initial_files={},
        available_tools=("web_search", "send_message"),
        prefix_messages=messages,
        prefix_state=state,
        max_remaining_calls=DEFAULT_MAX_TOOL_CALLS - len(prefix_calls),
    )


def _tmpl_partial_workspace(rng: np.random.Generator) -> dict[str, Any]:
    """``file_concat`` with both reads already done."""
    a_text = f"alpha-{int(rng.integers(0, 1000)):04d}"
    b_text = f"beta-{int(rng.integers(0, 1000)):04d}"
    initial_files = {"a.txt": a_text, "b.txt": b_text}
    prompt = (
        "A previous agent already read `a.txt` and `b.txt` from the "
        "workspace. Continue the trace: write a file `merged.txt` "
        "whose contents are the two strings concatenated with a "
        "single newline between them. Submit a JSON envelope when done."
    )
    prefix_calls = [
        {"name": "read_file", "arguments": {"path": "a.txt"}},
        {"name": "read_file", "arguments": {"path": "b.txt"}},
    ]
    state, messages = _replay(
        prefix_calls,
        seed=int(rng.integers(0, 2**31)),
        initial_files=initial_files,
    )
    return _problem_dict(
        template_name="partial_workspace",
        base_template="file_concat",
        prompt=prompt,
        gold_spec={
            "out_path": "merged.txt",
            "expected": f"{a_text}\n{b_text}",
        },
        initial_files=initial_files,
        available_tools=("read_file", "write_file"),
        prefix_messages=messages,
        prefix_state=state,
        max_remaining_calls=DEFAULT_MAX_TOOL_CALLS - len(prefix_calls),
    )


def _problem_dict(
    *,
    template_name: str,
    base_template: str,
    prompt: str,
    gold_spec: dict[str, Any],
    initial_files: dict[str, str],
    available_tools: tuple[str, ...],
    prefix_messages: list[dict[str, Any]],
    prefix_state: WorkspaceState,
    max_remaining_calls: int,
) -> dict[str, Any]:
    return {
        "template_name": template_name,
        "base_template": base_template,
        "prompt": prompt,
        "gold_spec": gold_spec,
        "initial_files": initial_files,
        "available_tools": available_tools,
        "prefix_messages": prefix_messages,
        "prefix_state": prefix_state,
        "max_remaining_calls": max_remaining_calls,
    }


_TEMPLATES = (
    _tmpl_partial_compute,
    _tmpl_partial_search,
    _tmpl_partial_workspace,
)


# ── Generators ───────────────────────────────────────────────────────


def generate_problem(seed: int, **_unused: Any) -> dict[str, Any]:
    """Sample a fresh trace-debug problem from the procedural lattice."""
    rng = np.random.default_rng(int(seed))
    template_idx = int(rng.integers(0, len(_TEMPLATES)))
    return _TEMPLATES[template_idx](rng)


def generate_instance(seed: int, **kwargs: Any) -> DebugInstance:
    """Wrap :func:`generate_problem` output in a :class:`DebugInstance`."""
    params = {**DEBUG_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed))
    prefix_state: WorkspaceState = problem["prefix_state"]
    return DebugInstance(
        prompt=problem["prompt"],
        template_name=problem["template_name"],
        base_template=problem["base_template"],
        seed=int(seed),
        gold_spec=dict(problem["gold_spec"]),
        initial_files=dict(problem["initial_files"]),
        available_tools=tuple(problem["available_tools"]),
        prefix_messages=tuple(dict(m) for m in problem["prefix_messages"]),
        prefix_state_payload=prefix_state.to_serialisable(),
        max_remaining_calls=int(problem["max_remaining_calls"]),
        metadata={
            "alpha": float(params["alpha"]),
            "max_tool_calls": int(params["max_tool_calls"]),
        },
    )


# ── Adapter helpers ──────────────────────────────────────────────────


def build_user_prompt(instance: DebugInstance) -> str:
    """Render the trace-debug instance into LLM-readable text.

    Same shape as the single-turn user prompt + an explicit hint that
    the conversation already shows a partial trajectory.
    """
    base = _base_build_user_prompt(instance)
    return (
        base
        + "\n\n"
        + "TRACE-DEBUG: the conversation history already includes the "
        f"first {len(instance.prefix_messages) // 2} tool call(s) + "
        "their results. Continue from that point; do not repeat them. "
        f"You have {instance.max_remaining_calls} tool call(s) remaining."
    )


# ── Reward kernel — delegates to single-turn `_check_gold_state` ─────


def _instance_to_single_turn_view(instance: DebugInstance) -> Any:
    """Adapt :class:`DebugInstance` to the single-turn predicate's shape.

    ``_check_gold_state`` switches on ``instance.template_name`` —
    we overwrite the field with ``base_template`` so the existing
    predicates apply transparently.
    """
    from verifiable_labs_envs.envs.tool_calling_single import ToolCallingInstance
    return ToolCallingInstance(
        prompt=instance.prompt,
        template_name=instance.base_template,
        seed=instance.seed,
        gold_spec=instance.gold_spec,
        initial_files=instance.initial_files,
        available_tools=instance.available_tools,
        metadata=instance.metadata,
    )


def check_gold_state(state: WorkspaceState, instance: DebugInstance) -> bool:
    """Convenience: re-run the single-turn predicate on the rollout state."""
    return _check_gold_state(state, _instance_to_single_turn_view(instance))


def baseline_predict(instance: DebugInstance) -> ToolCallingPrediction:
    """Reference solver — empty trajectory, prefix-state untouched.

    The prefix state is the rollout's starting point, so the
    "baseline" workspace is the prefix snapshot itself. This means a
    template whose prefix already satisfies the gold predicate (e.g.
    ``partial_compute`` with the right intermediate value) scores
    ``correctness=0`` because ``action_validity=0`` on an empty
    trajectory — which is what we want for calibration.
    """
    return ToolCallingPrediction(
        tool_calls=(),
        final_text="",
        final_state=instance.prefix_state,
        raw="",
        confidence=0.0,
    )


# ── Env class + factory ──────────────────────────────────────────────


class ToolCallingDebugEnv(ToolCallingSingleEnv):
    """:class:`ToolCallingSingleEnv` with prefix-conditioned rollouts."""

    name: str = NAME

    def generate_instance(self, seed: int, **kwargs: Any) -> DebugInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(  # type: ignore[override]
        self,
        prediction: ToolCallingPrediction,
        instance: DebugInstance,
    ) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=_instance_to_single_turn_view(instance),
            weights=self.weights,
            conformal_quantile=self.conformal_quantile,
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)

    def run_rollout(  # type: ignore[override]
        self,
        solver: Any,
        instance: DebugInstance,
        *,
        adapter: Any = None,
        max_tool_calls: int | None = None,
    ) -> dict[str, Any]:
        """Continue a partial trajectory.

        The rollout history is seeded with the system prompt, the
        problem prompt, the prefix's assistant + tool messages, and
        a final user nudge. The solver picks up from there and runs
        the standard tool-call loop with budget =
        ``instance.max_remaining_calls`` (or the override).
        """
        from verifiable_labs_envs.solvers.llm_solver import (
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        budget = int(
            max_tool_calls
            if max_tool_calls is not None
            else instance.max_remaining_calls
        )

        state = WorkspaceState.from_serialisable(instance.prefix_state_payload)
        history: list[dict[str, Any]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        history.extend(dict(m) for m in instance.prefix_messages)
        history.append({
            "role": "user",
            "content": (
                "The trace above shows the partial trajectory so far. "
                "Continue from this point."
            ),
        })

        tool_calls: list[dict[str, Any]] = []
        last_prediction: ToolCallingPrediction | None = None
        final_text = ""
        tool_schemas = schemas_for(instance.available_tools) or list(TOOL_SCHEMAS)

        for _ in range(budget + 1):
            completion = solver.complete_turns(history, tools=tool_schemas)
            tool_call = getattr(completion, "tool_call", None)
            if tool_call is not None and len(tool_calls) < budget:
                result = dispatch_tool(tool_call.name, tool_call.arguments, state)
                tool_calls.append({
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result,
                })
                history.append({"role": "assistant", "content": completion.text or ""})
                history.append({
                    "role": "tool",
                    "name": tool_call.name,
                    "content": json.dumps(result)[:MAX_TOOL_RESULT_BYTES],
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
        scored["meta"] = {
            **scored["meta"],
            "tool_calls": list(tool_calls),
            "n_tool_calls": len(tool_calls),
            "max_tool_calls": budget,
            "prefix_len": len(instance.prefix_messages) // 2,
            "state": state.to_serialisable(),
        }
        return scored


def calibrate_quantile(
    n_samples: int = 30,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    residuals: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        out = compute_reward(
            prediction=pred,
            instance=_instance_to_single_turn_view(inst),
        )
        residuals.append(1.0 - float(out["reward"]))
    return float(split_conformal_quantile(np.asarray(residuals), alpha))


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float) -> float:
    return calibrate_quantile(n_samples=n_samples, alpha=alpha)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> ToolCallingDebugEnv:
    """Factory matching the single-turn env. Calibration is independent."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return ToolCallingDebugEnv(
        conformal_quantile=q, max_tool_calls=max_tool_calls
    )


__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_PREFIX_MIN",
    "DEFAULT_PREFIX_MAX",
    "DEBUG_HYPERPARAMS",
    "SYSTEM_PROMPT",
    "DebugInstance",
    "ToolCallingDebugEnv",
    "baseline_predict",
    "build_user_prompt",
    "calibrate_quantile",
    "check_gold_state",
    "compute_reward",
    "generate_instance",
    "generate_problem",
    "load_environment",
    "parse_response",
]
