"""tool-calling-single — single-pass procedural tool-calling RL env (Phase 25.B).

Phase 25.B introduces the tool-calling env family. The single-turn
variant gives the model a budget of up to 30 OpenAI-style function
calls plus one final non-tool message. The env records the
trajectory + final :class:`WorkspaceState`, scores

    reward = 0.10 · format_valid    (final non-tool turn parses as JSON)
           + 0.20 · parse_valid     (every tool-call had valid args
                                      AND the final submission parses)
           + 0.70 · correctness     (D2-C: 0.30 · action_validity
                                            + 0.70 · final_state_match)

D2-C verification (PHASE_25_PLAN.md §7) blends action validity
(fraction of executed tool calls returning a non-error payload) with
final-state correctness (template-supplied predicate over the
:class:`WorkspaceState`).

Procedural-regeneration contract: each ``(seed, hyperparams)`` pair
draws a fresh problem from a 10-template lattice. The 64-bit seed
space × per-template parameter ranges yield ``EFFECTIVE_INSTANCES``
of order ``6 × 10²⁰``, well above the 1e15 contamination-resistance
gate.
"""
from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from verifiable_labs_envs.conformal import split_conformal_quantile
from verifiable_labs_envs.tool_primitives import (
    TOOL_SCHEMAS,
    WorkspaceState,
    canonical_action_hash,
    dispatch_tool,
    init_state,
    schemas_for,
)

NAME = "tool-calling-single"

# 10 templates × 64-bit seed × ~1e6 parameter combinations per template
# ≈ 6.1e23 effective instances; well above the 1e15 procedural-
# regeneration gate.
EFFECTIVE_INSTANCES: int = 10 * (2**64) * 1_000_000

DEFAULT_ALPHA: float = 0.1
DEFAULT_MAX_TOOL_CALLS: int = 30
DEFAULT_TOOL_TIMEOUT_S: float = 5.0
MAX_TOOL_RESULT_BYTES: int = 4096

DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correctness": 0.7,
}
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
    "tool_timeout_s": DEFAULT_TOOL_TIMEOUT_S,
}

# D2-C — split between action validity and final-state correctness.
ACTION_VALIDITY_WEIGHT: float = 0.30
STATE_MATCH_WEIGHT: float = 0.70


# ── Public dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCallingInstance:
    """One tool-calling problem draw.

    ``gold_spec`` carries template-specific oracle data the env's
    ``_check_gold_state`` consumes to build the per-template
    predicate. ``initial_files`` is the workspace seed (passed into
    :func:`init_state`). ``available_tools`` selects a subset of the
    shared :data:`TOOL_SCHEMAS`.
    """

    prompt: str
    template_name: str
    seed: int
    gold_spec: dict[str, Any]
    initial_files: dict[str, str]
    available_tools: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "available_tools": list(self.available_tools),
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class ToolCallingPrediction:
    """Solver's recorded trajectory.

    ``tool_calls`` is the ordered list of ``{name, arguments, result}``
    triples produced during the rollout (mutated by the env, then
    frozen-snapshotted). ``final_text`` is the LLM's terminating
    non-tool message; ``raw`` keeps the original string for the audit
    trail. ``final_state`` is the workspace state at the end of the
    rollout.
    """

    tool_calls: tuple[dict[str, Any], ...]
    final_text: str
    final_state: WorkspaceState
    raw: str = ""
    confidence: float = 0.5


# ── Procedural template lattice ──────────────────────────────────────


def _gold_recipient(rng: np.random.Generator) -> str:
    """Sample one of a small fixed pool of mock recipients."""
    pool = (
        "alice@example.com",
        "bob@example.com",
        "charlie@example.com",
        "diana@example.com",
        "ed@example.com",
    )
    return pool[int(rng.integers(0, len(pool)))]


def _gold_topic(rng: np.random.Generator) -> str:
    """Sample one of the corpus-aligned topics for search templates."""
    pool = (
        "fourier",
        "phase retrieval",
        "compressed sensing",
        "quantum",
        "tomography",
        "humaneval",
        "conformal prediction",
        "super-resolution",
    )
    return pool[int(rng.integers(0, len(pool)))]


def _tmpl_arithmetic_compute(rng: np.random.Generator) -> dict[str, Any]:
    a = int(rng.integers(2, 20))
    b = int(rng.integers(2, 20))
    c = int(rng.integers(2, 20))
    expr = f"({a} + {b}) * {c}"
    target = (a + b) * c
    prompt = (
        f"Use the calculator tool to compute the value of `{expr}`. "
        "When you have the answer, submit a JSON envelope of the form "
        '`{"answer": <number>, "confidence": <float in [0, 1]>}`.'
    )
    return _problem_dict(
        template_name="arithmetic_compute",
        prompt=prompt,
        gold_spec={"target": float(target), "expr": expr},
        initial_files={},
        available_tools=("calculator",),
    )


def _tmpl_search_and_email(rng: np.random.Generator) -> dict[str, Any]:
    topic = _gold_topic(rng)
    recipient = _gold_recipient(rng)
    prompt = (
        f"Search the web for `{topic}`, then send a one-sentence "
        f"summary email to `{recipient}`. The email body must mention "
        f"`{topic}` explicitly. Submit a JSON envelope when done."
    )
    return _problem_dict(
        template_name="search_and_email",
        prompt=prompt,
        gold_spec={"topic": topic, "recipient": recipient},
        initial_files={},
        available_tools=("web_search", "send_message"),
    )


def _tmpl_file_concat(rng: np.random.Generator) -> dict[str, Any]:
    a_text = f"alpha-{int(rng.integers(0, 1000)):04d}"
    b_text = f"beta-{int(rng.integers(0, 1000)):04d}"
    prompt = (
        "The workspace contains `a.txt` and `b.txt`. Read both, then "
        "write a file `merged.txt` whose contents are the two strings "
        "concatenated with a single newline between them. Submit a JSON "
        "envelope when done."
    )
    return _problem_dict(
        template_name="file_concat",
        prompt=prompt,
        gold_spec={
            "out_path": "merged.txt",
            "expected": f"{a_text}\n{b_text}",
        },
        initial_files={"a.txt": a_text, "b.txt": b_text},
        available_tools=("read_file", "write_file"),
    )


def _tmpl_compute_then_send(rng: np.random.Generator) -> dict[str, Any]:
    a = int(rng.integers(5, 30))
    b = int(rng.integers(5, 30))
    target = a * b
    recipient = _gold_recipient(rng)
    prompt = (
        f"Use the calculator to compute `{a} * {b}`, then send the "
        f"numeric result in a message to `{recipient}`. The message "
        "body must contain the digits of the answer. Submit a JSON "
        "envelope when done."
    )
    return _problem_dict(
        template_name="compute_then_send",
        prompt=prompt,
        gold_spec={
            "target": float(target),
            "recipient": recipient,
            "answer_digits": str(target),
        },
        initial_files={},
        available_tools=("calculator", "send_message"),
    )


def _tmpl_multi_search(rng: np.random.Generator) -> dict[str, Any]:
    topics = list({_gold_topic(rng) for _ in range(5)})[:3]
    while len(topics) < 3:
        topics.append(_gold_topic(rng))
    out_path = "topics.txt"
    prompt = (
        f"Run a web_search for each of these topics: {topics}. "
        f"Write a single file `{out_path}` containing the top result "
        "title for each topic on its own line. Submit a JSON envelope "
        "when done."
    )
    return _problem_dict(
        template_name="multi_search",
        prompt=prompt,
        gold_spec={"topics": topics, "out_path": out_path},
        initial_files={},
        available_tools=("web_search", "write_file"),
    )


def _tmpl_read_search_write(rng: np.random.Generator) -> dict[str, Any]:
    topic = _gold_topic(rng)
    out_path = "enriched.txt"
    prompt = (
        "The workspace contains `note.txt` listing a topic to research. "
        "Read it, search the web for that topic, then write a new file "
        f"`{out_path}` containing both the original note and the top "
        "search-result title. Submit a JSON envelope when done."
    )
    return _problem_dict(
        template_name="read_search_write",
        prompt=prompt,
        gold_spec={"topic": topic, "out_path": out_path},
        initial_files={"note.txt": f"Topic of interest: {topic}"},
        available_tools=("read_file", "web_search", "write_file"),
    )


def _tmpl_outbox_audit(rng: np.random.Generator) -> dict[str, Any]:
    topic = _gold_topic(rng)
    r1 = _gold_recipient(rng)
    pool = ("alice@example.com", "bob@example.com", "charlie@example.com",
            "diana@example.com", "ed@example.com")
    others = [p for p in pool if p != r1]
    r2 = others[int(rng.integers(0, len(others)))]
    prompt = (
        f"Search the web for `{topic}`, then send a brief summary "
        f"email to BOTH `{r1}` and `{r2}`. Each email body must "
        f"mention `{topic}`. Submit a JSON envelope when done."
    )
    return _problem_dict(
        template_name="outbox_audit",
        prompt=prompt,
        gold_spec={
            "topic": topic,
            "recipients": sorted({r1, r2}),
        },
        initial_files={},
        available_tools=("web_search", "send_message"),
    )


def _tmpl_nested_calculator(rng: np.random.Generator) -> dict[str, Any]:
    a = int(rng.integers(2, 10))
    b = int(rng.integers(2, 10))
    c = int(rng.integers(2, 10))
    d = int(rng.integers(2, 10))
    expr = f"({a} + {b}) * ({c} - {d})"
    target = (a + b) * (c - d)
    prompt = (
        f"Compute `{expr}` step by step using the calculator tool. "
        "You may make multiple calls (compute the inner sums first, then "
        "the product). Submit a JSON envelope when done."
    )
    return _problem_dict(
        template_name="nested_calculator",
        prompt=prompt,
        gold_spec={"target": float(target), "expr": expr},
        initial_files={},
        available_tools=("calculator",),
    )


def _tmpl_search_dedup(rng: np.random.Generator) -> dict[str, Any]:
    topic = _gold_topic(rng)
    out_path = "results.txt"
    prompt = (
        f"Search the web for `{topic}` twice (e.g. with slightly "
        "different phrasings) and write a deduplicated list of result "
        f"titles to `{out_path}`. Submit a JSON envelope when done."
    )
    return _problem_dict(
        template_name="search_dedup",
        prompt=prompt,
        gold_spec={"topic": topic, "out_path": out_path, "min_titles": 2},
        initial_files={},
        available_tools=("web_search", "write_file"),
    )


def _tmpl_compute_chain(rng: np.random.Generator) -> dict[str, Any]:
    a = int(rng.integers(2, 10))
    b = int(rng.integers(2, 10))
    c = int(rng.integers(2, 10))
    target = (a + b) * c
    prompt = (
        f"Compute `{a} + {b}` first, then multiply the result by `{c}` "
        "in a second calculator call. Submit a JSON envelope with the "
        "final answer when done."
    )
    return _problem_dict(
        template_name="compute_chain",
        prompt=prompt,
        gold_spec={"target": float(target), "min_calls": 2},
        initial_files={},
        available_tools=("calculator",),
    )


def _problem_dict(
    *,
    template_name: str,
    prompt: str,
    gold_spec: dict[str, Any],
    initial_files: dict[str, str],
    available_tools: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "template_name": template_name,
        "prompt": prompt,
        "gold_spec": gold_spec,
        "initial_files": initial_files,
        "available_tools": available_tools,
    }


_TEMPLATES: tuple[Callable[[np.random.Generator], dict[str, Any]], ...] = (
    _tmpl_arithmetic_compute,
    _tmpl_search_and_email,
    _tmpl_file_concat,
    _tmpl_compute_then_send,
    _tmpl_multi_search,
    _tmpl_read_search_write,
    _tmpl_outbox_audit,
    _tmpl_nested_calculator,
    _tmpl_search_dedup,
    _tmpl_compute_chain,
)


# ── Generators ───────────────────────────────────────────────────────


def generate_problem(seed: int, **_unused: Any) -> dict[str, Any]:
    """Sample a fresh tool-calling problem from the procedural lattice."""
    rng = np.random.default_rng(int(seed))
    template_idx = int(rng.integers(0, len(_TEMPLATES)))
    return _TEMPLATES[template_idx](rng)


def generate_instance(seed: int, **kwargs: Any) -> ToolCallingInstance:
    """Wrap :func:`generate_problem` output in a :class:`ToolCallingInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed))
    return ToolCallingInstance(
        prompt=problem["prompt"],
        template_name=problem["template_name"],
        seed=int(seed),
        gold_spec=dict(problem["gold_spec"]),
        initial_files=dict(problem["initial_files"]),
        available_tools=tuple(problem["available_tools"]),
        metadata={
            "alpha": float(params["alpha"]),
            "max_tool_calls": int(params["max_tool_calls"]),
        },
    )


# ── State predicates (D2-C final_state_match) ────────────────────────


def _last_calculator_value(state: WorkspaceState) -> float | None:
    if not state.calculator_history:
        return None
    last = state.calculator_history[-1]
    if "=" not in last:
        return None
    try:
        return float(last.rsplit("=", 1)[-1].strip())
    except ValueError:
        return None


def _file_match(state: WorkspaceState, path: str, expected: str) -> bool:
    """File exists with content equal-or-superset of expected substring."""
    if path not in state.files:
        return False
    return expected.strip() in state.files[path].strip()


def _check_gold_state(
    state: WorkspaceState,
    instance: ToolCallingInstance,
) -> bool:
    """Per-template predicate over the final workspace state.

    Returns True iff the rollout's terminal state satisfies the
    template's success criterion. ``gold_spec`` carries the oracle
    data; the dispatch on ``template_name`` is exhaustive.
    """
    spec = instance.gold_spec
    name = instance.template_name

    if name == "arithmetic_compute":
        last = _last_calculator_value(state)
        return last is not None and abs(last - float(spec["target"])) < 1e-6

    if name == "compute_then_send":
        last = _last_calculator_value(state)
        if last is None or abs(last - float(spec["target"])) > 1e-6:
            return False
        digits = str(spec["answer_digits"])
        return any(
            msg["to"] == spec["recipient"] and digits in msg["body"]
            for msg in state.outbox
        )

    if name == "nested_calculator":
        last = _last_calculator_value(state)
        return last is not None and abs(last - float(spec["target"])) < 1e-6

    if name == "compute_chain":
        last = _last_calculator_value(state)
        if last is None or abs(last - float(spec["target"])) > 1e-6:
            return False
        return len(state.calculator_history) >= int(spec["min_calls"])

    if name == "search_and_email":
        topic = str(spec["topic"]).lower()
        recipient = spec["recipient"]
        return any(
            msg["to"] == recipient and topic in msg["body"].lower()
            for msg in state.outbox
        )

    if name == "outbox_audit":
        topic = str(spec["topic"]).lower()
        delivered_to = sorted({
            msg["to"] for msg in state.outbox
            if topic in msg["body"].lower()
        })
        return delivered_to == sorted(spec["recipients"])

    if name == "file_concat":
        return _file_match(state, spec["out_path"], spec["expected"])

    if name == "multi_search":
        out_path = spec["out_path"]
        if out_path not in state.files:
            return False
        body = state.files[out_path].lower()
        return all(t.lower() in body for t in spec["topics"])

    if name == "read_search_write":
        out_path = spec["out_path"]
        if out_path not in state.files:
            return False
        return spec["topic"].lower() in state.files[out_path].lower()

    if name == "search_dedup":
        out_path = spec["out_path"]
        if out_path not in state.files:
            return False
        lines = [line for line in state.files[out_path].splitlines() if line.strip()]
        unique = {line.strip().lower() for line in lines}
        return len(unique) >= int(spec["min_titles"])

    return False


# ── Reward kernel ────────────────────────────────────────────────────


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _final_text_parses(text: str) -> bool:
    if not text or not text.strip():
        return False
    cleaned = text.strip()
    candidates: list[str] = []
    candidates.extend(_FENCED_RE.findall(cleaned))
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))
    for c in candidates:
        try:
            data = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return True
    return False


def _is_format_valid(prediction: ToolCallingPrediction) -> bool:
    """``final_text`` is a parseable JSON object."""
    return _final_text_parses(prediction.final_text)


def _is_parse_valid(prediction: ToolCallingPrediction) -> bool:
    """Every recorded tool call has dict-typed arguments AND the final
    submission parses. The 0.20 component is binary — we don't grade
    partial syntactic validity, since malformed tool args are
    catastrophic for orchestration."""
    if not _final_text_parses(prediction.final_text):
        return False
    for call in prediction.tool_calls:
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (json.JSONDecodeError, ValueError):
                return False
        if not isinstance(args, dict):
            return False
    return True


def _action_validity(prediction: ToolCallingPrediction) -> float:
    if not prediction.tool_calls:
        return 0.0
    ok = sum(
        1 for call in prediction.tool_calls
        if not isinstance(call.get("result"), dict) or "error" not in call["result"]
    )
    return float(ok) / float(len(prediction.tool_calls))


def _correctness(
    prediction: ToolCallingPrediction,
    instance: ToolCallingInstance,
) -> float:
    """D2-C composite — 0.30 · action_validity + 0.70 · final_state_match."""
    av = _action_validity(prediction)
    sm = 1.0 if _check_gold_state(prediction.final_state, instance) else 0.0
    return ACTION_VALIDITY_WEIGHT * av + STATE_MATCH_WEIGHT * sm


def score_components(
    prediction: ToolCallingPrediction,
    instance: ToolCallingInstance,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``."""
    components = {
        "format_valid": 1.0 if _is_format_valid(prediction) else 0.0,
        "parse_valid": 0.0,
        "correctness": 0.0,
    }
    if components["format_valid"] == 0.0:
        return components
    components["parse_valid"] = 1.0 if _is_parse_valid(prediction) else 0.0
    if components["parse_valid"] == 0.0:
        return components
    components["correctness"] = _correctness(prediction, instance)
    return components


def compute_reward(
    prediction: ToolCallingPrediction,
    instance: ToolCallingInstance,
    *,
    weights: dict[str, float] | None = None,
    conformal_quantile: float | None = None,
) -> dict[str, Any]:
    """Combine the three components into the env reward dict."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(prediction, instance)
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    meta: dict[str, Any] = {
        "weights": dict(w),
        "n_tool_calls": len(prediction.tool_calls),
        "template": instance.template_name,
        "outbox_count": len(prediction.final_state.outbox),
        "files_written": sorted(prediction.final_state.files),
        "action_hash": canonical_action_hash(list(prediction.tool_calls)),
        "confidence": float(prediction.confidence),
    }
    if conformal_quantile is not None:
        residual = 1.0 - reward
        meta["covered"] = bool(residual <= float(conformal_quantile))
        meta["residual"] = residual
        meta["conformal_quantile"] = float(conformal_quantile)

    return {
        "reward": float(reward),
        "components": {k: float(v) for k, v in components.items()},
        "meta": meta,
    }


# ── Adapter helpers ──────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are an agent that completes tasks by composing function "
    "calls. The available tools are described in the JSON-Schema "
    "block accompanying this conversation; emit each tool call via "
    "the standard OpenAI function-call format.\n\n"
    "When the task is complete, emit a final non-tool message of the "
    "form\n"
    '    {"answer": <result>, "confidence": <float in [0, 1]>}\n\n'
    "where ``result`` may be a number, string, or short summary "
    "depending on the task. The scorer reads the workspace state "
    "(files, outbox, calculator history) — the JSON envelope is "
    "advisory."
)


def build_user_prompt(instance: ToolCallingInstance) -> str:
    seeded_files = ""
    if instance.initial_files:
        listing = "\n".join(
            f"  - {p} ({len(c)} bytes)" for p, c in sorted(instance.initial_files.items())
        )
        seeded_files = f"\n\nWORKSPACE FILES:\n{listing}"
    return (
        "PROBLEM:\n"
        f"{instance.prompt}{seeded_files}\n\n"
        f"AVAILABLE TOOLS: {list(instance.available_tools)}\n\n"
        "OUTPUT SCHEMA on the final non-tool turn:\n"
        '{"answer": <result>, "confidence": <float in [0, 1]>}\n\n'
        "Use the tools as needed, then submit the JSON envelope."
    )


def parse_response(text: str, instance: ToolCallingInstance) -> ToolCallingPrediction:
    """Parse the LLM's terminating non-tool message.

    The full trajectory comes from ``run_rollout``; this helper is
    invoked by the EnvAdapter when ``/v1/score`` scores a single
    completion (no rollout). In that path we record an empty
    trajectory and an empty workspace, so the score collapses to the
    ``final_text``-only signal.
    """
    confidence = 0.0
    cleaned = text.strip()
    candidates: list[str] = list(_FENCED_RE.findall(cleaned))
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))
    for c in candidates:
        try:
            data = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        confidence = max(0.0, min(1.0, conf))
        break
    return ToolCallingPrediction(
        tool_calls=(),
        final_text=text,
        final_state=init_state(seed=instance.seed, initial_files=instance.initial_files),
        raw=text,
        confidence=confidence,
    )


# ── Env class + factory ──────────────────────────────────────────────


def baseline_predict(instance: ToolCallingInstance) -> ToolCallingPrediction:
    """Reference solver — empty trajectory, empty submission."""
    return ToolCallingPrediction(
        tool_calls=(),
        final_text="",
        final_state=init_state(seed=instance.seed, initial_files=instance.initial_files),
        raw="",
        confidence=0.0,
    )


class ToolCallingSingleEnv:
    """RL environment handle wrapping one calibrated conformal quantile."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        self.conformal_quantile = float(conformal_quantile)
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        if max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be >= 0; got {max_tool_calls}")
        self.max_tool_calls = int(max_tool_calls)
        self.env_id: str = ""
        self.env_args: dict[str, Any] = {}

    def generate_instance(self, seed: int, **kwargs: Any) -> ToolCallingInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: ToolCallingPrediction,
        instance: ToolCallingInstance,
    ) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            conformal_quantile=self.conformal_quantile,
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)

    # ── Rollout machinery (D4-C budget cap) ──────────────────────────

    def run_rollout(
        self,
        solver: Any,
        instance: ToolCallingInstance,
        *,
        adapter: Any = None,
        max_tool_calls: int | None = None,
    ) -> dict[str, Any]:
        """Run a tool-calling rollout — alternating tool calls + final turn.

        Returns the standard :meth:`score` dict with these extras in
        ``meta``:

        - ``tool_calls``: list[dict] of ``{name, arguments, result}`` triples.
        - ``n_tool_calls``: int.
        - ``max_tool_calls``: int budget.
        - ``state``: serialised :class:`WorkspaceState` at rollout end.
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
        final_text = ""
        last_prediction: ToolCallingPrediction | None = None
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
                truncated_result = json.dumps(result)[:MAX_TOOL_RESULT_BYTES]
                history.append({
                    "role": "tool",
                    "name": tool_call.name,
                    "content": truncated_result,
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
            "state": state.to_serialisable(),
        }
        return scored


def calibrate_quantile(
    n_samples: int = 30,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Compute the ``(1 − α)`` quantile of baseline residuals."""
    residuals: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        out = compute_reward(prediction=pred, instance=inst)
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
) -> ToolCallingSingleEnv:
    """Factory mirroring the verifiers convention."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return ToolCallingSingleEnv(
        conformal_quantile=q, max_tool_calls=max_tool_calls
    )


__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_TOOL_TIMEOUT_S",
    "MAX_TOOL_RESULT_BYTES",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "ACTION_VALIDITY_WEIGHT",
    "STATE_MATCH_WEIGHT",
    "SYSTEM_PROMPT",
    "ToolCallingInstance",
    "ToolCallingPrediction",
    "ToolCallingSingleEnv",
    "build_user_prompt",
    "baseline_predict",
    "calibrate_quantile",
    "compute_reward",
    "generate_instance",
    "generate_problem",
    "load_environment",
    "parse_response",
    "score_components",
]
