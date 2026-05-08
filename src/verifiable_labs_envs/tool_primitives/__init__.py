"""Shared mock-tool library for tool-calling envs (Phase 25.B).

PHASE_25_PLAN.md D10-A locks a single shared library of OpenAI-style
function-calling primitives that the three tool-calling envs
(`tool-calling-single`, `tool-calling-multiturn`, `tool-calling-debug`)
all import. Five v0.0.1 primitives:

| Tool          | Effect                                                       |
|---------------|---------------------------------------------------------------|
| `calculator`  | AST-walked arithmetic eval; appends to `state.calculator_history`. |
| `web_search`  | Mock corpus, deterministic ranking by token overlap.          |
| `read_file`   | Read a path from `state.files` (per-rollout dict).            |
| `write_file`  | Write a path into `state.files`.                              |
| `send_message`| Append delivery dict to `state.outbox`.                       |

All five are pure functions of `(args, state)`. State is the
:class:`WorkspaceState` dataclass — passed by reference, mutated in
place. Determinism is end-to-end: same `(seed, action_sequence)`
yields a byte-identical state.

PHASE_25_PLAN.md D2-C verification: action validity comes from the
fraction of tool calls returning a non-`error` payload.
PHASE_25_PLAN.md D5-B locks the soft-fail convention: tool errors
return as ``{"error": "..."}`` dicts; the rollout loop appends them
to the message history exactly like a successful result.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ── WorkspaceState ────────────────────────────────────────────────────


@dataclass
class WorkspaceState:
    """Per-rollout mutable state shared by all tool calls.

    Seeded from ``init_state(seed, initial_files=...)``. Tools mutate
    the dict-typed fields in place; the :class:`Trajectory` recorder
    in the env captures the final state at score time.
    """

    files: dict[str, str] = field(default_factory=dict)
    outbox: list[dict[str, Any]] = field(default_factory=list)
    calculator_history: list[str] = field(default_factory=list)
    web_search_calls: list[str] = field(default_factory=list)
    seed: int = 0

    def to_serialisable(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dict for hashing / persistence."""
        return {
            "files": dict(self.files),
            "outbox": [dict(m) for m in self.outbox],
            "calculator_history": list(self.calculator_history),
            "web_search_calls": list(self.web_search_calls),
            "seed": int(self.seed),
        }

    @classmethod
    def from_serialisable(cls, data: dict[str, Any]) -> WorkspaceState:
        return cls(
            files=dict(data.get("files") or {}),
            outbox=[dict(m) for m in (data.get("outbox") or [])],
            calculator_history=list(data.get("calculator_history") or []),
            web_search_calls=list(data.get("web_search_calls") or []),
            seed=int(data.get("seed", 0)),
        )


def init_state(
    seed: int,
    *,
    initial_files: dict[str, str] | None = None,
) -> WorkspaceState:
    """Build a fresh :class:`WorkspaceState` seeded from ``seed``."""
    return WorkspaceState(
        files=dict(initial_files or {}),
        outbox=[],
        calculator_history=[],
        web_search_calls=[],
        seed=int(seed),
    )


# ── calculator (AST-walked safe eval) ────────────────────────────────


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}
_MATH_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _safe_eval_arithmetic(expr: str) -> float:
    """Evaluate ``expr`` as a pure arithmetic expression.

    Walks the AST and rejects anything outside ``BinOp``, ``UnaryOp``,
    ``Constant``, ``Name`` (whitelisted math constants), and ``Call``
    (whitelisted math functions). Defends against arbitrary Python
    execution.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc}") from exc

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"non-numeric constant: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported operator: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARYOPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id in _MATH_NAMES:
                return _MATH_NAMES[node.id]
            raise ValueError(f"undefined name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct function calls allowed")
            fn = _MATH_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"undefined function: {node.func.id}")
            args = [_eval(a) for a in node.args]
            if node.keywords:
                raise ValueError("keyword arguments not supported")
            return float(fn(*args))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return float(_eval(tree))


def _tool_calculator(args: dict[str, Any], state: WorkspaceState) -> dict[str, Any]:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return {"error": "calculator: missing required argument expression"}
    try:
        value = _safe_eval_arithmetic(expr)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"calculator: {exc}"}
    state.calculator_history.append(f"{expr} = {value}")
    return {"value": value}


# ── web_search (mock corpus) ─────────────────────────────────────────


# The mock corpus spans the topic distribution used by the v0.0.1
# templates. Keep entries terse — most templates only inspect titles
# or a substring of the snippet.
_MOCK_CORPUS: list[dict[str, Any]] = [
    {"id": 0, "title": "Fourier transform basics", "snippet": "The Fourier transform decomposes a signal into its frequency components.", "url": "https://example.com/fourier-basics"},
    {"id": 1, "title": "Sparse Fourier recovery via OMP", "snippet": "Orthogonal matching pursuit reconstructs sparse signals from a small number of Fourier samples.", "url": "https://example.com/sparse-fourier"},
    {"id": 2, "title": "Phase retrieval algorithms", "snippet": "Gerchberg-Saxton and HIO are classical phase-retrieval methods that operate on magnitude-only measurements.", "url": "https://example.com/phase-retrieval"},
    {"id": 3, "title": "Compressed sensing tutorial", "snippet": "Compressed sensing recovers high-dimensional sparse signals from far fewer measurements than Nyquist demands.", "url": "https://example.com/cs"},
    {"id": 4, "title": "MRI knee reconstruction", "snippet": "Accelerated MRI from undersampled k-space data uses parallel imaging and compressed sensing.", "url": "https://example.com/mri-knee"},
    {"id": 5, "title": "CT tomography fundamentals", "snippet": "Filtered back-projection inverts the Radon transform for CT image reconstruction.", "url": "https://example.com/ct-tomography"},
    {"id": 6, "title": "Quantum error correction", "snippet": "The surface code is a leading candidate for fault-tolerant quantum computation.", "url": "https://example.com/quantum-ec"},
    {"id": 7, "title": "Symbolic regression overview", "snippet": "Symbolic regression searches the space of mathematical expressions to fit data.", "url": "https://example.com/symbolic-regression"},
    {"id": 8, "title": "Numerical linear algebra", "snippet": "QR, SVD, and Cholesky decompositions underpin most scientific-computing pipelines.", "url": "https://example.com/numerical-linalg"},
    {"id": 9, "title": "Reinforcement learning basics", "snippet": "RL agents learn policies by interacting with an environment and observing rewards.", "url": "https://example.com/rl-basics"},
    {"id": 10, "title": "Conformal prediction theory", "snippet": "Conformal prediction provides distribution-free coverage guarantees on prediction intervals.", "url": "https://example.com/conformal"},
    {"id": 11, "title": "Image super-resolution", "snippet": "Super-resolution upscales low-resolution images by exploiting natural-image priors.", "url": "https://example.com/super-resolution"},
    {"id": 12, "title": "DIV2K dataset summary", "snippet": "DIV2K is a 1000-image super-resolution benchmark with high-resolution ground truth.", "url": "https://example.com/div2k"},
    {"id": 13, "title": "Coherent diffraction imaging", "snippet": "CDI reconstructs nano-scale samples from oversampled diffraction patterns without lenses.", "url": "https://example.com/cdi"},
    {"id": 14, "title": "LoDoPaB CT dataset", "snippet": "LoDoPaB is a large-scale low-dose CT reconstruction benchmark with paired ground truth.", "url": "https://example.com/lodopab"},
    {"id": 15, "title": "Algebraic simplification", "snippet": "SymPy provides simplify, expand, factor, and substitute primitives for symbolic algebra.", "url": "https://example.com/algebra"},
    {"id": 16, "title": "Pytest fundamentals", "snippet": "Pytest discovers tests by naming convention and supports fixtures, parametrisation, and plugins.", "url": "https://example.com/pytest"},
    {"id": 17, "title": "HumanEval benchmark", "snippet": "HumanEval is a 164-problem code-completion benchmark used to score Python programmers.", "url": "https://example.com/humaneval"},
    {"id": 18, "title": "Procedural generation", "snippet": "Procedural generation synthesises content from a small seed plus a parameterised lattice.", "url": "https://example.com/procgen"},
    {"id": 19, "title": "OpenAI tool calling", "snippet": "The OpenAI API supports JSON-Schema function calling so models can request structured tool invocations.", "url": "https://example.com/tool-calling"},
]


def _tokenise(text: str) -> set[str]:
    """Lowercase + alnum-split tokeniser used by the mock ranker."""
    out: set[str] = set()
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.add("".join(cur))
                cur = []
    if cur:
        out.add("".join(cur))
    return out


def _overlap(doc: dict[str, Any], query_tokens: set[str]) -> int:
    """Token-overlap score between a corpus doc and the query tokens."""
    doc_tokens = _tokenise(doc.get("title", "")) | _tokenise(doc.get("snippet", ""))
    return len(doc_tokens & query_tokens)


def _tool_web_search(args: dict[str, Any], state: WorkspaceState) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "web_search: empty query"}
    try:
        top_k = int(args.get("top_k", 3))
    except (TypeError, ValueError):
        return {"error": "web_search: top_k must be an integer"}
    if top_k < 1 or top_k > 10:
        return {"error": "web_search: top_k must be in [1, 10]"}

    state.web_search_calls.append(query)
    query_tokens = _tokenise(query)
    # Stable sort: primary key is overlap (descending), secondary is id
    # (ascending) so ties are deterministic.
    scored = sorted(
        _MOCK_CORPUS,
        key=lambda doc: (-_overlap(doc, query_tokens), doc["id"]),
    )
    return {"results": [dict(d) for d in scored[:top_k]]}


# ── read_file / write_file (workspace-scoped) ────────────────────────


def _safe_workspace_path(path: str) -> str:
    """Validate a workspace-relative path before using it as a dict key."""
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("path is empty")
    if cleaned.startswith("/") or ".." in cleaned.split("/"):
        raise ValueError(f"path must be workspace-relative; got {path!r}")
    return cleaned


def _tool_read_file(args: dict[str, Any], state: WorkspaceState) -> dict[str, Any]:
    path_raw = args.get("path", "")
    try:
        path = _safe_workspace_path(str(path_raw))
    except ValueError as exc:
        return {"error": f"read_file: {exc}"}
    if path not in state.files:
        return {
            "error": f"read_file: unknown file {path!r}; "
            f"available: {sorted(state.files)}"
        }
    return {"content": state.files[path]}


def _tool_write_file(args: dict[str, Any], state: WorkspaceState) -> dict[str, Any]:
    path_raw = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(content, str):
        return {"error": "write_file: content must be a string"}
    try:
        path = _safe_workspace_path(str(path_raw))
    except ValueError as exc:
        return {"error": f"write_file: {exc}"}
    state.files[path] = content
    return {"ok": True, "bytes_written": len(content.encode("utf-8"))}


# ── send_message (mock outbox) ───────────────────────────────────────


def _tool_send_message(args: dict[str, Any], state: WorkspaceState) -> dict[str, Any]:
    to = str(args.get("to", "")).strip()
    body = str(args.get("body", "")).strip()
    if not to:
        return {"error": "send_message: missing required argument to"}
    if not body:
        return {"error": "send_message: missing required argument body"}
    state.outbox.append({"to": to, "body": body[:4000]})
    return {"ok": True, "delivery_id": f"msg_{len(state.outbox):06d}"}


# ── Schemas + dispatcher ─────────────────────────────────────────────


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a Python-style arithmetic expression. Returns "
                "the numeric value, or an error if the expression cannot "
                "be parsed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '3 * (4 + 5)'.",
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search a small mock corpus for documents matching the query. "
                "Returns up to top_k results ranked by token overlap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1..10).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the per-rollout workspace. Returns the "
                "string content or an error if the path is unknown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a file into the per-rollout workspace. Overwrites if "
                "the path already exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Append a message to the rollout's outbox. Returns a "
                "delivery_id for traceability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient identifier (e.g. an email address).",
                    },
                    "body": {
                        "type": "string",
                        "description": "Message body.",
                    },
                },
                "required": ["to", "body"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_DISPATCH: dict[str, Callable[[dict[str, Any], WorkspaceState], dict[str, Any]]] = {
    "calculator": _tool_calculator,
    "web_search": _tool_web_search,
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "send_message": _tool_send_message,
}


def schemas_for(names: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    """Return the subset of :data:`TOOL_SCHEMAS` matching ``names``.

    Order is preserved relative to ``names`` so the model-facing tool
    list is deterministic per env config.
    """
    by_name = {s["function"]["name"]: s for s in TOOL_SCHEMAS}
    return [by_name[n] for n in names if n in by_name]


def dispatch_tool(
    name: str,
    arguments: str | dict[str, Any] | None,
    state: WorkspaceState,
) -> dict[str, Any]:
    """Dispatch a single tool call against the workspace.

    ``arguments`` may be a JSON string (OpenAI format), a dict, or
    None. Unknown tools / malformed arguments surface as soft-fail
    error payloads (D5-B); the caller should treat them as
    non-terminal.
    """
    if isinstance(arguments, str):
        cleaned = arguments.strip()
        if not cleaned:
            args: dict[str, Any] = {}
        else:
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                return {"error": f"invalid JSON arguments: {exc}"}
            if not isinstance(parsed, dict):
                return {"error": "arguments must be a JSON object"}
            args = parsed
    elif arguments is None:
        args = {}
    elif isinstance(arguments, dict):
        args = arguments
    else:
        return {"error": f"arguments must be a JSON object; got {type(arguments).__name__}"}

    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name!r}"}
    return handler(args, state)


# ── Cache helper ─────────────────────────────────────────────────────


def canonical_action_hash(tool_calls: list[dict[str, Any]]) -> str:
    """SHA-256 truncated-to-16 hex of the canonical tool-call sequence.

    Used as the cache key in D9-B (per-process LRU on
    ``(env_id, seed, action_hash)``). Canonicalisation:

    1. Each call is reduced to ``{"name": <str>, "arguments": <dict>}``.
    2. Argument dicts are JSON-serialised with ``sort_keys=True``.
    3. The list is JSON-serialised with the same flag.
    4. SHA-256 hex digest, first 16 chars.

    Result types are NOT folded into the hash — the same
    ``(name, arguments)`` sequence yields the same hash regardless of
    tool-result drift, which is exactly the cache key we want.
    """
    canonical: list[dict[str, Any]] = []
    for call in tool_calls:
        name = str(call.get("name", ""))
        raw_args = call.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        canonical.append({"name": name, "arguments": args})
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_DISPATCH",
    "WorkspaceState",
    "canonical_action_hash",
    "dispatch_tool",
    "init_state",
    "schemas_for",
]
