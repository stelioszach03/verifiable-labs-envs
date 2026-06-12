"""code-humaneval-tools — tool-use code-execution RL environment.

Phase 24.D extends ``code-humaneval`` with three tool primitives the
LLM composes to converge on a working solution (D9-A locked):

- ``read_file(path)`` — return the contents of a file in the per-call
  workspace.
- ``write_file(path, content)`` — overwrite a file in the workspace.
- ``run_test(test_id)`` — run a single visible-test pytest node
  inside the D5 sandbox. Returns a short pass/fail summary.

The workspace is a per-rollout in-memory ``dict[str, str]`` (path →
content) seeded with ``solution.py`` (empty) and a visible-only
``test_solution.py`` module. Every ``run_test`` call materialises
the dict into a fresh sandbox tmpdir, runs pytest under D5 limits,
and returns truncated output.

On the final non-tool turn the LLM's submission is parsed as a
:class:`CodePrediction`. The scorer prefers the workspace's
``solution.py`` (since the tools are designed for the model to
iterate there), falling back to the parsed prediction's ``code``
field if the workspace was never written. Reward is computed
against visible ∪ hidden tests via the single-turn kernel.

Procedural-regeneration is unchanged from ``code-humaneval`` —
:class:`CodeHumanevalToolsEnv` reuses the same problem generator,
verifier, and conformal calibration. Only the rollout protocol
differs.
"""
from __future__ import annotations

import json
import re
from typing import Any

from verifiable_labs_envs.envs.code_humaneval import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.code_humaneval import (
    DEFAULT_MEM_BYTES,
    CodeHumanevalEnv,
    CodeInstance,
    CodePrediction,
    _cached_quantile,
)
from verifiable_labs_envs.sandbox import (
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

NAME = "code-humaneval-tools"
DEFAULT_MAX_TOOL_CALLS: int = 30
DEFAULT_TOOL_TIMEOUT_S: float = 5.0


# ── Tool schemas (OpenAI function-calling JSON) ──────────────────────


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Return the contents of a file in the per-call workspace. "
                "Files are seeded with `solution.py` (initially empty) and "
                "`test_solution.py` (visible test cases only). Returns the "
                "string content, or an error if the path is unknown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative path, e.g. 'solution.py'."
                        ),
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
                "Overwrite a file in the per-call workspace with the given "
                "content. The model typically writes its evolving "
                "`solution.py` here between `run_test` invocations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents (UTF-8 string).",
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
            "name": "run_test",
            "description": (
                "Run a single visible pytest case against the current "
                "workspace inside the D5-bounded sandbox. ``test_id`` is "
                "the test function name (e.g. 'test_visible_000'). "
                "Pass 'all' to run every visible case. Returns a short "
                "pass/fail summary; pytest stdout/stderr are truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": (
                            "Test function name from `test_solution.py`, "
                            "or 'all' for the full visible suite."
                        ),
                    },
                },
                "required": ["test_id"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Workspace ────────────────────────────────────────────────────────


def _format_visible_only_test_module(instance: CodeInstance) -> str:
    """Render a pytest module containing the visible tests only.

    Hidden tests are NEVER included (R10) — the model can only see
    pass/fail counts on visible cases via ``run_test``. The full
    hidden battery is reserved for the final scoring step.
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


def init_workspace(instance: CodeInstance) -> dict[str, str]:
    """Build the seed workspace handed to the rollout.

    Two files: ``solution.py`` (empty stub the model fills in) and
    ``test_solution.py`` (visible-test pytest module).
    """
    return {
        "solution.py": "",
        "test_solution.py": _format_visible_only_test_module(instance),
    }


def _safe_workspace_path(path: str) -> str:
    """Reject paths that escape the workspace.

    The caller's ``path`` is used as a dict key, so no actual
    filesystem traversal is possible — but rejecting bogus paths
    early surfaces clearer errors to the model.
    """
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("path is empty")
    if cleaned.startswith("/") or ".." in cleaned.split("/"):
        raise ValueError(f"path must be workspace-relative; got {path!r}")
    return cleaned


# ── Tool implementations ─────────────────────────────────────────────


def _tool_read_file(args: dict[str, Any], workspace: dict[str, str]) -> dict[str, Any]:
    path_raw = args.get("path", "")
    try:
        path = _safe_workspace_path(str(path_raw))
    except ValueError as exc:
        return {"error": str(exc)}
    if path not in workspace:
        return {"error": f"unknown file: {path!r}; available: {sorted(workspace)}"}
    return {"content": workspace[path]}


def _tool_write_file(args: dict[str, Any], workspace: dict[str, str]) -> dict[str, Any]:
    path_raw = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(content, str):
        return {"error": "content must be a string"}
    try:
        path = _safe_workspace_path(str(path_raw))
    except ValueError as exc:
        return {"error": str(exc)}
    workspace[path] = content
    return {"ok": True, "bytes_written": len(content.encode("utf-8"))}


def _tool_run_test(
    args: dict[str, Any],
    workspace: dict[str, str],
    *,
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> dict[str, Any]:
    """Run a visible test case (or 'all') inside the sandbox.

    The workspace dict is materialised into a fresh sandbox tmpdir on
    every call — no state leaks across run_test invocations.
    """
    test_id = str(args.get("test_id", "")).strip()
    if not test_id:
        return {"error": "missing required argument test_id"}
    files = dict(workspace)
    if "test_solution.py" not in files:
        return {"error": "workspace is missing test_solution.py"}

    # Allow either an explicit node ('test_solution.py::test_visible_000')
    # or the bare test name ('test_visible_000') — pytest accepts both.
    if test_id.lower() == "all":
        pytest_target = ["test_solution.py"]
    elif "::" in test_id:
        pytest_target = [test_id]
    else:
        pytest_target = [f"test_solution.py::{test_id}"]

    manifest = build_pytest_manifest(pytest_target, timeout_s=timeout_s)
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=manifest,
        mem_bytes=mem_bytes,
    )
    counts = parse_pytest_q_summary(result.stdout)
    out: dict[str, Any] = {
        "exit_code": int(result.exit_code),
        "passed": int(counts["passed"]),
        "failed": int(counts["failed"]) + int(counts["error"]),
        "stdout": result.stdout,
        "timed_out": bool(result.timed_out),
    }
    return out


_TOOL_DISPATCH = {
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "run_test": _tool_run_test,
}


def dispatch_tool(
    name: str,
    arguments: str | dict[str, Any],
    workspace: dict[str, str],
    *,
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> dict[str, Any]:
    """Dispatch a tool call against the per-rollout workspace.

    ``arguments`` may be a JSON string (OpenAI tool-call format) or a
    dict. Unknown tools return an ``{"error": ...}`` payload rather
    than raising — the rollout loop surfaces tool errors back to the
    model so it can recover.
    """
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {"error": f"invalid JSON arguments: {exc}"}
    else:
        args = arguments or {}
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name!r}"}
    if name == "run_test":
        return _tool_run_test(
            args, workspace, timeout_s=timeout_s, mem_bytes=mem_bytes
        )
    return handler(args, workspace)


# ── Adapter helpers ──────────────────────────────────────────────────


SYSTEM_PROMPT_TOOLS = (
    "You are an expert Python programmer. You have a per-call workspace "
    "containing two files: `solution.py` (empty — write your "
    "implementation here) and `test_solution.py` (visible test cases "
    "only — the hidden test suite is not shown).\n\n"
    "Use the tools `read_file`, `write_file`, and `run_test` to iterate "
    "on `solution.py` until your visible cases pass. The hidden suite "
    "is the held-out grading signal; do not assume visible-pass implies "
    "hidden-pass — review edge cases.\n\n"
    "When you are done, emit a final non-tool message of the form\n"
    '    {"code": "<final source>", "confidence": <float in [0, 1]>}\n'
    "The scorer reads the workspace's `solution.py` first; the JSON "
    "envelope is a fallback if you didn't write to the workspace."
)


_FENCED_RE = re.compile(r"```(?:python|json)?\s*(.+?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(instance: CodeInstance) -> str:
    return (
        "PROBLEM:\n"
        f"{instance.prompt}\n\n"
        "Available tools: `read_file`, `write_file`, `run_test`. Iterate "
        "until visible cases pass, then submit a JSON envelope.\n\n"
        "OUTPUT SCHEMA on submit:\n"
        '{"code": "<final source>", "confidence": <float in [0, 1]>}'
    )


def parse_response(text: str, instance: CodeInstance) -> CodePrediction:
    """Parse the LLM's terminating non-tool turn.

    Same shape as the single-turn adapter — permissive, falls back to
    empty code if no JSON envelope is recovered.
    """
    del instance
    cleaned = text.strip()
    candidates: list[str] = []
    fenced = _FENCED_RE.findall(cleaned)
    candidates.extend(fenced)
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        code = str(data.get("code", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return CodePrediction(code=code, raw=text, confidence=confidence)

    return CodePrediction(code="", raw=text, confidence=0.0)


# ── Env class ────────────────────────────────────────────────────────


class CodeHumanevalToolsEnv(CodeHumanevalEnv):
    """:class:`CodeHumanevalEnv` with a tool-use rollout entry point."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        super().__init__(conformal_quantile, hyperparams, weights)
        if max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be >= 0; got {max_tool_calls}")
        self.max_tool_calls = int(max_tool_calls)

    def run_rollout(
        self,
        solver: Any,
        instance: CodeInstance,
        *,
        adapter: Any = None,
        max_tool_calls: int | None = None,
    ) -> dict[str, Any]:
        """Run a tool-use rollout — alternating tool calls + final-answer turn.

        Returns the standard :meth:`score` dict with these extras in
        ``meta``:

        - ``tool_calls``: list[dict], each ``{"name": ..., "result": ...}``.
        - ``n_tool_calls``: int.
        - ``max_tool_calls``: int.
        - ``workspace_used``: bool — whether the model wrote ``solution.py``.

        ``adapter`` defaults to looking up ``code-humaneval-tools`` in
        the global EnvAdapter registry; pass an explicit adapter to
        bypass the lookup.
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

        workspace = init_workspace(instance)
        tool_timeout_s = float(
            self.hyperparams.get("tool_timeout_s", DEFAULT_TOOL_TIMEOUT_S)
        )
        history: list[dict[str, Any]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        tool_calls: list[dict[str, Any]] = []
        last_prediction: CodePrediction | None = None

        for _ in range(budget + 1):
            completion = solver.complete_turns(history, tools=TOOL_SCHEMAS)
            tool_call = getattr(completion, "tool_call", None)
            if tool_call is not None and len(tool_calls) < budget:
                result = dispatch_tool(
                    tool_call.name,
                    tool_call.arguments,
                    workspace,
                    timeout_s=tool_timeout_s,
                    mem_bytes=int(self.hyperparams["sandbox_mem_bytes"]),
                )
                tool_calls.append({"name": tool_call.name, "result": result})
                history.append(
                    {"role": "assistant", "content": completion.text or ""}
                )
                history.append(
                    {
                        "role": "tool",
                        "name": tool_call.name,
                        "content": json.dumps(result)[:4000],
                    }
                )
                continue
            try:
                last_prediction = adapter.parse_response(completion.text, instance)
            except LLMSolverError:
                if last_prediction is None:
                    raise
            break

        if last_prediction is None:
            last_prediction = CodePrediction(code="", raw="", confidence=0.0)

        # Scorer preference: workspace solution.py (where the tools
        # write) over the model's parsed JSON envelope. If both are
        # empty, score the empty prediction (zero reward).
        ws_code = workspace.get("solution.py", "").strip()
        if ws_code:
            scoring_pred = CodePrediction(
                code=ws_code,
                raw=json.dumps({"code": ws_code, "confidence": last_prediction.confidence}),
                confidence=last_prediction.confidence,
            )
            workspace_used = True
        else:
            scoring_pred = last_prediction
            workspace_used = False

        final = self.score(scoring_pred, instance)
        final["meta"] = {
            **final["meta"],
            "tool_calls": tool_calls,
            "n_tool_calls": len(tool_calls),
            "max_tool_calls": budget,
            "workspace_used": bool(workspace_used),
        }
        return final


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> CodeHumanevalToolsEnv:
    """Factory matching the single-turn env. Calibration is reused."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, float(_BASE_DEFAULTS["alpha"]))
    return CodeHumanevalToolsEnv(
        conformal_quantile=q, max_tool_calls=max_tool_calls
    )


__all__ = [
    "NAME",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_TOOL_TIMEOUT_S",
    "TOOL_SCHEMAS",
    "SYSTEM_PROMPT_TOOLS",
    "CodeHumanevalToolsEnv",
    "build_user_prompt",
    "dispatch_tool",
    "init_workspace",
    "load_environment",
    "parse_response",
]
