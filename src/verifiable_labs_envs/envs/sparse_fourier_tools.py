"""Environment 1c — tool-use sparse-Fourier recovery.

The LLM solves the standard sparse-Fourier problem but may call up to 5
Python tools before committing to a final ``(support_idx, support_amp_x1000)``
answer:

- ``fft_tool``            apply ``A = S . F`` to a candidate sparse vector.
- ``ifft_tool``           zero-fill + inverse DFT (the dirty-image heuristic).
- ``ista_tool``           run the classical OMP baseline on this instance.
- ``check_residual_tool`` report ``||y - A(x_hat)||_2`` for a candidate.

Tools reference the instance-bound measurement ``y`` and mask implicitly,
so tool-call payloads stay small. Meta records the tool-call count and
ordered tool-name sequence per episode.

Reward is computed on the final answer; tool calls do not score directly,
though frontier models should use them to improve that answer.
"""
from __future__ import annotations

import contextlib
import json
from typing import Any

import numpy as np

from verifiable_labs_envs.envs.sparse_fourier import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.sparse_fourier import (
    Instance,
    Prediction,
    SparseFourierEnv,
    _cached_quantile,
    ista_baseline,
)
from verifiable_labs_envs.forward_ops import (
    sparse_fourier_adjoint,
    sparse_fourier_forward,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolver

NAME = "sparse-fourier-recovery-tools"
DEFAULT_MAX_TOOL_CALLS: int = 5


# ──────────────────────────────────────────
# Tool schemas (OpenAI function-calling JSON)
# ──────────────────────────────────────────


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fft_tool",
            "description": (
                "Apply A = S . F to a candidate k-sparse signal and return the m complex "
                "Fourier coefficients at the observed mask positions (scaled x1000, real and "
                "imaginary parts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "support_idx": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "length-k indices of nonzero entries.",
                    },
                    "support_amp_x1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "amplitudes x1000 at those indices (same order).",
                    },
                },
                "required": ["support_idx", "support_amp_x1000"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ifft_tool",
            "description": (
                "Zero-fill the given frequency coefficients at the observed mask positions "
                "and return the real inverse-DFT signal (the \"dirty image\" of the "
                "measurement). Useful as a first-guess heuristic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "real_x1000_at_mask": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "real parts x1000 at each of the m mask positions, in mask order.",
                    },
                    "imag_x1000_at_mask": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "imaginary parts x1000 at each of the m mask positions.",
                    },
                },
                "required": ["real_x1000_at_mask", "imag_x1000_at_mask"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ista_tool",
            "description": (
                "Run the classical orthogonal-matching-pursuit (OMP) solver on THIS instance "
                "and return its support + amplitudes. Use when you want a strong baseline to "
                "compare against your own proposal."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_residual_tool",
            "description": (
                "For a proposed (support_idx, support_amp_x1000), compute the Fourier "
                "residual r = y - A(x_hat) and return its L2 norm (x1000) and max-abs (x1000)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "support_idx": {"type": "array", "items": {"type": "integer"}},
                    "support_amp_x1000": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["support_idx", "support_amp_x1000"],
                "additionalProperties": False,
            },
        },
    },
]


# ──────────────────────────────────────────
# Tool executors (pure Python, instance-bound at call time)
# ──────────────────────────────────────────


def _sparse_from_args(args: dict[str, Any], instance: Instance) -> np.ndarray:
    support = args.get("support_idx", []) or []
    amp = args.get("support_amp_x1000", []) or []
    x = np.zeros(instance.n, dtype=np.float64)
    for idx, a in zip(support, amp, strict=False):
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx_i < instance.n:
            with contextlib.suppress(TypeError, ValueError):
                x[idx_i] = float(a) / 1000.0
    return x


def _tool_fft(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    x = _sparse_from_args(args, instance)
    y_hat = sparse_fourier_forward(x.astype(np.complex128), instance.mask)
    return {
        "y_hat_re_x1000": [int(round(float(v) * 1000)) for v in y_hat.real.tolist()],
        "y_hat_im_x1000": [int(round(float(v) * 1000)) for v in y_hat.imag.tolist()],
    }


def _tool_ifft(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    re = args.get("real_x1000_at_mask", []) or []
    im = args.get("imag_x1000_at_mask", []) or []
    m = instance.mask.size
    if len(re) != m or len(im) != m:
        return {"error": f"expected {m} coefficients in each array; got re={len(re)}, im={len(im)}"}
    try:
        coeffs = np.array(
            [float(r) / 1000.0 + 1j * float(i) / 1000.0 for r, i in zip(re, im, strict=False)],
            dtype=np.complex128,
        )
    except (TypeError, ValueError) as exc:
        return {"error": f"non-numeric coefficient: {exc}"}
    z = sparse_fourier_adjoint(coeffs, instance.mask, instance.n)
    return {"signal_x1000": [int(round(float(v) * 1000)) for v in z.real.tolist()]}


def _tool_ista(args: dict[str, Any], instance: Instance) -> dict[str, Any]:  # noqa: ARG001
    pred = ista_baseline(**instance.as_inputs())
    support = pred.support_hat if pred.support_hat is not None else np.array([], dtype=np.int64)
    return {
        "support_idx": [int(i) for i in support.tolist()],
        "support_amp_x1000": [int(round(float(pred.x_hat[i]) * 1000)) for i in support.tolist()],
    }


def _tool_check_residual(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    x = _sparse_from_args(args, instance)
    y_hat = sparse_fourier_forward(x.astype(np.complex128), instance.mask)
    residual = np.asarray(instance.y) - np.asarray(y_hat)
    l2 = float(np.linalg.norm(residual))
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {
        "residual_l2_x1000": int(round(l2 * 1000)),
        "residual_max_abs_x1000": int(round(max_abs * 1000)),
    }


_TOOL_DISPATCH = {
    "fft_tool": _tool_fft,
    "ifft_tool": _tool_ifft,
    "ista_tool": _tool_ista,
    "check_residual_tool": _tool_check_residual,
}


def dispatch_tool(name: str, arguments: str | dict, instance: Instance) -> dict[str, Any]:
    """Dispatch a tool call. ``arguments`` may be a JSON string (OpenAI format) or a dict."""
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
    return handler(args, instance)


# ──────────────────────────────────────────
# Env
# ──────────────────────────────────────────


class SparseFourierToolsEnv(SparseFourierEnv):
    """``SparseFourierEnv`` with a tool-use rollout entry point."""

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

    def run_rollout_with_tools(
        self,
        solver: LLMSolver,
        instance: Instance,
        *,
        adapter: EnvAdapter | None = None,
        max_tool_calls: int | None = None,
        max_loops: int = 10,
    ) -> dict[str, Any]:
        """Run a tool-use rollout.

        The LLM is given the standard sparse-F prompt plus the 4 tool schemas.
        It may emit up to ``max_tool_calls`` tool invocations; each is
        dispatched server-side via :func:`dispatch_tool`, and the result is
        appended as a ``tool``-role message. After the cap (or when the LLM
        returns a no-tool message), we parse the final JSON answer.

        ``max_loops`` bounds the outer while-loop as a safety net against
        runaway LLMs (e.g. a model that keeps emitting tool calls after being
        asked for a final answer).
        """
        from verifiable_labs_envs.solvers.llm_solver import (
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        tool_cap = int(max_tool_calls if max_tool_calls is not None else self.max_tool_calls)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        tool_calls_total = 0
        tool_sequence: list[str] = []
        final_text: str | None = None

        for _ in range(max_loops):
            result = solver.complete_turns(messages, tools=TOOL_SCHEMAS)

            if result.tool_calls and tool_calls_total < tool_cap:
                remaining = tool_cap - tool_calls_total
                calls_this_turn = result.tool_calls[:remaining]
                messages.append({
                    "role": "assistant",
                    "content": result.text or "",
                    "tool_calls": calls_this_turn,
                })
                for call in calls_this_turn:
                    name = call.get("function", {}).get("name", "")
                    args = call.get("function", {}).get("arguments", "")
                    tool_result = dispatch_tool(name, args, instance)
                    tool_calls_total += 1
                    tool_sequence.append(name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(tool_result),
                    })
                continue

            if result.tool_calls and tool_calls_total >= tool_cap:
                # Over the cap: ask for final answer explicitly and stop tool-dispatch.
                messages.append({
                    "role": "assistant",
                    "content": result.text or "",
                    "tool_calls": result.tool_calls,
                })
                # Refuse to execute further tool calls; synthesize an error-tool reply.
                for call in result.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps({"error": "tool-call cap reached; output final JSON answer now"}),
                    })
                messages.append({
                    "role": "user",
                    "content": (
                        "Tool-call budget exhausted. Provide your final JSON answer now "
                        "with keys support_idx and support_amp_x1000. No tool calls."
                    ),
                })
                continue

            # No tool calls -> expect final content
            final_text = result.text or ""
            break

        if final_text is None:
            raise LLMSolverError(
                f"Tool-use rollout exited without a final answer after {max_loops} loops "
                f"(tool_calls={tool_calls_total})"
            )

        prediction: Prediction = adapter.parse_response(final_text, instance)
        scored = self.score(prediction, instance)
        scored["meta"] = {
            **scored["meta"],
            "tool_calls": tool_calls_total,
            "tool_sequence": tool_sequence,
            "max_tool_calls": tool_cap,
        }
        return scored


# ──────────────────────────────────────────
# Factory
# ──────────────────────────────────────────


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> SparseFourierToolsEnv:
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(
            n_samples=30,
            alpha=float(_BASE_DEFAULTS["alpha"]),
            n_bootstrap=5,
            n_iters=80,
        )
    else:
        q = _cached_quantile(
            n_samples=500,
            alpha=float(_BASE_DEFAULTS["alpha"]),
            n_bootstrap=20,
            n_iters=200,
        )
    return SparseFourierToolsEnv(conformal_quantile=q, max_tool_calls=max_tool_calls)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TOOL_CALLS",
    "TOOL_SCHEMAS",
    "SparseFourierToolsEnv",
    "dispatch_tool",
    "load_environment",
]
