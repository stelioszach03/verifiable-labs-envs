"""LLM-based solver infrastructure.

Two abstractions:
- ``LLMSolver`` — transport (OpenRouter chat completions, retries, error handling)
- ``EnvAdapter`` — per-environment prompt building + response parsing

Typical use::

    from verifiable_labs_envs import load_environment
    from verifiable_labs_envs.solvers import OpenRouterSolver

    env = load_environment("sparse-fourier-recovery")
    solver = OpenRouterSolver(model="anthropic/claude-haiku-4.5")

    instance = env.generate_instance(seed=0)
    pred = solver.solve(env.name, instance)  # calls registered adapter
    result = env.score(pred, instance)

No adapters are registered in this module — they live in
``verifiable_labs_envs.solvers.adapters`` and are registered on import.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

HAS_OPENROUTER_KEY: bool = bool(os.environ.get("OPENROUTER_API_KEY"))


class LLMSolverError(RuntimeError):
    """Raised when the LLM response cannot be parsed into a valid Prediction."""


@dataclass(frozen=True)
class CompletionResult:
    """Raw output of one chat-completion call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str  # as reported by the server (routing may have changed it)
    usd_cost: float | None = None  # populated if the provider reports it
    tool_calls: list[dict[str, Any]] | None = None  # populated when tools= is passed and model responds with a tool call


# ──────────────────────────────────────────
# LLMSolver base
# ──────────────────────────────────────────


class LLMSolver(ABC):
    """Transport-only abstraction. Subclasses implement ``.complete()`` and
    ``.complete_turns()`` for single-turn and multi-turn chat respectively."""

    model: str
    temperature: float
    max_tokens: int
    timeout_s: float
    label: str

    @abstractmethod
    def complete(self, system: str, user: str) -> CompletionResult:
        """Single chat-completion call. Must raise ``LLMSolverError`` on unrecoverable failure."""

    @abstractmethod
    def complete_turns(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Multi-turn chat-completion call. ``messages`` is an OpenAI-format
        list of ``{'role': 'system'|'user'|'assistant'|'tool', 'content': ...}``
        dicts; ``assistant`` messages may additionally carry a ``tool_calls``
        key when the previous response included tool calls.

        If ``tools`` is provided (OpenAI function-calling JSON schema list),
        the model may respond with ``tool_calls`` instead of ``content``.
        That case populates ``CompletionResult.tool_calls``; caller is
        responsible for dispatching the tool calls, appending tool-role
        result messages, and calling again.

        Used by multi-turn and tool-use environments.
        """

    def solve(self, env_name: str, instance: Any) -> Any:
        """Build the prompt via the registered adapter, call the model, parse the response."""
        if env_name not in _ADAPTERS:
            registered = ", ".join(sorted(_ADAPTERS)) or "<none>"
            raise LLMSolverError(
                f"No LLM adapter registered for env '{env_name}'. Registered: {registered}."
            )
        adapter = _ADAPTERS[env_name]
        result = self.complete(adapter.system_prompt, adapter.build_user_prompt(instance))
        try:
            return adapter.parse_response(result.text, instance)
        except LLMSolverError:
            raise
        except Exception as exc:
            # Surface parse failures with full context so debugging is tractable.
            raise LLMSolverError(
                f"Adapter {type(adapter).__name__} failed to parse response from {self.label}: {exc}"
            ) from exc


# ──────────────────────────────────────────
# OpenRouter concrete solver
# ──────────────────────────────────────────


class OpenRouterSolver(LLMSolver):
    """OpenRouter chat-completion solver via the ``openai`` SDK.

    Transport config:
    - ``base_url`` = ``https://openrouter.ai/api/v1``
    - ``api_key`` from ``OPENROUTER_API_KEY`` environment variable
    - Default headers ``HTTP-Referer`` + ``X-Title`` for OpenRouter attribution
    - ``max_retries=3`` at the SDK level (exponential backoff on 429 / 5xx)
    - Per-request ``timeout`` (default 120 s)

    Raises ``RuntimeError`` at construction if the API key is unset; callers
    that want to skip silently should check ``HAS_OPENROUTER_KEY`` first.
    """

    REFERER = "https://github.com/stelioszach03/verifiable-labs-envs"
    TITLE = "verifiable-labs-envs"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout_s: float = 120.0,
        label: str | None = None,
        max_retries: int = 3,
    ) -> None:
        if not HAS_OPENROUTER_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set in the environment. "
                "Set it (or load .env via python-dotenv) before constructing OpenRouterSolver."
            )
        # Lazy import so installs without openai can still use FakeLLMSolver.
        from openai import OpenAI

        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.label = label or model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={"HTTP-Referer": self.REFERER, "X-Title": self.TITLE},
            max_retries=int(max_retries),
            timeout=self.timeout_s,
        )

    def _invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Shared transport: send ``messages``, return ``CompletionResult``."""
        start = time.perf_counter()
        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"usage": {"include": True}},
        )
        if tools:
            create_kwargs["tools"] = tools
        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:  # openai.APIError, TimeoutError, etc.
            raise LLMSolverError(
                f"OpenRouter call for model={self.model!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        latency = time.perf_counter() - start

        message = response.choices[0].message
        text = message.content or ""
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        usd_cost: float | None = None
        raw_cost = getattr(usage, "cost", None)
        if raw_cost is not None:
            try:
                usd_cost = float(raw_cost)
            except (TypeError, ValueError):
                usd_cost = None

        tool_calls: list[dict[str, Any]] | None = None
        raw_tool_calls = getattr(message, "tool_calls", None) or None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                tool_calls.append({
                    "id": getattr(tc, "id", "") or "",
                    "type": getattr(tc, "type", "function") or "function",
                    "function": {
                        "name": getattr(tc.function, "name", "") or "",
                        "arguments": getattr(tc.function, "arguments", "") or "",
                    },
                })

        return CompletionResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            model=getattr(response, "model", self.model),
            usd_cost=usd_cost,
            tool_calls=tool_calls,
        )

    def complete(self, system: str, user: str) -> CompletionResult:
        return self._invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    def complete_turns(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        return self._invoke(messages, tools=tools)


# ──────────────────────────────────────────
# FakeLLMSolver — unit tests, offline development
# ──────────────────────────────────────────


class FakeLLMSolver(LLMSolver):
    """Test double that returns a canned response for every call.

    ``complete`` can be:
    - a string — returned verbatim as the response text
    - a callable ``(system, user) -> str``
    - a list / iterator of strings — dequeued per call; raises if exhausted
    """

    def __init__(
        self,
        response: str | list[str] | Any,
        *,
        label: str = "fake",
        usd_cost: float | None = 0.0,
    ) -> None:
        self.model = "fake"
        self.temperature = 0.0
        self.max_tokens = 8192
        self.timeout_s = 0.0
        self.label = label
        self._response = response
        self._usd_cost = usd_cost
        self._calls: list[tuple[str, str]] = []
        self._turn_calls: list[list[dict[str, str]]] = []

    @property
    def calls(self) -> list[tuple[str, str]]:
        return list(self._calls)

    @property
    def turn_calls(self) -> list[list[dict[str, str]]]:
        """The ``messages`` list of every `complete_turns` call, in order."""
        return [list(m) for m in self._turn_calls]

    def _next_text(self, context_tokens: int) -> tuple[str, int]:
        """Pop the next canned response. Returns (text, prompt_tokens)."""
        if callable(self._response):
            text = self._response(context_tokens, None)  # signature kept loose
        elif isinstance(self._response, list):
            if not self._response:
                raise LLMSolverError("FakeLLMSolver response queue is empty")
            text = self._response.pop(0)
        else:
            text = str(self._response)
        return text, context_tokens

    def complete(self, system: str, user: str) -> CompletionResult:
        self._calls.append((system, user))
        text: str
        if callable(self._response):
            text = self._response(system, user)
        elif isinstance(self._response, list):
            if not self._response:
                raise LLMSolverError("FakeLLMSolver response queue is empty")
            text = self._response.pop(0)
        else:
            text = str(self._response)
        return CompletionResult(
            text=text,
            prompt_tokens=len(system.split()) + len(user.split()),
            completion_tokens=len(text.split()),
            latency_s=0.0,
            model=self.label,
            usd_cost=self._usd_cost,
        )

    def complete_turns(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,  # noqa: ARG002 -- accepted for API parity
    ) -> CompletionResult:
        self._turn_calls.append([dict(m) for m in messages])
        if callable(self._response):
            raw = self._response(messages, None)
        elif isinstance(self._response, list):
            if not self._response:
                raise LLMSolverError("FakeLLMSolver response queue is empty")
            raw = self._response.pop(0)
        else:
            raw = self._response

        # Accept 3 canned-response shapes:
        #   - str   -> plain text final content
        #   - dict  -> {"text": str, "tool_calls": [...]} for tool-call dispatch
        #   - CompletionResult -> pass through verbatim (advanced scenarios)
        if isinstance(raw, CompletionResult):
            return raw
        if isinstance(raw, dict):
            text = str(raw.get("text", "") or "")
            tool_calls = raw.get("tool_calls")
        else:
            text = str(raw)
            tool_calls = None

        prompt_tokens = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                prompt_tokens += len(content.split())
        return CompletionResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            latency_s=0.0,
            model=self.label,
            usd_cost=self._usd_cost,
            tool_calls=tool_calls,
        )


# ──────────────────────────────────────────
# EnvAdapter base + registry
# ──────────────────────────────────────────


class EnvAdapter(ABC):
    """Per-environment prompt construction and response parsing."""

    env_name: str
    system_prompt: str

    @abstractmethod
    def build_user_prompt(self, instance: Any) -> str:
        """Compact JSON-friendly description of the problem + measurements.

        Implementations MUST keep the prompt well under 2000 tokens by
        downsampling arrays to a small integer grid before encoding.
        """

    @abstractmethod
    def parse_response(self, text: str, instance: Any) -> Any:
        """Extract a JSON block from ``text`` and build the env's ``Prediction``.

        Raises ``LLMSolverError`` on invalid JSON or missing fields.
        """

    def build_followup_turn(
        self,
        history: list[dict[str, str]],
        last_prediction: Any,
        instance: Any,
    ) -> str:
        """Build the user-message content for a follow-up turn in a multi-turn env.

        Default: raises ``NotImplementedError``. Multi-turn adapters override
        to return a fresh user turn conditioned on the solver's last
        prediction (e.g. an encoded residual image, a validation error
        message, a partial-credit hint). Single-turn adapters leave this
        untouched and continue to use the ``solve()`` path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support multi-turn rollouts"
        )


_ADAPTERS: dict[str, EnvAdapter] = {}


def register_adapter(adapter: EnvAdapter) -> None:
    """Register an adapter against its ``env_name``. Overwrites prior entries."""
    _ADAPTERS[adapter.env_name] = adapter


def get_adapter(env_name: str) -> EnvAdapter:
    try:
        return _ADAPTERS[env_name]
    except KeyError as exc:
        raise LLMSolverError(f"No LLM adapter registered for env '{env_name}'") from exc


def registered_env_names() -> list[str]:
    return sorted(_ADAPTERS)
