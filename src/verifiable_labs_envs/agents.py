"""Agent adapter interface for the ``verifiable`` CLI.

The CLI's ``run`` subcommand needs a uniform way to call "an agent" no
matter where it lives — a Python file the user wrote, a subprocess
script that talks JSON over stdin/stdout, or an OpenAI-compatible HTTP
client. This module defines that surface.

Three loaders, all returning the same :class:`Agent` protocol:

- :func:`load_python_agent` — imports a ``.py`` file by path; the file
  must define a top-level ``solve(observation: dict) -> dict``.
- :func:`load_subprocess_agent` — wraps a shell command. The CLI sends
  the observation as JSON on stdin; the subprocess returns a
  prediction JSON on stdout. Useful for non-Python agents.
- :class:`OpenAICompatibleAgent` — convenience class for calling any
  OpenAI-style chat-completions endpoint (OpenAI, OpenRouter, local
  vLLM, llama.cpp, Anthropic via gateway, etc.). Falls back to a
  ``FakeLLMSolver`` when the API key isn't set so CI can exercise the
  code path without spending money.

This is **separate** from the per-env LLM dispatch in
``solvers/llm_solver.py``. That module is the heavy path used by the
benchmark scripts; this is the lightweight CLI path.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class Agent(Protocol):
    """Minimal contract every agent must satisfy.

    The CLI calls ``solve(observation)`` for each episode. ``observation``
    is the env-specific dict produced by ``EnvAdapter.build_user_prompt``
    + the env's instance values; the return value must be parseable by
    the env's ``EnvAdapter.parse_response``.
    """

    name: str

    def solve(self, observation: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class _CallableAgent:
    """Wraps a plain ``solve`` function into the Agent protocol."""
    name: str
    _fn: Callable[[dict[str, Any]], dict[str, Any]]

    def solve(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._fn(observation)


# ── Python file loader ──────────────────────────────────────────


def load_python_agent(path: str | Path) -> Agent:
    """Import ``path`` and return its top-level ``solve`` as an Agent.

    The agent's ``name`` is the file's basename without ``.py``.
    Raises ``ValueError`` if the module has no ``solve`` callable.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"agent file not found: {p}")
    if p.suffix != ".py":
        raise ValueError(f"agent file must be a .py: {p}")

    spec = importlib.util.spec_from_file_location(f"_agent_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent module from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "solve", None)
    if not callable(fn):
        raise ValueError(
            f"agent file {p} does not define a top-level callable named 'solve'. "
            f"Expected: def solve(observation: dict) -> dict"
        )
    name = getattr(module, "AGENT_NAME", None) or p.stem
    return _CallableAgent(name=str(name), _fn=fn)


# ── Subprocess loader ───────────────────────────────────────────


@dataclass
class SubprocessAgent:
    """Calls an external command for each episode.

    Protocol:
      - parent writes ``json.dumps(observation)`` + newline to the child's stdin
      - child writes ``json.dumps(prediction)`` to stdout, exits 0

    Stderr is captured and surfaced if the child fails.
    """

    name: str
    command: list[str]
    timeout_s: float = 60.0

    def solve(self, observation: dict[str, Any]) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                self.command,
                input=json.dumps(observation),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"subprocess agent {self.name!r} timed out after {self.timeout_s}s"
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"subprocess agent {self.name!r} exited {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
        out = proc.stdout.strip()
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"subprocess agent {self.name!r} did not emit valid JSON on stdout: {e}. "
                f"First 200 bytes: {out[:200]!r}"
            ) from e


def load_subprocess_agent(
    command: str | list[str], *, name: str | None = None, timeout_s: float = 60.0,
) -> Agent:
    """Wrap a shell command into the Agent protocol."""
    cmd = command.split() if isinstance(command, str) else list(command)
    if not cmd:
        raise ValueError("subprocess agent command cannot be empty")
    derived = name or f"subprocess:{Path(cmd[0]).name}"
    return SubprocessAgent(name=derived, command=cmd, timeout_s=timeout_s)


# ── OpenAI-compatible HTTP loader ───────────────────────────────


@dataclass
class OpenAICompatibleAgent:
    """Calls an OpenAI-style chat-completions endpoint.

    Constructor reads from env vars by default:
      - ``OPENAI_API_KEY`` (or whatever ``api_key_env`` says)
      - ``OPENAI_BASE_URL`` (defaults to ``https://api.openai.com/v1``)

    If the API key is not set, falls back to ``FakeLLMSolver`` and
    returns a hard-coded "I don't know" response — this lets tests and
    CI exercise the wiring without spending money.

    The agent passes the env's ``observation`` as a single user message
    plus the env's ``system_prompt`` if present in the observation. The
    returned text is parsed as JSON; the parsed object is the prediction.
    """

    name: str
    model: str
    base_url: str
    api_key: str | None
    timeout_s: float = 60.0
    temperature: float = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        model: str = "openai/gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
    ) -> OpenAICompatibleAgent:
        api_key = os.environ.get(api_key_env)
        base_url = os.environ.get(base_url_env, "https://api.openai.com/v1")
        return cls(
            name=f"openai:{model}",
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    def solve(self, observation: dict[str, Any]) -> dict[str, Any]:
        system = observation.get("system_prompt", "You are a helpful assistant.")
        user = observation.get("prompt_text") or json.dumps(observation)
        if not self.api_key:
            return self._fake_response(system=system, user=user)
        # Lazy import so the CLI doesn't pull openai by default.
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed; run `pip install openai` to use "
                "the OpenAICompatibleAgent with a live API key"
            ) from e
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = resp.choices[0].message.content or ""
        # The agent returns a parsed dict; the env's adapter does the
        # final shape-checking. Best-effort: try to parse the LLM output
        # as JSON, otherwise pass the raw text under "answer_text".
        parsed = _try_parse_json(text)
        out: dict[str, Any] = parsed if isinstance(parsed, dict) else {"answer_text": text}
        out["_latency_ms"] = latency_ms
        return out

    @staticmethod
    def _fake_response(*, system: str, user: str) -> dict[str, Any]:
        """Deterministic stand-in for the API. Returns an obviously wrong
        prediction so the env's reward function classifies it as a parse
        failure or a near-zero reward — exactly what we want when the
        whole point is "the wiring works without a key."""
        return {
            "answer_text": "{}",
            "_fake": True,
            "_system_len": len(system),
            "_user_len": len(user),
        }


def _try_parse_json(text: str) -> Any:
    """Best-effort JSON parser. Strips markdown fences if the LLM
    wrapped its output. Returns ``None`` on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # strip ```<lang>\n ... \n```
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ── Convenience: pick a loader based on the spec ───────────────


def load_agent(spec: str) -> Agent:
    """Auto-detect: if ``spec`` ends in ``.py`` and exists → Python loader;
    if it starts with ``cmd:`` → subprocess loader; if it starts with
    ``openai:<model>`` → OpenAICompatibleAgent.
    """
    if spec.startswith("openai:"):
        return OpenAICompatibleAgent.from_env(model=spec[len("openai:") :])
    if spec.startswith("cmd:"):
        return load_subprocess_agent(spec[len("cmd:") :])
    p = Path(spec)
    if p.suffix == ".py":
        return load_python_agent(p)
    raise ValueError(
        f"could not detect agent type from spec {spec!r}. "
        f"Use a path ending in .py, prefix with 'cmd:' for subprocess, "
        f"or 'openai:<model>' for OpenAI-compatible HTTP."
    )


__all__ = [
    "Agent",
    "OpenAICompatibleAgent",
    "SubprocessAgent",
    "load_agent",
    "load_python_agent",
    "load_subprocess_agent",
]
