"""``Client`` and ``AsyncClient`` — the SDK entry points.

Both clients share a common URL scheme + payload shape; the difference
is one calls ``httpx.Client`` and the other calls ``httpx.AsyncClient``.
The duplication is deliberate — async / sync internals diverge enough
that a transport-abstraction adapter would obscure stack traces and
lose static-typing precision. Keep the two parallel; mirror changes.

API key handling: the Hosted Evaluation API v0.1.0-alpha is
**unauthenticated**. ``api_key`` is accepted for forward-compatibility
with v0.2 (Tier 2); when set, the SDK forwards it as the
``X-VL-API-Key`` header. No effect today.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from verifiable_labs.exceptions import TransportError, raise_for_status
from verifiable_labs.models import (
    EnvironmentList,
    HealthStatus,
    LeaderboardResponse,
)

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_S = 30.0


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "verifiable-labs-sdk/0.1.0a1",
    }
    if api_key:
        headers["X-VL-API-Key"] = api_key
    return headers


def _v1(path: str) -> str:
    return f"/v1{path}" if path.startswith("/") else f"/v1/{path}"


# ────────────────────────────────────────────────────────────────────
# Sync client
# ────────────────────────────────────────────────────────────────────


class Client:
    """Synchronous SDK client.

    Examples
    --------
    >>> from verifiable_labs import Client
    >>> client = Client()                                   # doctest: +SKIP
    >>> envs = client.environments()                        # doctest: +SKIP
    >>> env = client.env("stelioszach/sparse-fourier-recovery")  # doctest: +SKIP
    >>> result = env.evaluate(seed=42, answer="…model output…")  # doctest: +SKIP
    >>> print(result.reward)                                # doctest: +SKIP
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._owns_http_client = http_client is None
        self._http: httpx.Client = http_client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=_build_headers(api_key),
        )

    # ── Lifecycle ──────────────────────────────────────────────────
    def close(self) -> None:
        if self._owns_http_client:
            self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Low-level transport ────────────────────────────────────────
    def _request(
        self, method: str, path: str, *, json: Any = None, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = self._http.request(method, _v1(path), json=json, params=params)
        except httpx.TransportError as exc:
            raise TransportError(str(exc)) from exc
        raise_for_status(response)
        return response.json() if response.content else {}

    # ── High-level methods ─────────────────────────────────────────
    def health(self) -> HealthStatus:
        return HealthStatus.model_validate(self._request("GET", "/health"))

    def environments(self) -> EnvironmentList:
        return EnvironmentList.model_validate(self._request("GET", "/environments"))

    def env(self, env_id: str):
        """Return a :class:`~verifiable_labs.env.Environment` handle for
        ``env_id`` (qualified ``owner/id`` form is OK)."""
        from verifiable_labs.env import Environment as EnvHandle  # local import avoids cycle
        return EnvHandle(client=self, env_id=env_id)

    def leaderboard(self, env_id: str) -> LeaderboardResponse:
        params = {"env_id": env_id}
        data = self._request("GET", "/leaderboard", params=params)
        return LeaderboardResponse.model_validate(data)

    # internal — used by Environment / Session
    def _create_session(self, env_id: str, seed: int, env_kwargs: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", "/sessions",
            json={"env_id": env_id, "seed": seed, "env_kwargs": env_kwargs},
        )

    def _submit(self, session_id: str, *, answer_text: str | None,
                answer: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if answer_text is not None:
            body["answer_text"] = answer_text
        if answer is not None:
            body["answer"] = answer
        return self._request("POST", f"/sessions/{session_id}/submit", json=body)

    def _get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{session_id}")


# ────────────────────────────────────────────────────────────────────
# Async client
# ────────────────────────────────────────────────────────────────────


class AsyncClient:
    """Asynchronous SDK client. Mirrors :class:`Client` one-to-one.

    Examples
    --------
    >>> from verifiable_labs import AsyncClient
    >>> async with AsyncClient() as client:                # doctest: +SKIP
    ...     env = client.env("sparse-fourier-recovery")    # doctest: +SKIP
    ...     result = await env.evaluate(seed=0, answer="…")  # doctest: +SKIP
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._owns_http_client = http_client is None
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=_build_headers(api_key),
        )

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _request(
        self, method: str, path: str, *, json: Any = None, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = await self._http.request(method, _v1(path), json=json, params=params)
        except httpx.TransportError as exc:
            raise TransportError(str(exc)) from exc
        raise_for_status(response)
        return response.json() if response.content else {}

    async def health(self) -> HealthStatus:
        return HealthStatus.model_validate(await self._request("GET", "/health"))

    async def environments(self) -> EnvironmentList:
        return EnvironmentList.model_validate(await self._request("GET", "/environments"))

    def env(self, env_id: str):
        from verifiable_labs.env import AsyncEnvironment
        return AsyncEnvironment(client=self, env_id=env_id)

    async def leaderboard(self, env_id: str) -> LeaderboardResponse:
        params = {"env_id": env_id}
        data = await self._request("GET", "/leaderboard", params=params)
        return LeaderboardResponse.model_validate(data)

    async def _create_session(self, env_id: str, seed: int,
                              env_kwargs: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST", "/sessions",
            json={"env_id": env_id, "seed": seed, "env_kwargs": env_kwargs},
        )

    async def _submit(self, session_id: str, *, answer_text: str | None,
                      answer: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if answer_text is not None:
            body["answer_text"] = answer_text
        if answer is not None:
            body["answer"] = answer
        return await self._request("POST", f"/sessions/{session_id}/submit", json=body)

    async def _get_session(self, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/sessions/{session_id}")


__all__ = ["Client", "AsyncClient", "DEFAULT_BASE_URL", "DEFAULT_TIMEOUT_S"]
