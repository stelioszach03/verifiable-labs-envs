"""Environment handle returned by ``client.env(env_id)``.

Provides the two ergonomic flows the brief specifies:

* ``env.evaluate(seed, answer)`` — single-shot. Creates a session,
  submits, returns the score. Drops the session afterwards.
* ``env.start_session(seed)`` — multi-step. Returns a ``Session``
  object the caller can ``submit()`` to repeatedly (used by multi-turn
  envs, but works on single-turn envs too).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verifiable_labs.models import CreateSessionResponse, SubmitResponse
from verifiable_labs.session import AsyncSession, Session

if TYPE_CHECKING:
    from verifiable_labs.client import AsyncClient, Client


class Environment:
    """Sync env handle. Lightweight — caches no per-instance state."""

    def __init__(self, *, client: Client, env_id: str) -> None:
        self._client = client
        self._env_id = env_id

    @property
    def env_id(self) -> str:
        return self._env_id

    def start_session(
        self,
        seed: int = 0,
        *,
        env_kwargs: dict[str, Any] | None = None,
    ) -> Session:
        data = self._client._create_session(
            env_id=self._env_id, seed=seed, env_kwargs=env_kwargs or {},
        )
        return Session(self._client, response=CreateSessionResponse.model_validate(data))

    def evaluate(
        self,
        seed: int,
        answer: str | dict[str, Any],
        *,
        env_kwargs: dict[str, Any] | None = None,
    ) -> SubmitResponse:
        """One-shot: start session, submit answer, return score.

        ``answer`` accepts either a raw string (LLM output) or a
        structured dict (forwarded as the API's ``answer`` field; v0.1
        returns 422 for that path — keep using strings).
        """
        session = self.start_session(seed=seed, env_kwargs=env_kwargs)
        if isinstance(answer, str):
            return session.submit(answer_text=answer)
        return session.submit(answer=answer)


class AsyncEnvironment:
    """Async mirror of :class:`Environment`."""

    def __init__(self, *, client: AsyncClient, env_id: str) -> None:
        self._client = client
        self._env_id = env_id

    @property
    def env_id(self) -> str:
        return self._env_id

    async def start_session(
        self,
        seed: int = 0,
        *,
        env_kwargs: dict[str, Any] | None = None,
    ) -> AsyncSession:
        data = await self._client._create_session(
            env_id=self._env_id, seed=seed, env_kwargs=env_kwargs or {},
        )
        return AsyncSession(self._client, response=CreateSessionResponse.model_validate(data))

    async def evaluate(
        self,
        seed: int,
        answer: str | dict[str, Any],
        *,
        env_kwargs: dict[str, Any] | None = None,
    ) -> SubmitResponse:
        session = await self.start_session(seed=seed, env_kwargs=env_kwargs)
        if isinstance(answer, str):
            return await session.submit(answer_text=answer)
        return await session.submit(answer=answer)


__all__ = ["Environment", "AsyncEnvironment"]
