"""Sync + async session abstractions returned by ``env.start_session()``."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verifiable_labs.models import (
    CreateSessionResponse,
    SessionState,
    SubmitResponse,
)

if TYPE_CHECKING:
    from verifiable_labs.client import AsyncClient, Client


class Session:
    """A live evaluation session against the API.

    Wraps the API's session id + observation. Use :meth:`submit` to send
    an LLM answer; :attr:`complete` flips when the env signals
    completion (single-turn envs complete on first valid submission;
    multi-turn envs flip only after the API's turn-dispatcher agrees,
    which is a v0.2 feature).
    """

    def __init__(self, client: Client, *, response: CreateSessionResponse) -> None:
        self._client = client
        self._created = response
        self._submissions: list[SubmitResponse] = []
        self._complete = False

    # ── Public ──────────────────────────────────────────────────
    @property
    def session_id(self) -> str:
        return self._created.session_id

    @property
    def env_id(self) -> str:
        return self._created.env_id

    @property
    def seed(self) -> int:
        return self._created.seed

    @property
    def observation(self) -> dict[str, Any]:
        return self._created.observation

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._created.metadata)

    @property
    def history(self) -> list[SubmitResponse]:
        return list(self._submissions)

    @property
    def complete(self) -> bool:
        return self._complete

    def submit(
        self,
        answer_text: str | None = None,
        *,
        answer: dict[str, Any] | None = None,
    ) -> SubmitResponse:
        """Submit a model answer and return the score response.

        Pass either ``answer_text`` (raw model output that the env's
        per-env adapter parses) or ``answer`` (a structured prediction
        dict — reserved for v0.2 of the API; v0.1 returns 422 if the
        ``answer`` key is set).
        """
        if answer_text is None and answer is None:
            raise ValueError("Provide answer_text= or answer=.")
        data = self._client._submit(self.session_id, answer_text=answer_text, answer=answer)
        result = SubmitResponse.model_validate(data)
        self._submissions.append(result)
        if result.complete:
            self._complete = True
        return result

    def refresh(self) -> SessionState:
        """Re-fetch full session state from the server."""
        data = self._client._get_session(self.session_id)
        state = SessionState.model_validate(data)
        # Sync the client-side mirror in case multi-turn dispatch landed.
        self._submissions = list(state.submissions)
        self._complete = state.complete
        return state


class AsyncSession:
    """Async mirror of :class:`Session`."""

    def __init__(self, client: AsyncClient, *, response: CreateSessionResponse) -> None:
        self._client = client
        self._created = response
        self._submissions: list[SubmitResponse] = []
        self._complete = False

    @property
    def session_id(self) -> str:
        return self._created.session_id

    @property
    def env_id(self) -> str:
        return self._created.env_id

    @property
    def seed(self) -> int:
        return self._created.seed

    @property
    def observation(self) -> dict[str, Any]:
        return self._created.observation

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._created.metadata)

    @property
    def history(self) -> list[SubmitResponse]:
        return list(self._submissions)

    @property
    def complete(self) -> bool:
        return self._complete

    async def submit(
        self,
        answer_text: str | None = None,
        *,
        answer: dict[str, Any] | None = None,
    ) -> SubmitResponse:
        if answer_text is None and answer is None:
            raise ValueError("Provide answer_text= or answer=.")
        data = await self._client._submit(
            self.session_id, answer_text=answer_text, answer=answer
        )
        result = SubmitResponse.model_validate(data)
        self._submissions.append(result)
        if result.complete:
            self._complete = True
        return result

    async def refresh(self) -> SessionState:
        data = await self._client._get_session(self.session_id)
        state = SessionState.model_validate(data)
        self._submissions = list(state.submissions)
        self._complete = state.complete
        return state


__all__ = ["Session", "AsyncSession"]
