"""Verifiable Labs Python SDK — v0.1.0a1 (alpha).

Client library for the Hosted Evaluation API. Wraps the eight ``/v1/*``
REST endpoints in two parallel ergonomic surfaces:

* :class:`Client` — synchronous, ``with``-statement compatible.
* :class:`AsyncClient` — asynchronous, ``async with`` compatible.

Quickstart
----------

>>> from verifiable_labs import Client
>>> with Client() as client:                                    # doctest: +SKIP
...     env = client.env("stelioszach/sparse-fourier-recovery") # doctest: +SKIP
...     result = env.evaluate(seed=0, answer=my_model_output)   # doctest: +SKIP
...     print(result.reward)                                    # doctest: +SKIP

The default base URL is ``http://localhost:8000`` (run the API locally
via ``uvicorn verifiable_labs_api.app:app``). Override with
``Client(base_url="https://api.verifiable-labs.com")`` once the public
deploy is live.

The Hosted Evaluation API v0.1.0-alpha is **unauthenticated**; the
``api_key`` argument is accepted for forward-compatibility with v0.2
(Tier-2) but has no effect today.
"""
from __future__ import annotations

__version__ = "0.1.0a4"

from verifiable_labs.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_S,
    AsyncClient,
    Client,
)
from verifiable_labs.env import AsyncEnvironment, Environment
from verifiable_labs.exceptions import (
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
    VerifiableLabsError,
)
from verifiable_labs.models import (
    EnvironmentList,
    HealthStatus,
    LeaderboardResponse,
    LeaderboardRow,
    SessionState,
    SubmitResponse,
)
from verifiable_labs.session import AsyncSession, Session

# Convenience re-exports from the heavy env package, so users who
# `pip install verifiable-labs` get the same imports the docs show:
#
#     from verifiable_labs import load_environment, list_environments
#
# Wrapped in try/except for the rare slim install where the env package
# is absent. ``__all__`` is built dynamically so static type-checkers and
# IDEs see exactly what's actually exported.
_BASE_EXPORTS = [
    "__version__",
    # clients
    "Client",
    "AsyncClient",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
    # handles
    "Environment",
    "AsyncEnvironment",
    "Session",
    "AsyncSession",
    # models
    "EnvironmentList",
    "HealthStatus",
    "LeaderboardResponse",
    "LeaderboardRow",
    "SessionState",
    "SubmitResponse",
    # exceptions
    "VerifiableLabsError",
    "TransportError",
    "InvalidRequestError",
    "NotFoundError",
    "RateLimitError",
    "ServerError",
]

try:
    from verifiable_labs_envs import (  # noqa: F401
        list_environments,
        load_environment,
    )

    __all__ = [*_BASE_EXPORTS, "load_environment", "list_environments"]
except ImportError:
    # Slim install (no envs package) — clients still work, env helpers absent.
    __all__ = list(_BASE_EXPORTS)
