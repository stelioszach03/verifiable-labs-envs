"""Exception hierarchy for the Verifiable Labs SDK.

Every API error path raises a typed exception so callers can ``except``
on the specific failure mode rather than parsing string messages. The
root class is :class:`VerifiableLabsError`; HTTP-status-mapped subclasses
live below it.

Mapping (used by :func:`raise_for_status`):

  * 400 → :class:`InvalidRequestError`
  * 404 → :class:`NotFoundError`
  * 422 → :class:`InvalidRequestError`
  * 429 → :class:`RateLimitError`
  * 5xx → :class:`ServerError`
  * other 4xx → :class:`InvalidRequestError`
  * network / timeout → :class:`TransportError`
"""
from __future__ import annotations

from typing import Any

import httpx


class VerifiableLabsError(Exception):
    """Base class for all SDK-raised errors.

    Carries optional structured context (status code, response body) so
    integration tests and instrumentation can assert on it.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class TransportError(VerifiableLabsError):
    """Network / connection / timeout failure before an HTTP response."""


class InvalidRequestError(VerifiableLabsError):
    """The API rejected the request (400 / 422 / other 4xx).

    Typically: malformed payload, unknown env id, missing required field.
    """


class NotFoundError(VerifiableLabsError):
    """404 — env id, session id, or other resource not found."""


class RateLimitError(VerifiableLabsError):
    """429 — slowapi tripped the per-IP rate limit.

    The API replies with ``Retry-After`` semantics in the body; the
    message string preserves the server's own ``detail`` field so
    automated retries can read it.
    """


class ServerError(VerifiableLabsError):
    """5xx — the server crashed handling the request."""


def raise_for_status(response: httpx.Response) -> None:
    """Translate a non-2xx response into the matching SDK exception.

    Called by every transport method *before* ``response.json()`` so
    callers always see typed exceptions, never bare ``HTTPStatusError``.
    """
    if response.is_success:
        return
    code = response.status_code
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = response.text
    detail = (
        body.get("detail")
        if isinstance(body, dict) and "detail" in body
        else f"HTTP {code} from {response.request.url}"
    )
    if code == 404:
        raise NotFoundError(detail, status_code=code, response_body=body)
    if code == 429:
        raise RateLimitError(detail, status_code=code, response_body=body)
    if 500 <= code < 600:
        raise ServerError(detail, status_code=code, response_body=body)
    if 400 <= code < 500:
        raise InvalidRequestError(detail, status_code=code, response_body=body)
    raise VerifiableLabsError(detail, status_code=code, response_body=body)


__all__ = [
    "VerifiableLabsError",
    "TransportError",
    "InvalidRequestError",
    "NotFoundError",
    "RateLimitError",
    "ServerError",
    "raise_for_status",
]
