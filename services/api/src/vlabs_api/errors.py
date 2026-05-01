"""RFC 7807 problem-details exception classes + handler.

Every API error is a subclass of :class:`APIError` with a fixed
``status_code``, machine-readable ``code`` and human-readable
``title``. The FastAPI handler converts them to the
``application/problem+json`` response shape::

    {
      "type":   "https://api.verifiable-labs.com/errors/invalid_alpha",
      "title":  "alpha must be in (0, 1)",
      "status": 400,
      "code":   "invalid_alpha",
      "detail": "got 1.0"
    }
"""
from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

_BASE_TYPE = "https://api.verifiable-labs.com/errors"


class APIError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    title: str = "Internal server error"

    def __init__(self, detail: str | None = None, **extra: Any) -> None:
        self.detail = detail
        self.extra = extra
        super().__init__(detail or self.title)


# ── 4xx — client errors ───────────────────────────────────────────


class InvalidAlpha(APIError):
    status_code = 400
    code = "invalid_alpha"
    title = "alpha must be in (0, 1)"


class TracesTooFew(APIError):
    status_code = 400
    code = "traces_too_few"
    title = "need at least 2 calibration traces"


class TracesTooMany(APIError):
    status_code = 400
    code = "traces_too_many"
    title = "exceeded maximum traces per request"


class UnknownNonconformity(APIError):
    status_code = 400
    code = "unknown_nonconformity"
    title = "unknown non-conformity score"


class MissingRequiredKeys(APIError):
    status_code = 400
    code = "missing_required_keys"
    title = "trace is missing required keys for the chosen non-conformity"


class InvalidUncertainty(APIError):
    status_code = 400
    code = "invalid_uncertainty"
    title = "uncertainty (sigma) must be non-negative"


class InvalidScore(APIError):
    status_code = 400
    code = "invalid_score"
    title = "non-conformity score is non-finite"


class InvalidAPIKey(APIError):
    status_code = 401
    code = "invalid_api_key"
    title = "missing or invalid X-Vlabs-Key header"


class QuotaExceeded(APIError):
    status_code = 402
    code = "quota_exceeded"
    title = "monthly trace quota exhausted for this tier"


class CalibrationNotFound(APIError):
    status_code = 404
    code = "calibration_not_found"
    title = "no calibration with this id is owned by this API key"


class RateLimited(APIError):
    status_code = 429
    code = "rate_limited"
    title = "per-tier rate limit exceeded"


# ── 4xx — billing / auth (Stage B) ────────────────────────────────


class InvalidClerkToken(APIError):
    status_code = 401
    code = "invalid_clerk_token"
    title = "missing or invalid Clerk session token"


class WebhookSignatureInvalid(APIError):
    status_code = 400
    code = "webhook_signature_invalid"
    title = "Stripe webhook signature did not verify"


class WebhookEventUnsupported(APIError):
    status_code = 400
    code = "webhook_event_unsupported"
    title = "Stripe event type is not handled by this endpoint"


class StripeNotConfigured(APIError):
    status_code = 503
    code = "stripe_not_configured"
    title = "Stripe integration is not configured (test-mode only)"


class ClerkNotConfigured(APIError):
    status_code = 503
    code = "clerk_not_configured"
    title = "Clerk integration is not configured"


class APIKeyNotFoundForUser(APIError):
    status_code = 404
    code = "api_key_not_found"
    title = "no API key with this id is owned by the authenticated user"


# ── 4xx — Stage C ─────────────────────────────────────────────────


class BillingNotActivated(APIError):
    """Returned when VLABS_BILLING_ENABLED=false (default until C-corp lands)."""

    status_code = 503
    code = "billing_not_activated"
    title = "Billing activation pending. Free tier fully functional."


class NotAdmin(APIError):
    status_code = 403
    code = "not_admin"
    title = "this Clerk user is not in the admin allowlist"


# ── Handler ───────────────────────────────────────────────────────


def to_problem_json(_: Request, exc: APIError) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"{_BASE_TYPE}/{exc.code}",
        "title": exc.title,
        "status": exc.status_code,
        "code": exc.code,
    }
    if exc.detail is not None:
        body["detail"] = exc.detail
    body.update(exc.extra)
    headers = {"content-type": "application/problem+json"}
    if isinstance(exc, RateLimited) and "retry_after" in exc.extra:
        headers["Retry-After"] = str(exc.extra["retry_after"])
    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


__all__ = [
    "APIError",
    "InvalidAlpha",
    "TracesTooFew",
    "TracesTooMany",
    "UnknownNonconformity",
    "MissingRequiredKeys",
    "InvalidUncertainty",
    "InvalidScore",
    "InvalidAPIKey",
    "QuotaExceeded",
    "CalibrationNotFound",
    "RateLimited",
    "InvalidClerkToken",
    "WebhookSignatureInvalid",
    "WebhookEventUnsupported",
    "StripeNotConfigured",
    "ClerkNotConfigured",
    "APIKeyNotFoundForUser",
    "BillingNotActivated",
    "NotAdmin",
    "to_problem_json",
]
