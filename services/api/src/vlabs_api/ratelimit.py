"""Per-API-key rate limiting via slowapi.

Stage A ships a fixed ``100/minute`` cap (the free-tier limit). Stage B
will add tier-aware limits — pro 1000/min, team 10000/min — once Stripe
subscriptions populate ``request.state.auth.tier``. The wiring point is
the per-route ``@limiter.limit(...)`` decorator.

slowapi's limit-provider callable contract requires either a zero-arg
function or one with a literal ``key`` parameter (the output of
``key_func``). Tier-from-request inside the limit provider is awkward
under that contract; rather than fight it in Stage A we ship the static
limit and revisit when the tier system is real.
"""
from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from vlabs_api.errors import RateLimited

DEFAULT_LIMIT = "100/minute"


def _request_key(request: Request) -> str:
    auth = getattr(request.state, "auth", None)
    if auth is not None:
        return f"key:{auth.api_key_id}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_request_key, default_limits=[])


def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from vlabs_api.errors import to_problem_json

    as_problem = RateLimited(
        detail=str(exc.detail) if exc.detail else "rate limit exceeded",
        retry_after=int(getattr(exc, "retry_after", 0)) or 60,
    )
    return to_problem_json(request, as_problem)


__all__ = ["limiter", "rate_limit_handler", "DEFAULT_LIMIT"]
