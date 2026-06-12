"""Model-provider abstraction (SDK-safe, public).

This package defines the *interface* every agent/model backend implements,
plus a deterministic :class:`DummyProvider` used by CPU-only smoke tests.
Real provider implementations (OpenAI, Anthropic, Google, OpenRouter, custom
HTTP, self-hosted) live in the **private** platform — never in the
open-source SDK — because they carry key-handling, SSRF, and billing
concerns.

Design rules (security-first):

* ``dry_run`` must never perform network I/O.
* Provider configs never carry plaintext secrets; auth material is resolved
  out-of-band by the private platform and never logged.
* Smoke/CI uses :class:`DummyProvider` only — no real provider call.
"""

from __future__ import annotations

from .base import (
    AuthMode,
    CostEstimate,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderConfig,
    RetryPolicy,
)
from .dummy_provider import DummyProvider

__all__ = [
    "AuthMode",
    "CostEstimate",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ProviderConfig",
    "RetryPolicy",
    "DummyProvider",
]
