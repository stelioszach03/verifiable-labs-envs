"""Model-provider interface (SDK-safe, public).

Every agent/model backend implements :class:`ModelProvider`. The open-source
SDK ships only this interface plus the deterministic
:class:`~verifiable_labs_envs.providers.dummy_provider.DummyProvider`; real
backends (OpenAI/Anthropic/Google/OpenRouter/custom/self-hosted) live in the
private platform because they carry key-handling, SSRF, and billing concerns.

Security-first contract:

* :meth:`ModelProvider.dry_run` must never perform network I/O.
* :class:`ProviderConfig` never stores a plaintext secret. ``auth_mode``
  records *how* credentials are resolved; the actual material is resolved
  out-of-band by the private platform and is never logged or serialized here.
* CPU smoke tests and CI use :class:`DummyProvider` only.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any


class AuthMode(StrEnum):
    """How a provider's credentials are resolved.

    * ``VLABS_MANAGED`` — Verifiable Labs holds a wholesale key server-side.
    * ``CUSTOMER_BYOK`` — customer's own provider key (encrypted, project
      scoped, never logged/exported; resolved by the private platform).
    * ``SELF_HOSTED`` — customer-operated endpoint inside their VPC.
    * ``DUMMY`` — deterministic local stub; the only mode used in smoke/CI.
    """

    VLABS_MANAGED = "vlabs_managed"
    CUSTOMER_BYOK = "customer_byok"
    SELF_HOSTED = "self_hosted"
    DUMMY = "dummy"


@dataclass(frozen=True)
class RetryPolicy:
    """Retry/backoff budget for a provider call."""

    max_retries: int = 2
    backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")


@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for a provider — **never** holds a secret.

    The credential itself is resolved by the private platform via
    ``auth_mode`` + a server-side key reference; this object only records the
    routing/runtime knobs that are safe to serialize, log (after redaction),
    and ship in the open-source SDK.
    """

    provider_name: str
    model_name: str
    auth_mode: AuthMode = AuthMode.DUMMY
    endpoint_url: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    redaction_policy: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_name:
            raise ValueError("provider_name is required")
        if not self.model_name:
            raise ValueError("model_name is required")
        if not isinstance(self.auth_mode, AuthMode):
            raise TypeError("auth_mode must be an AuthMode")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be in [0, 2]")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

    def redacted(self) -> dict[str, Any]:
        """Log-safe view. There is no secret to strip (configs never hold
        one), but ``endpoint_url`` can carry a tenant hint, so it is masked."""
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "auth_mode": self.auth_mode.value,
            "endpoint_url": "***" if self.endpoint_url else None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class ModelRequest:
    """A single prompt/rollout request handed to a provider."""

    prompt: str
    system: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")


@dataclass(frozen=True)
class CostEstimate:
    """Pre-flight cost/latency estimate for a request (no call made)."""

    tokens_input: int
    tokens_output: int
    usd: float
    latency_ms_estimate: float
    is_dry_run: bool = True

    def __post_init__(self) -> None:
        for name in ("tokens_input", "tokens_output"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.usd < 0 or not math.isfinite(self.usd):
            raise ValueError("usd must be a finite, non-negative number")


@dataclass(frozen=True)
class ModelResponse:
    """A provider's response. ``is_dry_run`` marks a no-network stub."""

    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    provider_name: str
    model_name: str
    finish_reason: str = "stop"
    is_dry_run: bool = False
    cost: CostEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """Interface every model backend implements.

    Implementations must keep :meth:`dry_run` free of network I/O so the
    whole pipeline can be exercised on CPU in smoke/CI with zero spend and
    zero external dependency.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def validate_config(self) -> None:
        """Raise if the configuration is unusable (e.g. blocked endpoint).

        Must not perform network I/O — it validates shape/policy only.
        """

    @abstractmethod
    def estimate_cost(self, request: ModelRequest) -> CostEstimate:
        """Return a pre-flight cost/latency estimate without calling out."""

    @abstractmethod
    def run(self, request: ModelRequest) -> ModelResponse:
        """Execute the request for real. May perform network I/O in real
        providers; the dummy provider stays local."""

    @abstractmethod
    def dry_run(self, request: ModelRequest) -> ModelResponse:
        """Return a deterministic stub response. **Never** performs network
        I/O. Used by smoke/CI and by every pipeline run with
        ``dry_run=True``."""

    def with_overrides(self, **kwargs: Any) -> ModelProvider:
        """Return a copy of this provider with config fields overridden."""
        return type(self)(replace(self.config, **kwargs))
