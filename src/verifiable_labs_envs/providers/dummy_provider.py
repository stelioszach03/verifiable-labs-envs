"""Deterministic, network-free model provider for smoke tests and CI.

:class:`DummyProvider` produces reproducible "responses" from a seeded hash of
the request — no API key, no network, no spend. It is the only provider used
by ``scripts/smoke_test_pipeline.py`` and by any pipeline run executed in
``dry_run`` mode. Real providers live in the private platform.
"""

from __future__ import annotations

import hashlib

from .base import (
    AuthMode,
    CostEstimate,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderConfig,
)

# Illustrative per-token prices (USD) for the dummy estimator. These are NOT
# real provider prices; the private billing layer owns true pricing tables.
_DUMMY_USD_PER_1K_INPUT = 0.0005
_DUMMY_USD_PER_1K_OUTPUT = 0.0015


def _approx_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token)."""
    return max(1, (len(text) + 3) // 4)


class DummyProvider(ModelProvider):
    """A provider that never leaves the process.

    The response text is a deterministic function of the prompt, so smoke
    tests are reproducible and free of flakiness. ``run`` and ``dry_run``
    return identical content (both are local); ``run`` simply does not set
    ``is_dry_run``.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(
            config
            or ProviderConfig(
                provider_name="dummy",
                model_name="dummy-1",
                auth_mode=AuthMode.DUMMY,
            )
        )

    def validate_config(self) -> None:
        # Dummy provider is always valid; it has nothing to reach out to.
        if self.config.auth_mode is not AuthMode.DUMMY:
            # Allow non-dummy auth_mode but never resolve a real key here.
            pass

    def _synth(self, request: ModelRequest) -> str:
        h = hashlib.sha256(
            (request.system or "").encode() + b"\x00" + request.prompt.encode()
        ).hexdigest()
        # A compact, deterministic "answer" the verifiers can score.
        return f"dummy-response::{h[:16]}"

    def estimate_cost(self, request: ModelRequest) -> CostEstimate:
        ti = _approx_tokens((request.system or "") + request.prompt)
        to = min(self.config.max_tokens, _approx_tokens(self._synth(request)))
        usd = ti / 1000.0 * _DUMMY_USD_PER_1K_INPUT + to / 1000.0 * _DUMMY_USD_PER_1K_OUTPUT
        return CostEstimate(
            tokens_input=ti,
            tokens_output=to,
            usd=round(usd, 6),
            latency_ms_estimate=1.0,
            is_dry_run=True,
        )

    def _respond(self, request: ModelRequest, *, dry: bool) -> ModelResponse:
        text = self._synth(request)
        est = self.estimate_cost(request)
        return ModelResponse(
            text=text,
            tokens_input=est.tokens_input,
            tokens_output=est.tokens_output,
            latency_ms=1.0,
            provider_name=self.config.provider_name,
            model_name=self.config.model_name,
            finish_reason="stop",
            is_dry_run=dry,
            cost=est,
        )

    def run(self, request: ModelRequest) -> ModelResponse:
        # No network — identical to dry_run except the flag.
        return self._respond(request, dry=False)

    def dry_run(self, request: ModelRequest) -> ModelResponse:
        return self._respond(request, dry=True)
