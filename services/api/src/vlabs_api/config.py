"""Runtime configuration loaded from environment variables.

Defaults are tuned for local development (``pgserver`` Postgres,
non-secret pepper). Production overrides every value via Fly.io
secrets.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class TierLimits(BaseSettings):
    """Built-in tier quotas + rate limits.

    Stored on the settings object (rather than DB) for v0.0.1 — kept
    flat and immutable per deploy. Will move to DB in Stage B/C if we
    introduce dynamic plan management.
    """

    free_traces_per_month: int = 10_000
    free_rpm: int = 100
    pro_traces_per_month: int = 1_000_000
    pro_rpm: int = 1_000
    team_traces_per_month: int = 10_000_000
    team_rpm: int = 10_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required at runtime ───────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://vlabs:vlabs@localhost:5433/vlabs"
    )
    vlabs_api_key_hash_pepper: str = "dev-pepper-not-for-production-use-only"

    # ── Optional ──────────────────────────────────────────────────
    vlabs_log_level: str = "INFO"
    vlabs_environment: Literal["dev", "staging", "prod"] = "dev"

    # ── Stage B: Stripe (TEST MODE ONLY until C-corp registered) ──
    stripe_secret_key: str | None = None  # sk_test_... only in dev
    stripe_webhook_secret: str | None = None  # whsec_... from Stripe Dashboard
    stripe_price_id_pro: str | None = None
    stripe_price_id_team: str | None = None
    stripe_price_id_pro_overage: str | None = None
    stripe_price_id_team_overage: str | None = None
    stripe_billing_portal_return_url: str = "http://localhost:3000/dashboard/billing"
    stripe_checkout_success_url: str = "http://localhost:3000/dashboard/billing?status=success"
    stripe_checkout_cancel_url: str = "http://localhost:3000/dashboard/billing?status=cancel"

    # ── Stage B: Clerk (dashboard auth only) ──────────────────────
    clerk_secret_key: str | None = None
    clerk_publishable_key: str | None = None
    clerk_jwt_issuer: str | None = None  # e.g. https://something.clerk.accounts.dev
    clerk_jwks_url: str | None = None  # auto-derived from issuer if not set

    # ── Tier limits (composed in) ─────────────────────────────────
    @property
    def tiers(self) -> TierLimits:
        return TierLimits()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — reads env once per process."""
    return Settings()


__all__ = ["Settings", "TierLimits", "get_settings"]
