"""Pytest fixtures — pgserver Postgres, ASGI client, throwaway API keys.

Setup contract (in order):
  1. import-time: spin up pgserver, set DATABASE_URL + pepper env vars.
  2. import vlabs_api modules (now safe — env is in place).
  3. neutralise ``app.router.lifespan_context`` so the test client doesn't
     init/dispose the engine on every request — fixtures own that.
  4. session-scoped fixture creates the schema once.
  5. autouse function-scoped fixture truncates between tests.
"""
from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import AsyncIterator

import pgserver
import pytest
from httpx import ASGITransport, AsyncClient

# ── Step 1: pgserver + env ────────────────────────────────────────────

_PG_DATA = "/tmp/vlabs_api_pg_test"
_pg_server = pgserver.get_server(_PG_DATA)
_uri = _pg_server.get_uri()
if _uri.startswith("postgresql://"):
    _uri = _uri.replace("postgresql://", "postgresql+asyncpg://", 1)

os.environ["DATABASE_URL"] = _uri
os.environ.setdefault("VLABS_API_KEY_HASH_PEPPER", "test-pepper-not-for-prod")
os.environ.setdefault("VLABS_ENVIRONMENT", "dev")
os.environ.setdefault("VLABS_LOG_LEVEL", "WARNING")

# ── Step 2: now safe to import vlabs_api ─────────────────────────────
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from vlabs_api.config import get_settings  # noqa: E402

get_settings.cache_clear()  # any earlier import may have cached defaults

import vlabs_api.db as db  # noqa: E402
from vlabs_api.db import Base  # noqa: E402
from vlabs_api.main import app as _app  # noqa: E402

# ── Step 3: neutralise app lifespan ──────────────────────────────────

@contextlib.asynccontextmanager
async def _noop_lifespan(_a):
    yield


_app.router.lifespan_context = _noop_lifespan


# ── Step 4: schema once per session ─────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
async def setup_db() -> AsyncIterator[None]:
    engine = db.init_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await db.dispose_engine()


# ── Step 5: truncate between tests ───────────────────────────────────


@pytest.fixture(autouse=True)
async def truncate_data(setup_db) -> AsyncIterator[None]:
    yield
    if db._SessionFactory is None:
        return
    async with db._SessionFactory() as s:  # type: ignore[misc]
        await s.execute(
            text(
                "TRUNCATE TABLE subscriptions, usage_counters, evaluations, "
                "calibration_runs, api_keys, users RESTART IDENTITY CASCADE"
            )
        )
        await s.commit()


# ── Test helpers ─────────────────────────────────────────────────────


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    async with db._SessionFactory() as s:  # type: ignore[misc]
        yield s


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def api_key(session: AsyncSession) -> tuple[str, dict]:
    """Create a fresh user + free-tier API key.

    Returns ``(plaintext_key, info_dict)`` where ``info_dict`` has
    ``user_id`` and ``api_key_id``.
    """
    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    from vlabs_api.db import APIKey, User

    user = User(email=f"test-{uuid.uuid4().hex[:8]}@example.com", name="Test User")
    session.add(user)
    await session.flush()

    plaintext = generate_plaintext_key()
    api_row = APIKey(
        user_id=user.id,
        key_hash=hash_plaintext_key(plaintext),
        key_prefix=key_prefix(plaintext),
        name="test-key",
    )
    session.add(api_row)
    await session.commit()
    return plaintext, {
        "user_id": user.id,
        "api_key_id": api_row.id,
    }


def _gauss_traces(n: int, seed: int = 0, sigma: float = 0.5) -> list[dict]:
    """Synthetic Gaussian (predicted, reference, sigma) triples for tests."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = float(rng.standard_normal())
        out.append(
            {
                "predicted_reward": x,
                "reference_reward": x + sigma * float(rng.standard_normal()),
                "uncertainty": sigma,
            }
        )
    return out


@pytest.fixture
def gauss_traces():
    return _gauss_traces


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear the in-memory rate limiter buckets between tests."""
    from vlabs_api.ratelimit import reset_for_tests

    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
async def clerk_user(session: AsyncSession) -> tuple[str, User]:  # noqa: F821
    """Create a User linked to a fake Clerk id; return ``(fake_jwt, User)``.

    The JWT is just a sentinel string — tests monkeypatch
    :func:`vlabs_api.clerk_auth._verify_jwt` to return claims without
    actually verifying it.
    """
    from vlabs_api.db import User

    user = User(
        email=f"clerk-{uuid.uuid4().hex[:8]}@example.com",
        name="Clerk User",
        clerk_user_id=f"user_{uuid.uuid4().hex[:16]}",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return f"fake-jwt-for-{user.clerk_user_id}", user


@pytest.fixture
def stub_clerk_verify(monkeypatch):
    """Patch ``_verify_jwt`` to return claims based on the bearer token.

    Tests that need different claims call ``stub_clerk_verify(claims)``.
    """
    from vlabs_api import clerk_auth

    def _factory(claims: dict | None = None):
        def _stub(token: str):
            if claims is not None:
                return claims
            # Default: pull clerk_user_id from the fake-jwt-for-... token.
            if token.startswith("fake-jwt-for-"):
                return {
                    "sub": token.removeprefix("fake-jwt-for-"),
                    "exp": 9999999999,
                    "iat": 0,
                    "email": "stub@example.com",
                }
            return {"sub": "user_stub", "exp": 9999999999, "iat": 0}

        monkeypatch.setattr(clerk_auth, "_verify_jwt", _stub)

    return _factory


@pytest.fixture
def stub_stripe(monkeypatch):
    """Replace Stripe SDK calls used by billing.py with deterministic stubs."""
    import stripe

    state = {"customers": {}, "checkouts": [], "portals": [], "next_id": 1}

    def _next(prefix: str) -> str:
        state["next_id"] += 1
        return f"{prefix}_test_{state['next_id']:06d}"

    class _Customer:
        @staticmethod
        def create(**kwargs):
            cid = _next("cus")
            state["customers"][cid] = kwargs
            return {"id": cid, **kwargs}

    class _CheckoutSession:
        @staticmethod
        def create(**kwargs):
            sid = _next("cs")
            state["checkouts"].append({"id": sid, **kwargs})
            return {"id": sid, "url": f"https://stripe.test/checkout/{sid}"}

    class _Checkout:
        Session = _CheckoutSession

    class _PortalSession:
        @staticmethod
        def create(**kwargs):
            pid = _next("ps")
            state["portals"].append({"id": pid, **kwargs})
            return {"id": pid, "url": f"https://stripe.test/portal/{pid}"}

    class _BillingPortal:
        Session = _PortalSession

    monkeypatch.setattr(stripe, "Customer", _Customer)
    monkeypatch.setattr(stripe, "checkout", _Checkout)
    monkeypatch.setattr(stripe, "billing_portal", _BillingPortal)

    # Also pre-populate config with sane test values.
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_dummy")
    monkeypatch.setenv("STRIPE_PRICE_ID_PRO", "price_test_pro")
    monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_test_team")
    from vlabs_api.config import get_settings

    get_settings.cache_clear()
    return state


@pytest.fixture
async def calibrated(client: AsyncClient, api_key) -> dict:
    """Calibrate on 50 Gaussian traces and return the response body."""
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "scaled_residual",
        "traces": _gauss_traces(50, seed=1),
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    return r.json()
