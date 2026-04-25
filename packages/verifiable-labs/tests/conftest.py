"""Shared fixtures for the SDK test suite."""
from __future__ import annotations

import pytest
import respx

from verifiable_labs import AsyncClient, Client

TEST_BASE_URL = "https://api.test.example"


@pytest.fixture()
def base_url() -> str:
    return TEST_BASE_URL


@pytest.fixture()
def mock_api():
    """respx router that intercepts calls to ``TEST_BASE_URL``."""
    with respx.mock(base_url=TEST_BASE_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture()
def client(base_url):
    c = Client(base_url=base_url, timeout=2.0)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture()
def async_client(base_url):
    return AsyncClient(base_url=base_url, timeout=2.0)


# ── Canned API responses (match server-side schemas exactly) ──────


@pytest.fixture()
def health_payload():
    return {
        "status": "ok",
        "version": "0.1.0-alpha",
        "uptime_s": 12.5,
        "sessions_active": 3,
    }


@pytest.fixture()
def envs_payload():
    return {
        "environments": [
            {
                "id": "sparse-fourier-recovery",
                "qualified_id": "stelioszach/sparse-fourier-recovery",
                "domain": "compressed-sensing",
                "multi_turn": False,
                "tool_use": False,
                "description": "1D sparse Fourier recovery with OMP baseline.",
            },
            {
                "id": "sparse-fourier-recovery-multiturn",
                "qualified_id": "stelioszach/sparse-fourier-recovery-multiturn",
                "domain": "compressed-sensing",
                "multi_turn": True,
                "tool_use": False,
                "description": "3-turn dialogue variant of sparse Fourier recovery.",
            },
        ],
        "count": 2,
    }


@pytest.fixture()
def session_create_payload():
    return {
        "session_id": "s-123",
        "env_id": "sparse-fourier-recovery",
        "seed": 42,
        "observation": {
            "prompt_text": "Recover the sparse signal…",
            "system_prompt": "You are an expert…",
            "inputs": {"n": 256, "k": 10, "mask": [1, 5, 9]},
        },
        "metadata": {
            "env_id": "sparse-fourier-recovery",
            "qualified_id": "stelioszach/sparse-fourier-recovery",
            "adapter_attached": True,
            "ttl_seconds": 3600,
        },
        "created_at": "2026-04-25T15:00:00+00:00",
        "expires_at": "2026-04-25T16:00:00+00:00",
    }


@pytest.fixture()
def submit_payload():
    return {
        "session_id": "s-123",
        "reward": 0.842,
        "components": {"nmse": 0.91, "support": 0.85, "conformal": 0.76},
        "coverage": 0.91,
        "parse_ok": True,
        "complete": True,
        "meta": {"weights": {"nmse": 0.4, "support": 0.3, "conformal": 0.3}},
    }


@pytest.fixture()
def session_state_payload(submit_payload):
    return {
        "session_id": "s-123",
        "env_id": "sparse-fourier-recovery",
        "seed": 42,
        "created_at": "2026-04-25T15:00:00+00:00",
        "expires_at": "2026-04-25T16:00:00+00:00",
        "submissions": [submit_payload],
        "complete": True,
    }


@pytest.fixture()
def leaderboard_payload():
    return {
        "env_id": "sparse-fourier-recovery",
        "rows": [
            {
                "model": "anthropic/claude-haiku-4.5",
                "n": 12,
                "mean_reward": 0.554,
                "std_reward": 0.082,
                "parse_fail_rate": 0.05,
            },
            {
                "model": "openai/gpt-5.4",
                "n": 9,
                "mean_reward": 0.519,
                "std_reward": 0.073,
                "parse_fail_rate": 0.02,
            },
            {
                "model": "openai/gpt-5.4-mini",
                "n": 12,
                "mean_reward": 0.483,
                "std_reward": 0.061,
                "parse_fail_rate": 0.10,
            },
        ],
        "sources": [
            "complete_matrix_single_turn.csv",
            "llm_benchmark_v2.csv",
        ],
    }
