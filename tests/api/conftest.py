"""Shared fixtures for the API test suite."""
from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from verifiable_labs_api import create_app


@pytest.fixture()
def app():
    """Fresh app instance with a loose rate limit so tests don't trip it."""
    return create_app(rate_limit="1000/minute", session_ttl_seconds=300)


@pytest.fixture()
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def truth_text_for():
    """Builds a JSON answer string that the sparse-fourier adapter can parse,
    populated with the instance's ground-truth support + amplitudes.

    Used by the session-lifecycle test to verify the whole submit path
    end-to-end without needing an LLM.
    """
    def _build(instance_inputs: dict, support_idx: list[int],
               support_amp_x1000: list[int]) -> str:
        return json.dumps({
            "support_idx": list(support_idx),
            "support_amp_x1000": list(support_amp_x1000),
        })
    return _build
