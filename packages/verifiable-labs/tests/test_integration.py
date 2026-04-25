"""Integration test — spins up a real uvicorn instance from
``verifiable_labs_api.app:app`` in a background thread, then runs the
SDK against it. Skipped when the API package is not importable
(useful for CI matrix slots that only install the SDK).

Run with: ``pytest -m integration packages/verifiable-labs/tests/``
or just ``pytest`` (the marker doesn't deselect by default).
"""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

pytestmark = pytest.mark.integration

try:
    import uvicorn
    from verifiable_labs_api import create_app
    HAVE_API = True
except ImportError:
    HAVE_API = False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api():
    if not HAVE_API:
        pytest.skip("verifiable_labs_api not installed; skip integration test")

    port = _free_port()
    app = create_app(rate_limit="1000/minute", session_ttl_seconds=300)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port,
        log_level="warning",
        loop="asyncio",   # avoid uvloop interference with pytest event loop
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to be live (max 5 s).
    deadline = time.monotonic() + 5.0
    import httpx
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("uvicorn did not come up in 5 s")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def test_sdk_against_live_api_health(live_api):
    from verifiable_labs import Client
    with Client(base_url=live_api) as c:
        h = c.health()
        assert h.status == "ok"
        assert h.version.startswith("0.1.0-alpha")


def test_sdk_against_live_api_environments(live_api):
    from verifiable_labs import Client
    with Client(base_url=live_api) as c:
        envs = c.environments()
        assert envs.count >= 10
        ids = {e.id for e in envs.environments}
        assert "sparse-fourier-recovery" in ids


def test_sdk_against_live_api_full_session_round_trip(live_api):
    """End-to-end: SDK creates session, submits truth, gets high reward."""
    from verifiable_labs_envs import load_environment

    from verifiable_labs import Client

    # Pre-compute the truth payload outside the SDK so we exercise the
    # SDK's submit path with a real LLM-style JSON answer.
    env_local = load_environment("sparse-fourier-recovery", calibration_quantile=2.0)
    inst = env_local.generate_instance(seed=0)
    truth = json.dumps({
        "support_idx": [int(i) for i in inst.support_true],
        "support_amp_x1000": [
            int(round(float(v) * 1000)) for v in inst.x_true[inst.support_true]
        ],
    })

    with Client(base_url=live_api) as c:
        env = c.env("stelioszach/sparse-fourier-recovery")
        result = env.evaluate(
            seed=0,
            answer=truth,
            env_kwargs={"calibration_quantile": 2.0},
        )
        assert result.parse_ok
        assert result.complete  # single-turn env
        assert result.reward > 0.85, f"truth submission should score high: {result}"
        assert "nmse" in result.components


def test_sdk_against_live_api_leaderboard(live_api):
    from verifiable_labs import Client
    with Client(base_url=live_api) as c:
        lb = c.leaderboard("sparse-fourier-recovery")
        assert lb.env_id == "sparse-fourier-recovery"
        # Sprint-1 v2 + paper-final 1A contributed enough rows that
        # there should be at least one model.
        assert isinstance(lb.rows, list)
