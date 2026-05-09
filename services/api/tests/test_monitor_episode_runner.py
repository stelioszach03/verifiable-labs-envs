"""Tests for the Phase 28.C custom monitor episode runner."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from vlabs_api.monitor_episode_runner import (
    MonitorEpisodeResult,
    compute_run_summary,
    run_monitor_episode,
    run_monitor_episodes,
)


def _fake_chat_completion(content: str) -> dict[str, Any]:
    """Mimic an OpenAI-compatible /v1/chat/completions response body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 32,
            "completion_tokens": 16,
            "total_tokens": 48,
        },
    }


class _FakeTransport(httpx.AsyncBaseTransport):
    """httpx transport that returns canned chat-completion responses."""

    def __init__(self, content_for: dict[str, str] | None = None,
                 default_content: str = '{"answer": "0"}') -> None:
        self.content_for = content_for or {}
        self.default_content = default_content
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8") or "{}")
        messages = body.get("messages") or []
        user_msg = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        self.calls.append({"url": str(request.url), "user": user_msg})
        # Pick canned content if any key matches a substring of user prompt.
        content = self.default_content
        for needle, response in self.content_for.items():
            if needle in user_msg:
                content = response
                break
        return httpx.Response(
            status_code=200,
            json=_fake_chat_completion(content),
        )


def _fake_http(content_for=None, default='{"answer": "0"}') -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_FakeTransport(content_for, default))


# ── compute_run_summary ───────────────────────────────────────────


def test_compute_run_summary_empty() -> None:
    summary = compute_run_summary([])
    assert summary["per_env"] == {}
    assert summary["overall_mean_reward"] is None
    assert summary["n_total"] == 0


def test_compute_run_summary_single_env() -> None:
    rs = [
        MonitorEpisodeResult(
            env_id="math-algebra", seed=0, reward=0.8, components={},
            coverage=True, cost_usd_estimate=0.01, success=True,
        ),
        MonitorEpisodeResult(
            env_id="math-algebra", seed=1, reward=0.6, components={},
            coverage=False, cost_usd_estimate=0.01, success=True,
        ),
    ]
    summary = compute_run_summary(rs)
    assert summary["n_total"] == 2
    assert summary["n_success"] == 2
    assert summary["overall_mean_reward"] == pytest.approx(0.7)
    assert summary["overall_coverage"] == pytest.approx(0.5)
    env_stats = summary["per_env"]["math-algebra"]
    assert env_stats["n"] == 2
    assert env_stats["mean_reward"] == pytest.approx(0.7)
    assert env_stats["coverage"] == pytest.approx(0.5)


def test_compute_run_summary_multi_env_aggregation() -> None:
    rs = [
        MonitorEpisodeResult("math-algebra", 0, 1.0, {}, True, 0.0, True),
        MonitorEpisodeResult("math-algebra", 1, 0.0, {}, False, 0.0, True),
        MonitorEpisodeResult("code-humaneval", 0, 0.5, {}, True, 0.0, True),
    ]
    summary = compute_run_summary(rs)
    assert set(summary["per_env"]) == {"math-algebra", "code-humaneval"}
    assert summary["per_env"]["math-algebra"]["n"] == 2
    assert summary["per_env"]["code-humaneval"]["n"] == 1
    assert summary["overall_mean_reward"] == pytest.approx(0.5)


def test_compute_run_summary_counts_failures() -> None:
    rs = [
        MonitorEpisodeResult("math-algebra", 0, 0.0, {}, False, 0.0, False, "x"),
        MonitorEpisodeResult("math-algebra", 1, 0.5, {}, True, 0.0, True),
    ]
    summary = compute_run_summary(rs)
    assert summary["n_success"] == 1
    assert summary["n_total"] == 2


# ── run_monitor_episode ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_monitor_episode_happy_path() -> None:
    async with _fake_http(
        default=json.dumps({"answer": "0"})
    ) as client:
        result = await run_monitor_episode(
            env_id="math-algebra",
            seed=0,
            endpoint_url="https://fake-llm.test/v1",
            api_key="sk-fake",
            model="gpt-4o-mini",
            http_client=client,
        )
    assert result.env_id == "math-algebra"
    assert result.success is True
    assert isinstance(result.reward, float)
    assert 0.0 <= result.reward <= 1.0


@pytest.mark.asyncio
async def test_run_monitor_episode_records_failure_on_500() -> None:
    class _ErrTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(status_code=500, text="boom")

    async with httpx.AsyncClient(transport=_ErrTransport()) as client:
        result = await run_monitor_episode(
            env_id="math-algebra",
            seed=0,
            endpoint_url="https://broken.test/v1",
            api_key="sk-fake",
            model="gpt-4o-mini",
            http_client=client,
        )
    assert result.success is False
    assert result.error
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_run_monitor_episode_unknown_env_marks_failure() -> None:
    async with _fake_http() as client:
        result = await run_monitor_episode(
            env_id="does-not-exist",
            seed=0,
            endpoint_url="https://fake.test/v1",
            api_key="sk-fake",
            model="gpt-4o-mini",
            http_client=client,
        )
    assert result.success is False
    assert result.error and "instance" in result.error


@pytest.mark.asyncio
async def test_run_monitor_episode_handles_unparseable_completion() -> None:
    """Completion that doesn't match the env's adapter envelope must
    NEVER raise; the runner records a result row with reward=0 (and
    either ``success=False`` with a parse error or ``success=True``
    with reward=0, depending on the env adapter's parser strictness)."""
    async with _fake_http(default="this is unparseable prose") as client:
        result = await run_monitor_episode(
            env_id="math-algebra",
            seed=0,
            endpoint_url="https://fake.test/v1",
            api_key="sk-fake",
            model="gpt-4o-mini",
            http_client=client,
        )
    assert result.reward == 0.0
    # Either path is acceptable as long as the runner didn't crash.
    assert result.error is None or "score" in result.error or "parse" in result.error


# ── run_monitor_episodes (batch) ───────────────────────────────────


@pytest.mark.asyncio
async def test_run_monitor_episodes_batch_runs_each_seed() -> None:
    async with _fake_http(default='{"answer": "0"}') as client:
        results = await run_monitor_episodes(
            env_subset=["math-algebra"],
            episodes_per_env=3,
            endpoint_url="https://fake.test/v1",
            api_key="sk-fake",
            model="gpt-4o-mini",
            http_client=client,
            seed_start=10,
        )
    assert len(results) == 3
    assert {r.seed for r in results} == {10, 11, 12}


@pytest.mark.asyncio
async def test_run_monitor_episodes_multi_env() -> None:
    async with _fake_http(default='{"answer": "0"}') as client:
        results = await run_monitor_episodes(
            env_subset=["math-algebra", "code-humaneval"],
            episodes_per_env=2,
            endpoint_url="https://fake.test/v1",
            api_key="sk-fake",
            model="gpt-4o-mini",
            http_client=client,
            seed_start=0,
        )
    assert len(results) == 4  # 2 envs × 2 episodes
    by_env = {r.env_id for r in results}
    assert by_env == {"math-algebra", "code-humaneval"}


# ── PDF rendering ──────────────────────────────────────────────────


def test_render_monitor_pdf_emits_valid_pdf_header() -> None:
    from vlabs_api.monitor_pdf import render_monitor_pdf

    pdf = render_monitor_pdf(
        monitor_name="qwen-prod",
        monitor_id="mon_abc",
        run_id="mr_def",
        scheduled_at="2026-05-09T06:00:00Z",
        finished_at="2026-05-09T06:01:00Z",
        verdict="ok",
        summary={
            "per_env": {
                "math-algebra": {
                    "n": 5,
                    "mean_reward": 0.62,
                    "std_reward": 0.1,
                    "coverage": 0.9,
                }
            },
            "overall_mean_reward": 0.62,
            "overall_coverage": 0.9,
            "n_total": 5,
        },
    )
    assert pdf.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf
    assert b"qwen-prod" in pdf
    assert b"OK" in pdf  # verdict.upper()


def test_render_monitor_pdf_handles_empty_summary() -> None:
    from vlabs_api.monitor_pdf import render_monitor_pdf

    pdf = render_monitor_pdf(
        monitor_name="empty-test",
        monitor_id="mon_x",
        run_id="mr_y",
        scheduled_at="2026-05-09T06:00:00Z",
        finished_at="2026-05-09T06:01:00Z",
        verdict="failed",
        summary={"per_env": {}, "n_total": 0},
    )
    assert pdf.startswith(b"%PDF-1.4")
    assert b"no episodes recorded" in pdf
