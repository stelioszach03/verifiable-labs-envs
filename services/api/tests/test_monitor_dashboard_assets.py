"""Static-asset sanity checks for the Phase 28.E dashboard pages.

The Next.js dashboard pages live under ``services/landing/app/dashboard/
monitors/``. A full e2e Playwright suite is out of scope for v0.0.1
alpha — these tests verify that the .tsx files exist + reference the
expected API endpoints so a future schema rename can't silently break
the dashboard.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASH_ROOT = REPO_ROOT / "services" / "landing" / "app" / "dashboard" / "monitors"


def test_monitor_index_page_exists() -> None:
    p = DASH_ROOT / "page.tsx"
    assert p.exists(), f"missing {p}"


def test_monitor_detail_page_exists() -> None:
    p = DASH_ROOT / "[id]" / "page.tsx"
    assert p.exists(), f"missing {p}"


def test_monitor_run_page_exists() -> None:
    p = DASH_ROOT / "[id]" / "runs" / "[rid]" / "page.tsx"
    assert p.exists(), f"missing {p}"


def test_monitor_index_page_fetches_v1_monitors() -> None:
    text = (DASH_ROOT / "page.tsx").read_text(encoding="utf-8")
    assert "/v1/monitors" in text


def test_monitor_detail_page_fetches_runs_endpoint() -> None:
    text = (DASH_ROOT / "[id]" / "page.tsx").read_text(encoding="utf-8")
    assert "/v1/monitors/" in text
    assert "/runs" in text


def test_monitor_run_page_fetches_run_detail_endpoint() -> None:
    text = (DASH_ROOT / "[id]" / "runs" / "[rid]" / "page.tsx").read_text(
        encoding="utf-8"
    )
    assert "/v1/monitors/" in text
    assert "/runs/" in text


def test_dashboard_pages_use_x_vlabs_key_header() -> None:
    """Every monitor dashboard page authenticates via X-Vlabs-Key."""
    for sub in ("page.tsx", "[id]/page.tsx", "[id]/runs/[rid]/page.tsx"):
        text = (DASH_ROOT / sub).read_text(encoding="utf-8")
        assert "X-Vlabs-Key" in text, f"missing X-Vlabs-Key in {sub}"


def test_dashboard_pages_use_clerk_auth() -> None:
    """Every monitor dashboard page imports Clerk's `auth` helper."""
    for sub in ("page.tsx", "[id]/page.tsx", "[id]/runs/[rid]/page.tsx"):
        text = (DASH_ROOT / sub).read_text(encoding="utf-8")
        assert "@clerk/nextjs" in text, f"missing Clerk import in {sub}"


def test_dashboard_run_page_handles_three_verdict_branches() -> None:
    """The run page renders a verdict color for ok / warning / regressed."""
    text = (DASH_ROOT / "[id]" / "runs" / "[rid]" / "page.tsx").read_text(
        encoding="utf-8"
    )
    assert "ok" in text
    assert "warning" in text
    assert "regressed" in text


def test_dashboard_pages_runtime_edge() -> None:
    """All pages export `runtime = "edge"` for deploy compatibility."""
    for sub in ("page.tsx", "[id]/page.tsx", "[id]/runs/[rid]/page.tsx"):
        text = (DASH_ROOT / sub).read_text(encoding="utf-8")
        assert 'runtime = "edge"' in text or "runtime = 'edge'" in text


def test_dashboard_index_links_to_monitor_detail() -> None:
    """Index page must link each row to /dashboard/monitors/{id}."""
    text = (DASH_ROOT / "page.tsx").read_text(encoding="utf-8")
    assert "/dashboard/monitors/" in text
