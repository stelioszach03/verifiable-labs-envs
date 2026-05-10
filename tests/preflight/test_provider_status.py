"""Tests for scripts/preflight/provider_status.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "preflight" / "provider_status.py"


def _load_module():
    """Load provider_status.py as a module.

    Registers the module in ``sys.modules`` BEFORE ``exec_module``
    runs because the dataclasses machinery looks the owning module up
    in ``sys.modules`` to evaluate forward references on
    :class:`ProviderResult`. Skipping that step yields
    ``AttributeError: 'NoneType' has no attribute '__dict__'``.
    """
    spec = importlib.util.spec_from_file_location("provider_status", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ps():
    return _load_module()


# ── individual probe parsers ───────────────────────────────────────


def test_probe_vultr_ok(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        assert "vultr" in url
        return 200, {"account": {"balance": 250.5}}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_vultr("token")
    assert r.provider == "vultr"
    assert r.status == "ok"
    assert r.balance_usd == 250.5


def test_probe_vultr_no_credit(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"account": {"balance": 0}}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_vultr("token")
    assert r.status == "no_credit"
    assert r.balance_usd == 0.0


def test_probe_vultr_unauth(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 401, {"error": "invalid"}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_vultr("bad")
    assert r.status == "unauth"


def test_probe_runpod_unauth(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 401, "denied"

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_runpod("bad")
    assert r.status == "unauth"
    assert r.gpu_available is None


def test_probe_runpod_ok_marks_gpu_available(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"data": {"myself": {"balance": 5.0}}}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_runpod("token")
    assert r.status == "ok"
    assert r.gpu_available is True


def test_probe_digitalocean_ok(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"account": {"email": "x@y.com"}}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_digitalocean("token")
    assert r.status == "ok"
    assert r.gpu_available is True


def test_probe_hf_ok(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"name": "stelios"}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_hf("token")
    assert r.status == "ok"
    assert r.gpu_available is False


def test_probe_hf_unauth(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 401, "bad"

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_hf("bad")
    assert r.status == "unauth"


def test_probe_wandb_ok(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"data": {"viewer": {"name": "stelios"}}}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_wandb("token")
    assert r.status == "ok"


def test_probe_openrouter_with_remaining_balance(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"data": {"usage": 1.0, "limit": 5.0}}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_openrouter("token")
    assert r.status == "ok"
    assert r.balance_usd == 4.0


def test_probe_openrouter_no_credit(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"data": {"usage": 5.0, "limit": 5.0}}

    monkeypatch.setattr(ps, "_http_get", _stub)
    r = ps.probe_openrouter("token")
    assert r.status == "no_credit"
    assert r.balance_usd == 0.0


# ── probe_all + render ─────────────────────────────────────────────


def test_probe_all_marks_unset_provider_as_unauth(ps) -> None:
    results = ps.probe_all(env={}, only=["hf"])
    assert len(results) == 1
    assert results[0].status == "unauth"
    assert results[0].detail == "no token"


def test_probe_all_only_filter_subsets_providers(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        return 200, {"name": "stelios"}

    monkeypatch.setattr(ps, "_http_get", _stub)
    results = ps.probe_all(env={"HF_TOKEN": "t"}, only=["hf"])
    assert {r.provider for r in results} == {"hf"}


def test_probe_all_catches_exceptions_in_probe(ps, monkeypatch) -> None:
    def _stub(url, headers, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(ps, "_http_get", _stub)
    results = ps.probe_all(env={"HF_TOKEN": "t"}, only=["hf"])
    assert results[0].status == "error"
    assert "RuntimeError" in results[0].detail


def test_render_csv_has_header_and_rows(ps) -> None:
    results = [
        ps.ProviderResult("vultr", "ok", 100.0, None),
        ps.ProviderResult("hf", "unauth", None, False),
    ]
    out = ps.render_csv(results)
    lines = out.strip().splitlines()
    assert lines[0] == "provider,status,balance_usd,gpu_available"
    assert lines[1].startswith("vultr,ok,100.00,")
    assert lines[2].startswith("hf,unauth,,false")


def test_render_json_round_trip(ps) -> None:
    results = [
        ps.ProviderResult("vultr", "ok", 250.5, True, "detail"),
    ]
    payload = json.loads(ps.render_json(results))
    assert payload[0]["provider"] == "vultr"
    assert payload[0]["balance_usd"] == 250.5


def test_main_csv_default(ps, monkeypatch, capsys) -> None:
    rc = ps.main(["--only", "hf"])
    out = capsys.readouterr().out
    assert "provider,status" in out
    # No HF_TOKEN in os.environ → unauth → return code stays 0 (no
    # 'error' status), only network/parse failures bump rc.
    assert rc == 0


def test_main_json_mode(ps, monkeypatch, capsys) -> None:
    rc = ps.main(["--json", "--only", "hf"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert payload[0]["provider"] == "hf"
    assert rc == 0
