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
    """Modern ``Bearer {token}`` scheme returns 200 from /graphql POST.

    Note that probe_wandb uses ``_http_post`` (not ``_http_get``) for
    the W&B endpoint because /graphql is POST-only — see provider_
    status.py docstring."""

    def _stub_post(url, headers, json_body, timeout):
        assert "graphql" in url
        assert headers["Authorization"].startswith("Bearer ")
        assert json_body["query"] == "{ viewer { username } }"
        return 200, {"data": {"viewer": {"username": "stelios"}}}

    monkeypatch.setattr(ps, "_http_post", _stub_post)
    r = ps.probe_wandb("token")
    assert r.status == "ok"


def test_probe_wandb_falls_back_legacy_scheme(ps, monkeypatch) -> None:
    """First call (modern Bearer) 401, second call (legacy Bearer
    api:) 200 — probe_wandb returns ok."""
    calls: list[str] = []

    def _stub_post(url, headers, json_body, timeout):
        calls.append(headers["Authorization"])
        if headers["Authorization"].startswith("Bearer api:"):
            return 200, {"data": {"viewer": {"username": "stelios"}}}
        return 401, {"error": "Malformed token"}

    monkeypatch.setattr(ps, "_http_post", _stub_post)
    r = ps.probe_wandb("legacy-40-hex-key")
    assert r.status == "ok"
    assert len(calls) == 2  # tried modern first, then legacy


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


# ── Oracle ────────────────────────────────────────────────────────


def test_probe_oracle_unauth_when_ocid_fields_missing(
    ps, monkeypatch
) -> None:
    """The OCID quintet is required; if any of the four key fields
    is missing the probe must report unauth instead of attempting
    to import ``oci`` or hit any network."""
    for k in (
        "ORACLE_TENANCY_OCID",
        "ORACLE_USER_OCID",
        "ORACLE_FINGERPRINT",
        "ORACLE_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(k, raising=False)
    r = ps.probe_oracle("any-token")
    assert r.provider == "oracle"
    assert r.status == "unauth"
    assert "ORACLE_" in r.detail


def test_probe_oracle_config_only_when_oci_sdk_missing(
    ps, monkeypatch
) -> None:
    """All OCID fields present but ``oci`` sdk not installed → returns
    ok with a 'config-shape' detail flagging that the credentials
    weren't actually exercised. This keeps the preflight green for
    machines that don't have the SDK installed yet."""
    monkeypatch.setenv("ORACLE_TENANCY_OCID", "ocid1.tenancy.oc1..abc")
    monkeypatch.setenv("ORACLE_USER_OCID", "ocid1.user.oc1..xyz")
    monkeypatch.setenv("ORACLE_FINGERPRINT", "12:34:56:78:9a:bc:de:f0")
    monkeypatch.setenv("ORACLE_PRIVATE_KEY_PATH", "/tmp/no_such_key.pem")

    # Force ImportError on `import oci` by stubbing importlib.
    import builtins

    real_import = builtins.__import__

    def _fail_oci(name, *args, **kwargs):
        if name == "oci":
            raise ImportError("oci not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_oci)

    r = ps.probe_oracle("any-token")
    assert r.status == "ok"
    assert "config-shape" in r.detail
    assert r.gpu_available is True


def test_probe_oracle_unauth_when_keypair_path_missing(
    ps, monkeypatch, tmp_path
) -> None:
    """All env vars present, ``oci`` SDK present, but the keypair PEM
    file doesn't exist on disk — probe must report unauth with a
    'private key not found' detail rather than letting the SDK
    raise an opaque error."""
    monkeypatch.setenv("ORACLE_TENANCY_OCID", "ocid1.tenancy.oc1..abc")
    monkeypatch.setenv("ORACLE_USER_OCID", "ocid1.user.oc1..xyz")
    monkeypatch.setenv("ORACLE_FINGERPRINT", "12:34:56:78:9a:bc:de:f0")
    monkeypatch.setenv(
        "ORACLE_PRIVATE_KEY_PATH", str(tmp_path / "missing.pem")
    )

    # Provide a fake `oci` module so the real ImportError path is
    # not taken (we want the keypair-missing branch).
    import sys
    import types

    fake_oci = types.ModuleType("oci")
    fake_oci.config = types.SimpleNamespace(validate_config=lambda c: None)
    fake_oci.identity = types.SimpleNamespace(IdentityClient=lambda c: None)
    monkeypatch.setitem(sys.modules, "oci", fake_oci)

    r = ps.probe_oracle("any-token")
    assert r.status == "unauth"
    assert "private key not found" in r.detail


def test_oracle_registered_in_probes_dict(ps) -> None:
    """Pin the registry — Oracle must be reachable via the standard
    ``--only oracle`` CLI path (and via :func:`probe_all`).

    Gating on ``ORACLE_TENANCY_OCID`` (not the auth-token env var)
    because OCID + signed keypair is the canonical OCI auth path and
    the auth token is an opt-in alternative most users skip.
    """
    assert "oracle" in ps.PROBES
    env_var, func = ps.PROBES["oracle"]
    assert env_var == "ORACLE_TENANCY_OCID"
    assert func.__name__ == "probe_oracle"


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
