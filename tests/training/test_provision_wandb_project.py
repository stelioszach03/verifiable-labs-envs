"""Tests for scripts/training/provision_wandb_project.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "training" / "provision_wandb_project.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "provision_wandb_project", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pw():
    return _load_module()


# ── modern wandb.Api.create_project path ─────────────────────────


class _ModernApi:
    """Stub of ``wandb.Api`` exposing ``create_project`` (≥ 0.17)."""

    def __init__(
        self,
        *,
        default_entity: str = "stelioszach03",
        raise_exists: bool = False,
        raise_other: Exception | None = None,
    ):
        self.default_entity = default_entity
        self.calls: list[tuple[str, str | None]] = []
        self._raise_exists = raise_exists
        self._raise_other = raise_other

    def create_project(self, name: str, entity: str | None = None) -> None:
        self.calls.append((name, entity))
        if self._raise_other is not None:
            raise self._raise_other
        if self._raise_exists:
            raise RuntimeError("project already exists")


class _ModernWandb:
    """Stub of the wandb module with ``Api`` returning :class:`_ModernApi`."""

    def __init__(self, api: _ModernApi):
        self._api = api
        self.login_calls: list[dict[str, Any]] = []

    # The script does ``wandb_module.Api()`` (no args).
    def Api(self) -> _ModernApi:  # noqa: N802 — match wandb naming
        return self._api

    def login(self, *, key: str, anonymous: str, relogin: bool) -> None:
        self.login_calls.append({"anonymous": anonymous, "relogin": relogin})


# ── legacy init/finish fallback path ─────────────────────────────


class _LegacyApi:
    """Stub of an older ``wandb.Api`` that lacks ``create_project``."""

    default_entity = "stelioszach03"

    # Note: deliberately no ``create_project`` attribute — the
    # AttributeError is what triggers the fallback path.


class _LegacyWandb:
    """Stub wandb module simulating SDK < 0.17 (no ``create_project``)."""

    def __init__(
        self,
        *,
        entity: str = "stelioszach03",
        url: str = "https://wandb.ai/stelioszach03/verifiable-labs/runs/abc",
    ):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._entity = entity
        self._url = url

    def Api(self) -> _LegacyApi:  # noqa: N802 — match wandb naming
        return _LegacyApi()

    def login(self, *, key: str, anonymous: str, relogin: bool) -> None:
        self.calls.append(
            ("login", {"anonymous": anonymous, "relogin": relogin})
        )

    def init(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("init", kwargs))
        return SimpleNamespace(entity=self._entity, url=self._url)

    def finish(self) -> None:
        self.calls.append(("finish", {}))


# ── tests ────────────────────────────────────────────────────────


def test_provision_dry_run_skips_network(pw) -> None:
    fake = _ModernWandb(_ModernApi())
    out = pw.provision_project(
        "verifiable-labs", api_key="t", dry_run=True, wandb_module=fake
    )
    assert out["status"] == "dry-run"
    assert out["project"] == "verifiable-labs"
    # No login or API calls when dry-run.
    assert fake.login_calls == []


def test_provision_modern_api_creates_project(pw) -> None:
    api = _ModernApi(default_entity="stelioszach03")
    fake = _ModernWandb(api)
    out = pw.provision_project(
        "verifiable-labs", api_key="key123", wandb_module=fake
    )
    assert out["status"] == "created"
    assert out["entity"] == "stelioszach03"
    assert api.calls == [("verifiable-labs", "stelioszach03")]
    # Login fired exactly once.
    assert len(fake.login_calls) == 1


def test_provision_modern_api_treats_already_exists_as_success(pw) -> None:
    api = _ModernApi(raise_exists=True)
    fake = _ModernWandb(api)
    out = pw.provision_project(
        "verifiable-labs", api_key="k", wandb_module=fake
    )
    assert out["status"] == "exists"
    assert api.calls == [("verifiable-labs", "stelioszach03")]


def test_provision_modern_api_propagates_other_errors(pw) -> None:
    api = _ModernApi(raise_other=RuntimeError("boom — auth failed"))
    fake = _ModernWandb(api)
    with pytest.raises(RuntimeError, match="boom"):
        pw.provision_project("p", api_key="k", wandb_module=fake)


def test_provision_falls_back_to_init_when_create_project_missing(
    pw,
) -> None:
    fake = _LegacyWandb()
    out = pw.provision_project("p", api_key="k", wandb_module=fake)
    assert out["status"] == "ok"
    seq = [c[0] for c in fake.calls]
    # Legacy path: login → init → finish.
    assert seq == ["login", "init", "finish"]


def test_provision_legacy_init_passes_marker_metadata(pw) -> None:
    fake = _LegacyWandb()
    pw.provision_project("foo", api_key="k", wandb_module=fake)
    init_kwargs = next(c[1] for c in fake.calls if c[0] == "init")
    assert init_kwargs["project"] == "foo"
    assert init_kwargs["name"] == "provisioning"
    assert init_kwargs["job_type"] == "provision"


def test_provision_sets_wandb_api_key_in_env(pw, monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    fake = _ModernWandb(_ModernApi())
    pw.provision_project("p", api_key="my-secret", wandb_module=fake)
    import os

    assert os.environ.get("WANDB_API_KEY") == "my-secret"


def test_main_missing_token_returns_2(pw, monkeypatch, capsys) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(pw, "_load_env_file", lambda _p: None)
    rc = pw.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "WANDB_API_KEY" in err


def test_main_dry_run_emits_json(pw, monkeypatch, capsys) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    rc = pw.main(["--dry-run", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload["status"] == "dry-run"
    assert payload["project"] == "verifiable-labs"
