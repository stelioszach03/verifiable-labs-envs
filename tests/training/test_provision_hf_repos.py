"""Tests for scripts/training/provision_hf_repos.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "training" / "provision_hf_repos.py"


def _load_module():
    """Load the script as ``provision_hf_repos`` so dataclass forward
    refs inside ``RepoSpec`` resolve via ``sys.modules`` (same trick as
    tests/preflight/test_provider_status.py)."""
    spec = importlib.util.spec_from_file_location("provision_hf_repos", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ph():
    return _load_module()


class _FakeApi:
    """Minimal stub matching the slice of ``HfApi`` we use."""

    def __init__(self, owner: str = "stelioszach03"):
        self.calls: list[dict[str, Any]] = []
        self._owner = owner

    def whoami(self) -> dict[str, str]:
        return {"name": self._owner, "type": "user"}

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> None:
        self.calls.append(
            dict(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                exist_ok=exist_ok,
            )
        )


def test_repos_constant_lists_four_canonical_repos(ph) -> None:
    """Two datasets + two models = four total (RM + PRM × dataset + model)."""
    names = [r.repo_id_template for r in ph.REPOS]
    assert len(names) == 4
    assert "{owner}/rm-dataset-v0.0.1" in names
    assert "{owner}/prm-dataset-v0.0.1" in names
    assert "{owner}/rm-qwen-1-5b-v0.0.1" in names
    assert "{owner}/prm-qwen-1-5b-v0.0.1" in names
    types = sorted(r.repo_type for r in ph.REPOS)
    assert types == ["dataset", "dataset", "model", "model"]


def test_provision_one_dry_run_skips_api_call(ph) -> None:
    api = _FakeApi()
    spec = ph.REPOS[0]
    out = ph.provision_one(api, spec, owner="stelioszach03", dry_run=True)
    assert out["status"] == "dry-run"
    assert out["repo_id"] == "stelioszach03/rm-dataset-v0.0.1"
    # No calls made.
    assert api.calls == []


def test_provision_one_calls_create_repo_with_exist_ok(ph) -> None:
    api = _FakeApi()
    spec = ph.REPOS[0]
    out = ph.provision_one(api, spec, owner="stelioszach03", dry_run=False)
    assert out["status"] == "ok"
    assert len(api.calls) == 1
    call = api.calls[0]
    assert call["repo_id"] == "stelioszach03/rm-dataset-v0.0.1"
    assert call["repo_type"] == "dataset"
    assert call["exist_ok"] is True


def test_provision_all_creates_four_repos(ph, monkeypatch) -> None:
    api = _FakeApi(owner="stelioszach03")

    def _fake_hfapi(token):
        return api

    # Monkey-patch the import inside provision_all().
    fake_hf = type(sys)("huggingface_hub")
    fake_hf.HfApi = lambda token=None: api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    results = ph.provision_all(token="t")
    assert len(results) == 4
    assert all(r["status"] == "ok" for r in results)
    assert {r["repo_id"] for r in results} == {
        "stelioszach03/rm-dataset-v0.0.1",
        "stelioszach03/prm-dataset-v0.0.1",
        "stelioszach03/rm-qwen-1-5b-v0.0.1",
        "stelioszach03/prm-qwen-1-5b-v0.0.1",
    }
    assert len(api.calls) == 4


def test_provision_all_raises_when_whoami_returns_no_name(ph, monkeypatch) -> None:
    class _NoName:
        def whoami(self):
            return {"type": "user"}

        def create_repo(self, **_):
            return None

    fake_hf = type(sys)("huggingface_hub")
    fake_hf.HfApi = lambda token=None: _NoName()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(RuntimeError, match="no 'name' field"):
        ph.provision_all(token="t")


def test_main_missing_token_returns_2(ph, monkeypatch, capsys) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Also block the env file source so the test doesn't pick up the
    # real token from ~/.vlabs-secrets/.
    monkeypatch.setattr(ph, "_load_env_file", lambda _p: None)
    rc = ph.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "HF_TOKEN" in err


def test_main_dry_run_emits_json(ph, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HF_TOKEN", "fake")

    api = _FakeApi(owner="alice")
    fake_hf = type(sys)("huggingface_hub")
    fake_hf.HfApi = lambda token=None: api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    rc = ph.main(["--dry-run", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert len(payload) == 4
    assert all(r["status"] == "dry-run" for r in payload)
    # No real API calls when --dry-run.
    assert api.calls == []
