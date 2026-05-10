"""Tests for scripts/preflight/check_deploy_readiness.sh."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "preflight" / "check_deploy_readiness.sh"


def _run_script(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )


def _base_env(**overrides: str) -> dict[str, str]:
    """Build an env dict with all required secrets present so we can
    isolate which check we're exercising. Names mirror the Pydantic
    Settings + the deploy provisioner (Phase 1+2 alignment)."""
    env = {
        "PATH": "/usr/bin:/bin",
        "DATABASE_URL": "postgres://stub",
        "VLABS_API_KEY_HASH_PEPPER": "stub",
        "UPSTASH_REDIS_REST_URL": "https://stub",
        "UPSTASH_REDIS_REST_TOKEN": "stub",
        "VLABS_R2_ACCOUNT_ID": "stub",
        "VLABS_R2_ACCESS_KEY_ID": "stub",
        "VLABS_R2_SECRET_ACCESS_KEY": "stub",
        "VLABS_R2_BUCKET_NAME": "stub",
        "VLABS_R2_PUBLIC_URL": "https://stub",
        "VLABS_DATA_LLM_KEY_ENCRYPTION": "stub",
        "VLABS_EMAIL_FROM": "stub@example.com",
        "VLABS_EMAIL_API_KEY": "stub",
        "CLERK_SECRET_KEY": "stub",
        "CLERK_PUBLISHABLE_KEY": "stub",
        "CLERK_JWT_ISSUER": "https://stub",
        "CLERK_JWKS_URL": "https://stub",
    }
    env.update(overrides)
    return env


def test_script_is_executable_or_runnable() -> None:
    """Script must exist + be readable; we invoke via bash so the
    +x bit is optional but the file has to be present."""
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_script_passes_with_full_env() -> None:
    """Happy path — every required secret in env, --no-fly so we don't
    poke flyctl."""
    r = _run_script(_base_env(), "--no-fly")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERDICT: GO" in r.stderr


def test_script_fails_with_missing_secret() -> None:
    env = _base_env()
    del env["DATABASE_URL"]
    r = _run_script(env, "--no-fly")
    assert r.returncode == 1
    assert "missing in env" in r.stderr or "missing in env" in r.stdout
    assert "DATABASE_URL" in r.stderr or "DATABASE_URL" in r.stdout


def test_script_fails_when_fake_email_set_to_true() -> None:
    env = _base_env(VLABS_LOCAL_FAKE_EMAIL="true")
    r = _run_script(env, "--no-fly")
    assert r.returncode == 1
    assert "VLABS_LOCAL_FAKE_EMAIL=true" in r.stderr


def test_script_fails_when_fake_pki_set_to_true() -> None:
    env = _base_env(VLABS_LOCAL_FAKE_PKI="true")
    r = _run_script(env, "--no-fly")
    assert r.returncode == 1
    assert "VLABS_LOCAL_FAKE_PKI=true" in r.stderr


def test_script_emits_json_when_requested() -> None:
    r = _run_script(_base_env(), "--no-fly", "--json")
    assert r.returncode == 0
    # Stdout is the JSON payload; stderr stays human-readable.
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "GO"
    assert payload["fail"] == 0
    assert payload["pass"] >= 5
    assert any(
        item["name"] == "fly_toml_present" for item in payload["results"]
    )


def test_script_unknown_arg_fails_fast() -> None:
    r = _run_script(_base_env(), "--no-such-flag")
    assert r.returncode == 2


def test_script_lists_all_nine_migrations() -> None:
    r = _run_script(_base_env(), "--no-fly", "--json")
    payload = json.loads(r.stdout)
    mig = next(
        item
        for item in payload["results"]
        if item["name"] == "migrations_0001_to_0009"
    )
    assert mig["status"] == "pass"
    assert "9 files" in mig["detail"]


def test_script_dockerfile_resolves_relative_to_fly_toml() -> None:
    r = _run_script(_base_env(), "--no-fly", "--json")
    payload = json.loads(r.stdout)
    docker_check = next(
        item
        for item in payload["results"]
        if item["name"] == "dockerfile_present"
    )
    assert docker_check["status"] == "pass"
