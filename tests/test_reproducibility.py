"""Tests for ``verifiable_labs_envs.repro`` and CLI metadata wiring.

Covers the four reproducibility helpers
(``canonical_json``, ``config_hash``, ``instance_hash``, ``reward_hash``)
plus an end-to-end ``verifiable run`` invocation that asserts every
trace JSONL line carries the three hash keys in its ``metadata`` field.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifiable_labs_envs.repro import (
    EXCLUDED_FROM_CONFIG_HASH,
    canonical_json,
    config_hash,
    instance_hash,
    reward_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ── canonical_json ────────────────────────────────────────────────────


def test_canonical_json_sorted_keys_stable() -> None:
    a = {"b": 2, "a": 1, "c": [3, 1, 2]}
    b = {"c": [3, 1, 2], "a": 1, "b": 2}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_no_whitespace() -> None:
    s = canonical_json({"a": 1, "b": [2, 3]})
    assert " " not in s
    assert "\n" not in s


def test_canonical_json_ascii_safe() -> None:
    s = canonical_json({"name": "Στέλιος"})
    # ensure_ascii=True → non-ASCII is escaped as \uXXXX
    assert "\\u" in s
    # And the canonical string is pure ASCII bytes:
    s.encode("ascii")  # raises if non-ASCII slipped through


# ── config_hash ───────────────────────────────────────────────────────


def test_config_hash_identical_dicts_match() -> None:
    cfg = {"env_id": "x", "n": 5, "lr": 1e-6}
    assert config_hash(cfg) == config_hash(dict(cfg))


def test_config_hash_reordered_keys_match() -> None:
    a = {"alpha": 1, "beta": 2, "gamma": 3}
    b = {"gamma": 3, "alpha": 1, "beta": 2}
    assert config_hash(a) == config_hash(b)


def test_config_hash_excludes_runtime_only_fields() -> None:
    base = {"env_id": "x", "n": 5}
    decorated = {
        **base,
        "timestamp": "2026-04-29T12:00:00Z",
        "wandb_run_id": "abc-123",
        "host": "kevin",
    }
    assert config_hash(base) == config_hash(decorated)
    # Sanity: the EXCLUDED set is exactly what the decorated dict added.
    assert EXCLUDED_FROM_CONFIG_HASH == frozenset(
        {"timestamp", "wandb_run_id", "host"}
    )


def test_config_hash_distinguishes_distinct_configs() -> None:
    h1 = config_hash({"env_id": "a", "n": 5})
    h2 = config_hash({"env_id": "b", "n": 5})
    h3 = config_hash({"env_id": "a", "n": 6})
    assert h1 != h2 != h3
    assert h1 != h3


def test_config_hash_equivalent_floats_match_when_value_equal() -> None:
    """1.5 vs 1.5 → identical hash; 1 vs 1.0 → different (JSON repr differs)."""
    assert config_hash({"x": 1.5}) == config_hash({"x": 1.5})
    # Subtle: Python's int 1 and float 1.0 serialize differently in JSON.
    assert config_hash({"x": 1}) != config_hash({"x": 1.0})


def test_config_hash_format_is_sha256_prefixed() -> None:
    h = config_hash({"env_id": "x"})
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 16  # 16 hex chars after the prefix


# ── instance_hash ─────────────────────────────────────────────────────


def test_instance_hash_determinism_across_invocations() -> None:
    args = ("env_a", "0.0.1", 42, {"sigma": 0.05, "k": 10})
    h1 = instance_hash(*args)
    h2 = instance_hash(*args)
    assert h1 == h2


def test_instance_hash_distinguishes_seeds() -> None:
    h1 = instance_hash("env_a", "0.0.1", 42, {})
    h2 = instance_hash("env_a", "0.0.1", 43, {})
    assert h1 != h2


def test_instance_hash_distinguishes_env_versions() -> None:
    h1 = instance_hash("env_a", "0.0.1", 42, {})
    h2 = instance_hash("env_a", "0.0.2", 42, {})
    assert h1 != h2


def test_instance_hash_distinguishes_prior_params() -> None:
    h1 = instance_hash("env_a", "0.0.1", 42, {"sigma": 0.05})
    h2 = instance_hash("env_a", "0.0.1", 42, {"sigma": 0.10})
    assert h1 != h2


def test_instance_hash_none_prior_params_normalised_to_empty() -> None:
    h_none = instance_hash("env_a", "0.0.1", 42, None)
    h_empty = instance_hash("env_a", "0.0.1", 42, {})
    assert h_none == h_empty


def test_instance_hash_format_is_sha256_prefixed() -> None:
    h = instance_hash("env_a", "0.0.1", 42, {})
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 16


# ── reward_hash ───────────────────────────────────────────────────────


def test_reward_hash_six_decimal_rounding() -> None:
    assert reward_hash(0.123456789) == "0.123457"


def test_reward_hash_half_pads_zeros() -> None:
    assert reward_hash(0.5) == "0.500000"


def test_reward_hash_zero() -> None:
    assert reward_hash(0.0) == "0.000000"


def test_reward_hash_negative() -> None:
    assert reward_hash(-0.5) == "-0.500000"


def test_reward_hash_returns_string() -> None:
    h = reward_hash(0.342)
    assert isinstance(h, str)
    assert "." in h
    # Should always have exactly 6 digits after the decimal.
    assert len(h.split(".")[-1]) == 6


def test_reward_hash_accepts_int_via_float_coercion() -> None:
    assert reward_hash(1) == "1.000000"


# ── end-to-end CLI ────────────────────────────────────────────────────


def test_e2e_cli_run_traces_have_three_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run ``verifiable run --n 3`` programmatically and assert that
    every trace JSONL line has ``config_hash``, ``instance_hash`` and
    ``reward_hash`` populated in its ``metadata`` field."""
    from verifiable_labs_envs.cli import main

    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "m3_check.jsonl"
    rc = main([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", "examples/agents/zero_agent.py",
        "--n", "3",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0, "verifiable run exited non-zero"

    lines = out.read_text().splitlines()
    assert len(lines) == 3, f"expected 3 traces, got {len(lines)}"

    seen_instance_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for line in lines:
        d = json.loads(line)
        assert "metadata" in d, f"trace missing metadata: {d}"
        meta = d["metadata"]

        # All 3 hash keys present.
        for key in ("config_hash", "instance_hash", "reward_hash"):
            assert key in meta, f"metadata missing {key!r}: {meta}"

        # Format checks.
        assert meta["config_hash"].startswith("sha256:")
        assert meta["instance_hash"].startswith("sha256:")
        assert isinstance(meta["reward_hash"], str)
        assert "." in meta["reward_hash"]

        # reward_hash must be the 6-decimal repr of the trace's reward.
        assert meta["reward_hash"] == f"{d['reward']:.6f}"

        # config_hash is per-run → all 3 traces share one value.
        config_hashes.add(meta["config_hash"])
        # instance_hash is per-episode → all 3 must differ (seeds 0, 1, 2).
        seen_instance_hashes.add(meta["instance_hash"])

    assert len(config_hashes) == 1, f"config_hash drifted across traces: {config_hashes}"
    assert len(seen_instance_hashes) == 3, (
        f"expected 3 distinct instance_hashes for 3 distinct seeds, "
        f"got {len(seen_instance_hashes)}"
    )


def test_e2e_cli_run_two_invocations_share_config_and_instance_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running with identical args reproduces identical
    ``config_hash`` AND identical per-seed ``instance_hash`` values."""
    from verifiable_labs_envs.cli import main

    monkeypatch.chdir(REPO_ROOT)

    def _run(out_path: Path) -> list[dict]:
        rc = main([
            "run",
            "--env", "sparse-fourier-recovery",
            "--agent", "examples/agents/zero_agent.py",
            "--n", "2",
            "--out", str(out_path),
            "--quiet",
        ])
        assert rc == 0
        return [json.loads(l) for l in out_path.read_text().splitlines()]

    a = _run(tmp_path / "a.jsonl")
    b = _run(tmp_path / "b.jsonl")
    assert len(a) == len(b) == 2
    for ta, tb in zip(a, b, strict=True):
        assert ta["metadata"]["config_hash"] == tb["metadata"]["config_hash"]
        assert ta["metadata"]["instance_hash"] == tb["metadata"]["instance_hash"]
        assert ta["metadata"]["reward_hash"] == tb["metadata"]["reward_hash"]
