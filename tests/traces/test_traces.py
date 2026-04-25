"""Tests for the canonical JSONL trace format."""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs.traces import (
    SCHEMA_VERSION,
    FailureType,
    Trace,
    hash_payload,
    read_jsonl,
    write_jsonl,
)


def test_new_fills_id_and_timestamp():
    t = Trace.new(
        env_name="sparse-fourier-recovery",
        agent_name="zero",
        reward=0.0,
        parse_success=True,
    )
    assert t.trace_id.startswith("t_")
    assert len(t.trace_id) > 5
    assert "T" in t.timestamp  # ISO-8601 marker
    assert t.schema_version == SCHEMA_VERSION
    assert t.failure_type == FailureType.NONE


def test_to_dict_drops_none_optionals_keeps_required():
    t = Trace.new(
        env_name="env-a",
        agent_name="zero",
        reward=0.5,
        parse_success=False,
    )
    d = t.to_dict()
    # Required keys always present.
    for k in ("schema_version", "trace_id", "env_name", "agent_name", "reward", "parse_success"):
        assert k in d
    # Optional None fields not present.
    assert "seed" not in d
    assert "coverage" not in d
    # Empty list / dict optionals also dropped.
    assert "artifacts" not in d
    assert "metadata" not in d
    # parse_success is preserved as bool, not coerced.
    assert d["parse_success"] is False


def test_to_dict_keeps_optional_when_set():
    t = Trace.new(
        env_name="env-a",
        agent_name="zero",
        reward=0.5,
        parse_success=True,
        seed=42,
        coverage=0.91,
        reward_components={"nmse": 0.6, "support": 0.7},
        artifacts=["runs/log.json"],
    )
    d = t.to_dict()
    assert d["seed"] == 42
    assert d["coverage"] == pytest.approx(0.91)
    assert d["reward_components"] == {"nmse": 0.6, "support": 0.7}
    assert d["artifacts"] == ["runs/log.json"]


def test_from_dict_round_trip():
    original = Trace.new(
        env_name="env-a",
        agent_name="zero",
        reward=0.5,
        parse_success=True,
        seed=7,
        latency_ms=123.4,
        failure_type=FailureType.NONE,
    )
    d = original.to_dict()
    restored = Trace.from_dict(d)
    assert restored.env_name == original.env_name
    assert restored.seed == 7
    assert restored.latency_ms == pytest.approx(123.4)
    assert restored.failure_type == FailureType.NONE


def test_from_dict_tolerates_missing_optionals():
    minimal = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": "t_abc123",
        "env_name": "env-a",
        "agent_name": "zero",
        "reward": 0.5,
        "parse_success": True,
    }
    t = Trace.from_dict(minimal)
    assert t.seed is None
    assert t.failure_type == FailureType.NONE
    assert t.reward_components == {}


def test_from_dict_rejects_missing_required_key():
    bad = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": "t_abc123",
        "env_name": "env-a",
        "agent_name": "zero",
        # missing reward + parse_success
    }
    with pytest.raises(ValueError, match="missing required keys"):
        Trace.from_dict(bad)


def test_from_dict_ignores_unknown_keys():
    d = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": "t_xyz",
        "env_name": "env-a",
        "agent_name": "zero",
        "reward": 0.5,
        "parse_success": True,
        "future_field_we_havent_invented_yet": [1, 2, 3],
    }
    t = Trace.from_dict(d)
    assert t.env_name == "env-a"


def test_failure_type_enum_round_trip():
    for ft in FailureType:
        t = Trace.new(
            env_name="env",
            agent_name="zero",
            reward=0.0,
            parse_success=False,
            failure_type=ft,
        )
        d = t.to_dict()
        assert d["failure_type"] == ft.value
        restored = Trace.from_dict(d)
        assert restored.failure_type == ft


def test_jsonl_round_trip(tmp_path):
    traces = [
        Trace.new(env_name="a", agent_name="zero", reward=0.1, parse_success=True, seed=0),
        Trace.new(env_name="a", agent_name="zero", reward=0.4, parse_success=True, seed=1),
        Trace.new(
            env_name="a", agent_name="zero", reward=0.0, parse_success=False, seed=2,
            failure_type=FailureType.PARSE_ERROR,
        ),
    ]
    path = tmp_path / "out.jsonl"
    n = write_jsonl(traces, path)
    assert n == 3
    text = path.read_text()
    assert text.count("\n") == 3
    # Each line is valid JSON.
    for line in text.splitlines():
        json.loads(line)
    restored = read_jsonl(path)
    assert len(restored) == 3
    assert restored[0].seed == 0
    assert restored[2].failure_type == FailureType.PARSE_ERROR


def test_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"schema_version":1,"trace_id":"t_a","env_name":"e","agent_name":"z","reward":0.5,"parse_success":true}\n'
        "\n"
        '{"schema_version":1,"trace_id":"t_b","env_name":"e","agent_name":"z","reward":0.6,"parse_success":true}\n'
    )
    out = read_jsonl(path)
    assert [t.trace_id for t in out] == ["t_a", "t_b"]


def test_jsonl_raises_on_malformed_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"schema_version":1,"trace_id":"t_a","env_name":"e","agent_name":"z","reward":0.5,"parse_success":true}\n'
        "not json at all\n"
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        read_jsonl(path)


def test_jsonl_raises_on_missing_required_key_with_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"schema_version":1,"trace_id":"t_a","env_name":"e"}\n')
    with pytest.raises(ValueError, match=":1.*missing required keys"):
        read_jsonl(path)


def test_jsonl_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nope.jsonl")


def test_jsonl_creates_parent_dirs(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "out.jsonl"
    n = write_jsonl(
        [Trace.new(env_name="e", agent_name="z", reward=0.1, parse_success=True)], deep,
    )
    assert n == 1
    assert deep.exists()


def test_hash_payload_stable_and_short():
    h1 = hash_payload({"support_idx": [1, 2, 3], "support_amp_x1000": [10, 20, 30]})
    h2 = hash_payload({"support_amp_x1000": [10, 20, 30], "support_idx": [1, 2, 3]})
    assert h1 == h2  # key order doesn't matter
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 16


def test_to_json_is_valid_json():
    t = Trace.new(
        env_name="env",
        agent_name="zero",
        reward=0.5,
        parse_success=True,
        seed=42,
        reward_components={"nmse": 0.7},
    )
    parsed = json.loads(t.to_json())
    assert parsed["env_name"] == "env"
    assert parsed["seed"] == 42
    assert parsed["reward_components"]["nmse"] == 0.7
