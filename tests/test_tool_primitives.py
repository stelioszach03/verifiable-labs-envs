"""Determinism + behaviour tests for the shared tool-primitives library
(Phase 25.B).

PHASE_25_PLAN.md §16 Check 6 mandates the 6 named tests below; the
rest of the file pins down the calculator AST sandbox, web_search
ranking shape, error-payload conventions, and schema integrity.
"""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs.tool_primitives import (
    TOOL_DISPATCH,
    TOOL_SCHEMAS,
    WorkspaceState,
    canonical_action_hash,
    dispatch_tool,
    init_state,
    schemas_for,
)

# ── Mandatory determinism checks (Phase 25 gate Check 6) ─────────────


def test_calculator_seed_determinism() -> None:
    """Same `(seed, expression)` yields a byte-identical state delta."""
    s1 = init_state(seed=42)
    s2 = init_state(seed=42)
    r1 = dispatch_tool("calculator", {"expression": "3 * (4 + 5)"}, s1)
    r2 = dispatch_tool("calculator", {"expression": "3 * (4 + 5)"}, s2)
    assert r1 == r2
    assert s1.calculator_history == s2.calculator_history


def test_web_search_seed_determinism() -> None:
    """Same `(seed, query, top_k)` yields the same ranked result list."""
    s1 = init_state(seed=7)
    s2 = init_state(seed=7)
    r1 = dispatch_tool("web_search", {"query": "fourier transform", "top_k": 5}, s1)
    r2 = dispatch_tool("web_search", {"query": "fourier transform", "top_k": 5}, s2)
    assert r1 == r2
    assert [d["id"] for d in r1["results"]] == [d["id"] for d in r2["results"]]


def test_read_write_file_workspace_state() -> None:
    """write_file persists in state.files; read_file returns the same string."""
    s = init_state(seed=0)
    w = dispatch_tool(
        "write_file",
        {"path": "note.txt", "content": "hello world"},
        s,
    )
    assert w["ok"] is True
    assert w["bytes_written"] == len(b"hello world")
    r = dispatch_tool("read_file", {"path": "note.txt"}, s)
    assert r == {"content": "hello world"}
    assert s.files == {"note.txt": "hello world"}


def test_send_message_workspace_mutation() -> None:
    """send_message appends to state.outbox with stable delivery_id."""
    s = init_state(seed=0)
    r1 = dispatch_tool(
        "send_message",
        {"to": "alice@example.com", "body": "hi"},
        s,
    )
    r2 = dispatch_tool(
        "send_message",
        {"to": "bob@example.com", "body": "hello"},
        s,
    )
    assert r1["delivery_id"] == "msg_000001"
    assert r2["delivery_id"] == "msg_000002"
    assert len(s.outbox) == 2
    assert s.outbox[0]["to"] == "alice@example.com"
    assert s.outbox[1]["body"] == "hello"


def test_workspace_state_serialization_roundtrip() -> None:
    """to_serialisable / from_serialisable round-trip preserves content."""
    s = init_state(seed=11, initial_files={"a.txt": "alpha"})
    dispatch_tool("write_file", {"path": "b.txt", "content": "beta"}, s)
    dispatch_tool("calculator", {"expression": "2 + 2"}, s)
    dispatch_tool("send_message", {"to": "x", "body": "y"}, s)
    dispatch_tool("web_search", {"query": "fourier"}, s)

    payload = s.to_serialisable()
    blob = json.dumps(payload, sort_keys=True)
    rehydrated = WorkspaceState.from_serialisable(json.loads(blob))

    assert rehydrated.files == s.files
    assert rehydrated.outbox == s.outbox
    assert rehydrated.calculator_history == s.calculator_history
    assert rehydrated.web_search_calls == s.web_search_calls
    assert rehydrated.seed == s.seed


def test_action_hash_stability() -> None:
    """canonical_action_hash is stable under arg-key reordering."""
    a = [
        {"name": "calculator", "arguments": {"expression": "1 + 2"}},
        {"name": "send_message", "arguments": {"to": "alice", "body": "hi"}},
    ]
    b = [
        {"name": "calculator", "arguments": {"expression": "1 + 2"}},
        # reorder dict keys; canonical JSON sorts them.
        {"name": "send_message", "arguments": {"body": "hi", "to": "alice"}},
    ]
    assert canonical_action_hash(a) == canonical_action_hash(b)
    # Hash changes when the call order changes.
    c = list(reversed(a))
    assert canonical_action_hash(a) != canonical_action_hash(c)
    # Hash format: 16 lowercase hex chars.
    h = canonical_action_hash(a)
    assert len(h) == 16
    int(h, 16)


# ── Calculator ───────────────────────────────────────────────────────


def test_calculator_handles_constants_and_functions() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", {"expression": "sqrt(16) + 2"}, s)
    assert r == {"value": 6.0}


def test_calculator_rejects_arbitrary_python() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", {"expression": "__import__('os').system('echo pwned')"}, s)
    assert "error" in r
    # No mutation happens on error — history stays empty.
    assert s.calculator_history == []


def test_calculator_rejects_undefined_name() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", {"expression": "x + 1"}, s)
    assert "error" in r


def test_calculator_division_by_zero_returns_error() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", {"expression": "1 / 0"}, s)
    assert "error" in r


def test_calculator_missing_argument_returns_error() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", {}, s)
    assert "error" in r


# ── web_search ───────────────────────────────────────────────────────


def test_web_search_returns_at_most_top_k() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("web_search", {"query": "fourier", "top_k": 2}, s)
    assert len(r["results"]) <= 2


def test_web_search_clamps_invalid_top_k() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("web_search", {"query": "fourier", "top_k": 50}, s)
    assert "error" in r


def test_web_search_records_call_in_state() -> None:
    s = init_state(seed=0)
    dispatch_tool("web_search", {"query": "phase retrieval"}, s)
    assert s.web_search_calls == ["phase retrieval"]


def test_web_search_empty_query_errors() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("web_search", {"query": "  "}, s)
    assert "error" in r


# ── read_file / write_file ───────────────────────────────────────────


def test_read_file_unknown_path_errors() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("read_file", {"path": "missing.txt"}, s)
    assert "error" in r


def test_write_file_rejects_path_escape() -> None:
    s = init_state(seed=0)
    r = dispatch_tool(
        "write_file", {"path": "../etc/passwd", "content": "pwn"}, s
    )
    assert "error" in r
    assert s.files == {}


def test_write_file_rejects_absolute_path() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("write_file", {"path": "/etc/passwd", "content": "x"}, s)
    assert "error" in r


def test_write_file_rejects_non_string_content() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("write_file", {"path": "x.txt", "content": 42}, s)
    assert "error" in r


# ── send_message ─────────────────────────────────────────────────────


def test_send_message_missing_to_errors() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("send_message", {"body": "hi"}, s)
    assert "error" in r


def test_send_message_missing_body_errors() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("send_message", {"to": "alice"}, s)
    assert "error" in r


# ── Dispatcher / schemas / soft-fail conventions ─────────────────────


def test_dispatch_unknown_tool_returns_error() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("frobnicate", {}, s)
    assert "error" in r and "unknown" in r["error"].lower()


def test_dispatch_accepts_json_string_arguments() -> None:
    s = init_state(seed=0)
    r = dispatch_tool(
        "write_file",
        json.dumps({"path": "x.py", "content": "y = 2"}),
        s,
    )
    assert r["ok"] is True


def test_dispatch_invalid_json_string_arguments_errors() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("write_file", "{not json", s)
    assert "error" in r


def test_dispatch_none_arguments_treated_as_empty() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", None, s)
    assert "error" in r  # missing expression


def test_dispatch_non_dict_json_arguments_errors() -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", json.dumps([1, 2, 3]), s)
    assert "error" in r


def test_tool_schemas_cover_five_primitives() -> None:
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {"calculator", "web_search", "read_file", "write_file", "send_message"}


def test_tool_dispatch_keys_match_schema_names() -> None:
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert set(TOOL_DISPATCH.keys()) == schema_names


def test_schemas_for_returns_subset_in_requested_order() -> None:
    sub = schemas_for(["calculator", "send_message"])
    assert [s["function"]["name"] for s in sub] == ["calculator", "send_message"]


def test_schemas_for_drops_unknown_names() -> None:
    sub = schemas_for(["calculator", "nope"])
    assert [s["function"]["name"] for s in sub] == ["calculator"]


def test_every_schema_has_additional_properties_false() -> None:
    """PHASE_25_PLAN.md §6 — every schema must lock additional_properties."""
    for s in TOOL_SCHEMAS:
        params = s["function"]["parameters"]
        assert params.get("additionalProperties") is False, (
            f"{s['function']['name']!r} schema missing additionalProperties=false"
        )


def test_init_state_seeds_files_and_resets_lists() -> None:
    s = init_state(seed=99, initial_files={"a.txt": "x"})
    assert s.seed == 99
    assert s.files == {"a.txt": "x"}
    assert s.outbox == [] and s.calculator_history == [] and s.web_search_calls == []


def test_error_payloads_use_string_messages() -> None:
    """D5-B convention: every error payload is `{"error": <str>}`."""
    s = init_state(seed=0)
    cases = [
        dispatch_tool("calculator", {}, s),
        dispatch_tool("web_search", {"query": ""}, s),
        dispatch_tool("read_file", {"path": "missing"}, s),
        dispatch_tool("write_file", {"path": "/abs", "content": ""}, s),
        dispatch_tool("send_message", {"to": "", "body": ""}, s),
        dispatch_tool("frobnicate", {}, s),
    ]
    for r in cases:
        assert "error" in r and isinstance(r["error"], str)


@pytest.mark.parametrize("expr,expected", [
    ("1 + 2", 3.0),
    ("(3 * 5) + (7 - 2)", 20.0),
    ("2 ** 10", 1024.0),
    ("abs(-7)", 7.0),
    ("min(3, 5, 1)", 1.0),
    ("max(round(2.6), 2)", 3.0),
    ("pi * 2 - tau", pytest.approx(0.0, abs=1e-9)),
])
def test_calculator_assorted_expressions(expr: str, expected: float) -> None:
    s = init_state(seed=0)
    r = dispatch_tool("calculator", {"expression": expr}, s)
    assert "value" in r, r
    if isinstance(expected, float):
        assert r["value"] == pytest.approx(expected)
    else:
        assert r["value"] == expected
