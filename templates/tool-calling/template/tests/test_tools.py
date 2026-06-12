"""Tool-primitive contract tests for __ENV_ID__.

The env's reward kernel relies on the shared library at
``verifiable_labs_envs.tool_primitives``. The local re-export in
``__ENV_PY__.tools`` must hand back the same surface; the platform-
level primitive suite lives in the parent repo's
``tests/test_tool_primitives.py``.
"""
from __future__ import annotations

from __ENV_PY__.tools import (
    TOOL_DISPATCH,
    TOOL_SCHEMAS,
    canonical_action_hash,
    dispatch_tool,
    init_state,
    schemas_for,
)


def test_re_exports_match_platform_primitive_count():
    assert len(TOOL_SCHEMAS) == 5
    assert set(TOOL_DISPATCH) == {
        "calculator",
        "web_search",
        "read_file",
        "write_file",
        "send_message",
    }


def test_dispatch_calculator_smoke():
    s = init_state(seed=0)
    r = dispatch_tool("calculator", {"expression": "2 + 2"}, s)
    assert r == {"value": 4.0}


def test_schemas_for_returns_subset():
    sub = schemas_for(["calculator"])
    assert len(sub) == 1
    assert sub[0]["function"]["name"] == "calculator"


def test_canonical_action_hash_stable():
    h1 = canonical_action_hash([{"name": "calculator", "arguments": {"expression": "1+1"}}])
    h2 = canonical_action_hash([{"name": "calculator", "arguments": {"expression": "1+1"}}])
    assert h1 == h2
    assert len(h1) == 16
