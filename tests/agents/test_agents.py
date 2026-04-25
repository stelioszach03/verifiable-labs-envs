"""Tests for the agent adapter interface and the example agents."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from verifiable_labs_envs.agents import (
    OpenAICompatibleAgent,
    SubprocessAgent,
    load_agent,
    load_python_agent,
    load_subprocess_agent,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "agents"


# ── Python loader ───────────────────────────────────────────


def test_load_python_agent_zero(tmp_path):
    agent = load_python_agent(EXAMPLES / "zero_agent.py")
    assert agent.name == "zero"
    out = agent.solve({"inputs": {"k": 5, "n": 256}})
    assert out["support_idx"] == [0, 1, 2, 3, 4]
    assert out["support_amp_x1000"] == [0, 0, 0, 0, 0]


def test_load_python_agent_random_is_deterministic_per_observation():
    agent = load_python_agent(EXAMPLES / "random_agent.py")
    obs = {"inputs": {"k": 3, "n": 32}}
    a = agent.solve(obs)
    b = agent.solve(obs)
    assert a == b
    # But different seeds → different output.
    c = agent.solve({"inputs": {"k": 3, "n": 32, "extra": "x"}})
    assert c != a


def test_load_python_agent_zero_image_shape():
    agent = load_python_agent(EXAMPLES / "zero_agent.py")
    out = agent.solve({"inputs": {"h": 16, "w": 16}})
    assert len(out["image_x255"]) == 256
    assert all(v == 128 for v in out["image_x255"])
    assert all(v == 10 for v in out["uncertainty_x255"])


def test_load_python_agent_zero_unknown_shape_falls_back():
    agent = load_python_agent(EXAMPLES / "zero_agent.py")
    out = agent.solve({"inputs": {}})
    assert "answer_text" in out


def test_load_python_agent_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_python_agent(tmp_path / "nope.py")


def test_load_python_agent_no_solve_raises(tmp_path):
    bad = tmp_path / "agent_no_solve.py"
    bad.write_text("# no solve here\nx = 1\n")
    with pytest.raises(ValueError, match="solve"):
        load_python_agent(bad)


def test_load_python_agent_rejects_non_py(tmp_path):
    bad = tmp_path / "agent.txt"
    bad.write_text("solve = lambda obs: {}\n")
    with pytest.raises(ValueError, match=".py"):
        load_python_agent(bad)


# ── Subprocess loader ───────────────────────────────────────


def test_subprocess_agent_round_trip(tmp_path):
    """Subprocess agent: child reads JSON on stdin, writes JSON on stdout."""
    script = tmp_path / "echo_agent.py"
    script.write_text(
        "import json, sys\n"
        "obs = json.loads(sys.stdin.read())\n"
        "print(json.dumps({'echoed_k': obs['inputs']['k']}))\n"
    )
    import sys as _sys
    agent = load_subprocess_agent(f"{_sys.executable} {script}", name="echo")
    out = agent.solve({"inputs": {"k": 7}})
    assert out == {"echoed_k": 7}


def test_subprocess_agent_propagates_nonzero_exit(tmp_path):
    script = tmp_path / "fail_agent.py"
    script.write_text("import sys; sys.exit(2)\n")
    import sys as _sys
    agent = load_subprocess_agent(f"{_sys.executable} {script}")
    with pytest.raises(RuntimeError, match="exited 2"):
        agent.solve({"inputs": {"k": 1}})


def test_subprocess_agent_propagates_invalid_json(tmp_path):
    script = tmp_path / "bad_agent.py"
    script.write_text("print('not json at all')\n")
    import sys as _sys
    agent = load_subprocess_agent(f"{_sys.executable} {script}")
    with pytest.raises(ValueError, match="not emit valid JSON"):
        agent.solve({"inputs": {"k": 1}})


def test_subprocess_agent_timeout(tmp_path):
    script = tmp_path / "slow_agent.py"
    script.write_text("import time; time.sleep(5)\n")
    import sys as _sys
    agent = load_subprocess_agent(f"{_sys.executable} {script}", timeout_s=0.5)
    with pytest.raises(TimeoutError):
        agent.solve({"inputs": {"k": 1}})


def test_subprocess_agent_empty_command_rejected():
    with pytest.raises(ValueError, match="empty"):
        load_subprocess_agent([])


# ── OpenAI-compatible agent (no key path) ────────────────────


def test_openai_agent_falls_back_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = OpenAICompatibleAgent.from_env(model="test-model")
    out = agent.solve({"prompt_text": "hi", "system_prompt": "be helpful"})
    assert out["_fake"] is True
    assert out["_user_len"] == 2
    assert out["_system_len"] == 10


def test_openai_agent_reads_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-gateway.example/v1")
    agent = OpenAICompatibleAgent.from_env(model="test-model")
    assert agent.base_url == "https://my-gateway.example/v1"


# ── load_agent dispatcher ────────────────────────────────────


def test_load_agent_dispatches_to_python(tmp_path):
    agent = load_agent(str(EXAMPLES / "zero_agent.py"))
    assert agent.name == "zero"


def test_load_agent_dispatches_to_subprocess(tmp_path):
    script = tmp_path / "x.py"
    script.write_text("import json,sys; print(json.dumps({}))\n")
    agent = load_agent(f"cmd:{os.sys.executable} {script}")
    assert isinstance(agent, SubprocessAgent)


def test_load_agent_dispatches_to_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = load_agent("openai:my-model")
    assert agent.model == "my-model"


def test_load_agent_rejects_unknown_spec():
    with pytest.raises(ValueError, match="could not detect"):
        load_agent("totally-bogus")


# ── Smoke: openai_compatible_agent.py example file ──────────


def test_openai_compatible_agent_example_loads(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = load_python_agent(EXAMPLES / "openai_compatible_agent.py")
    out = agent.solve({"prompt_text": "hello", "system_prompt": "x"})
    assert out["_fake"] is True


# ── Smoke: simple_baseline_agent emits the sentinel ─────────


def test_simple_baseline_agent_returns_sentinel():
    """Returns the classical-baseline sentinel; CLI does the dispatch."""
    agent = load_python_agent(EXAMPLES / "simple_baseline_agent.py")
    out = agent.solve({"env_id": "sparse-fourier-recovery", "seed": 0})
    assert out == {"__classical_baseline__": True}
    assert agent.name == "simple-baseline"
