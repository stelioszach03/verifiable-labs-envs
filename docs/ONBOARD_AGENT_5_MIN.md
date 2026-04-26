# Onboard your agent in 5 minutes

End goal: get **your** agent scored on a Verifiable Labs env, with a
JSONL trace + Markdown report you can show to a teammate or paste into
a CI workflow. Estimated time: 5 minutes if you already have a
working solver. No paid API keys required for the smoke run.

## Prerequisites

```bash
pip install -e ".[dev]"     # this repo
verifiable envs              # confirm the install worked
```

## Step 1 — pick the agent shape that fits

The CLI's `--agent` flag accepts three forms. Pick whichever your
existing code is closest to.

| your code looks like… | use this |
|---|---|
| a Python function that takes a problem dict and returns a prediction dict | **Python file** |
| an executable in any language that reads JSON on stdin and writes JSON on stdout | **subprocess** |
| an HTTP endpoint that speaks the OpenAI chat-completions schema | **OpenAI-compatible** |

## Step 2A — Python file

Create a single file (anywhere):

```python
# my_agent.py
from typing import Any

AGENT_NAME = "my-agent"  # optional; defaults to the filename stem

def solve(observation: dict[str, Any]) -> dict[str, Any]:
    """Verifiable Labs calls this once per episode.

    `observation` is a dict with these keys:
      - "env_id"        str           — env id
      - "seed"          int           — episode seed (deterministic)
      - "env_kwargs"    dict          — kwargs passed to load_environment
      - "system_prompt" str           — the env's recommended system prompt
      - "prompt_text"   str           — the human-readable problem statement
      - "inputs"        dict          — structured problem inputs
                                        (e.g. n, k, mask, y for sparse-Fourier)

    Return a dict matching the env's prediction schema. For
    sparse-Fourier-style envs:
        {"support_idx": [k sorted ints in [0, n)],
         "support_amp_x1000": [k signed ints, same order]}
    For image envs:
        {"image_x255": [H*W ints in [0, 255]],
         "uncertainty_x255": [H*W ints]}
    """
    inputs = observation["inputs"]
    k = inputs["k"]
    n = inputs["n"]
    # Replace this with your real model call.
    return {
        "support_idx": list(range(k)),
        "support_amp_x1000": [0] * k,
    }
```

Run it:

```bash
verifiable run --env sparse-fourier-recovery --agent my_agent.py \
    --n 5 --out runs/me.jsonl
verifiable report --run runs/me.jsonl --out reports/me.md
```

Done. The Markdown report at `reports/me.md` has 12 sections (mean
reward, parse-fail rate, per-component breakdown, gap to classical,
best/worst episodes, and recommendations).

## Step 2B — Subprocess

If your agent isn't Python, write a thin wrapper that:

1. reads one JSON object from stdin (the observation)
2. writes one JSON object to stdout (the prediction)
3. exits 0 on success

Example bash wrapper around a hypothetical `./my_solver` binary:

```bash
#!/usr/bin/env bash
# my_agent.sh
input=$(cat)
echo "$input" | ./my_solver --json
```

Run it:

```bash
verifiable run --env sparse-fourier-recovery --agent "cmd:./my_agent.sh" \
    --n 5 --out runs/me.jsonl
```

The CLI handles timeouts (default 60 s per episode) and prints stderr
on non-zero exit codes. See [`src/verifiable_labs_envs/agents.py::SubprocessAgent`](../src/verifiable_labs_envs/agents.py)
for the protocol details.

## Step 2C — OpenAI-compatible HTTP

If your agent is a chat-completions endpoint (OpenAI, OpenRouter,
local vLLM, llama.cpp, Anthropic via gateway, etc.), use the
`openai:<model>` shortcut:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1     # optional
export VL_AGENT_MODEL=anthropic/claude-haiku-4.5        # optional

verifiable run --env sparse-fourier-recovery \
    --agent "openai:$VL_AGENT_MODEL" \
    --n 5 --out runs/llm.jsonl
```

If `OPENAI_API_KEY` is unset the agent falls back to a deterministic
fake response — useful for CI smoke tests where you don't want to
spend money.

## Step 3 — interpret the report

Read `reports/me.md`. Three sections matter most:

- **Reward distribution** — mean, std, min/max. A mean below the
  classical baseline (~0.55 for sparse-Fourier) is a real capability
  gap on this env, not a prompt-engineering issue.
- **Failure modes** — parse-failure rate >5 % usually means your
  output schema is wrong. Inspect the per-trace `metadata.parse_error`
  field in the JSONL.
- **Recommended next actions** — auto-generated based on your
  metrics.

## Step 4 — gate CI on this

Copy [`.github/workflows/verifiable-eval-example.yml`](../.github/workflows/verifiable-eval-example.yml)
into your own repo and replace the agent path. The workflow runs in
~30 s on every PR and uploads the JSONL + Markdown as artefacts you
can download from the Actions tab.

## Common patterns

### "I want to compare two of my agents"

```bash
verifiable run --env <id> --agent v1.py --n 30 --out runs/v1.jsonl
verifiable run --env <id> --agent v2.py --n 30 --out runs/v2.jsonl
verifiable compare --runs runs/v1.jsonl runs/v2.jsonl
```

The compare table shows mean reward, std, parse-fail %, gap to
classical baseline, and latency for each run.

### "I want to compare my agent to the classical baseline"

Two ways:

```bash
# 1. Use the bundled simple_baseline_agent — runs the env's classical
#    solver through the same pipeline:
verifiable run --env <id> --agent examples/agents/simple_baseline_agent.py \
    --n 30 --out runs/baseline.jsonl

# 2. Or, easier: pass --with-baseline to your agent's run. Each trace
#    will then carry classical_baseline_reward + gap_to_classical.
verifiable run --env <id> --agent my_agent.py --n 30 \
    --out runs/me.jsonl --with-baseline
```

### "I want to debug a single failed episode"

The JSONL traces include `metadata.parse_error` for parse-failed
episodes. Grep:

```bash
jq 'select(.parse_success == false) | {seed, metadata}' runs/me.jsonl
```

## See also

- [`docs/api-reference/cli.md`](api-reference/cli.md) — full CLI surface
- [`docs/CUSTOM_ENVIRONMENTS.md`](CUSTOM_ENVIRONMENTS.md) — write your own env
