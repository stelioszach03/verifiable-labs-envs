# Demo commands (verbatim)

Copy-pasteable commands for the 90-second recording. Each command is
verified to run from a clean clone after `pip install -e ".[dev]"`.

Use this exact order. The recording matches [`DEMO_SCRIPT_90_SEC.md`](DEMO_SCRIPT_90_SEC.md).

## Setup (do *before* hitting record)

```bash
git clone https://github.com/stelioszach03/verifiable-labs-envs.git
cd verifiable-labs-envs
python3.11 -m venv ~/.venvs/verifiable-labs && source ~/.venvs/verifiable-labs/bin/activate
pip install -e ".[dev]" --quiet
mkdir -p runs reports
```

(If you're on macOS in `~/Documents/`, the iCloud-Drive note in the
README applies — make sure the venv is *outside* iCloud-synced
storage.)

## Demo (record this part)

```bash
# 1. List the 10 envs.
verifiable envs

# 2. Run the zero agent (no API key, ~1 second).
verifiable run \
    --env sparse-fourier-recovery \
    --agent examples/agents/zero_agent.py \
    --n 3 \
    --out runs/demo.jsonl

# 3. Render the Markdown report.
verifiable report --run runs/demo.jsonl --out reports/demo.md

# 4. Open the report (or `cat reports/demo.md` for terminal-only).
cat reports/demo.md | head -40
```

## Bonus shots (optional, for longer videos)

### Compare two agents

```bash
verifiable run --env sparse-fourier-recovery \
    --agent examples/agents/random_agent.py \
    --n 3 --out runs/random.jsonl --quiet

verifiable compare --runs runs/demo.jsonl runs/random.jsonl
```

### Run against the classical baseline

```bash
verifiable run --env sparse-fourier-recovery \
    --agent examples/agents/simple_baseline_agent.py \
    --n 3 --out runs/baseline.jsonl

verifiable compare --runs runs/demo.jsonl runs/baseline.jsonl
```

### API health check (if you can show a separate terminal)

```bash
# In another terminal, with the API extras installed:
uvicorn verifiable_labs_api.app:app --port 8000 &
sleep 2
curl -s http://localhost:8000/v1/health | jq
# {"status":"ok","version":"0.1.0-alpha","uptime_s":2.1,"sessions_active":0}
```

### SDK snippet (if you switch to a Python REPL)

```python
from verifiable_labs import Client

with Client(base_url="http://localhost:8000") as c:
    print(c.health())
    envs = c.environments()
    print(f"{envs.count} envs, first 3: {[e.id for e in envs.environments[:3]]}")
```

### Custom env scaffold

```bash
verifiable init-env demo-env --domain "demo"
ls environments/demo-env/
verifiable validate-env environments/demo-env --skip-adapter-check
# Calibration check fails on the unfilled scaffold — expected.
```

### Training-signal demo

```bash
python examples/training_signal_demo.py --quick
cat results/training_signal_demo.md | head -30
```

## What to *not* show

- The full pytest run (60+ s; not cinematic).
- The OpenAI agent without an API key (the fake fallback isn't
  visually informative).
- The training-proof notebook full run (~8 min; cite results from
  `notebooks/README.md` instead).

## Troubleshooting on the day

| symptom | fix |
|---|---|
| `verifiable: command not found` | `source ~/.venvs/verifiable-labs/bin/activate` then `pip install -e .` again |
| `ModuleNotFoundError: verifiable_labs_envs` | iCloud Drive corrupted the .pth file; see README → "Install" → "macOS + iCloud Drive" |
| API doesn't start | port 8000 in use? change with `--port 8001` |
| Report has no components | the env's adapter parser failed; check `metadata.parse_error` in the JSONL |
