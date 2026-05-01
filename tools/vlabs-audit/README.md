# vlabs-audit

> Capability report generator. Run any frontier model against Verifiable Labs
> environments and produce a professional LaTeX/PDF report — usable as a sales
> artefact for cold outreach or a defensible artefact for funding /
> due-diligence conversations.

A sample anonymised report (Haiku 4.5 across three Verifiable Labs envs,
90 episodes, conformal-calibrated rewards, ~$0.43 spend) lives at
[`reports/sample_anonymized.pdf`](../../reports/sample_anonymized.pdf).

## Install

`vlabs-audit` is a sibling package to the SDK in this monorepo. From the
repo root:

```bash
pip install -e tools/vlabs-audit          # production deps only
pip install -e "tools/vlabs-audit[dev]"   # plus pytest, ruff
```

That installs the `vlabs-audit` console script. The audit pipeline shells
out to the SDK's `verifiable run`, so the SDK must also be installed
(`pip install -e packages/verifiable-labs`).

The LaTeX/PDF render step uses [`tectonic`](https://tectonic-typesetting.github.io/),
a static Rust binary with no full TeX Live dependency. Install it via:

```bash
# Linux / macOS — prebuilt static binary, no Rust toolchain needed.
curl --proto '=https' --tlsv1.2 -fsSL \
  https://drop-sh.fullyjustified.net | sh
sudo mv tectonic /usr/local/bin/
```

(or `cargo install tectonic`, or via Homebrew). The audit still runs without
tectonic — it just skips the LaTeX render with a clear message.

## API keys

Configure provider keys via environment variables. The SDK routes models
by prefix:

| prefix | provider | env var |
|---|---|---|
| `claude-*` | Anthropic direct | `ANTHROPIC_API_KEY` |
| `gpt-*`, `o1`, `o3`, `o4` | OpenAI direct | `OPENAI_API_KEY` |
| `gemini-*` | Google direct | `GOOGLE_API_KEY` |
| `<provider>/<model>` | OpenRouter | `OPENROUTER_API_KEY` |

OpenRouter routing is the cheapest path during development. Set the key
once:

```bash
read -rs -p "OPENROUTER_API_KEY: " key && \
  umask 077 && \
  printf 'export OPENROUTER_API_KEY=%q\n' "$key" > ~/.vlabs-env && \
  chmod 600 ~/.vlabs-env && unset key
```

then `source ~/.vlabs-env` before running `vlabs-audit`.

## Quick start

```bash
# Single env, 5 episodes, anonymised PDF report.
vlabs-audit audit \
  --model "anthropic/claude-haiku-4.5" \
  --envs sparse-fourier-recovery \
  --episodes 5 \
  --alpha 0.1 \
  --parallel 4 \
  --anonymize \
  --output report.pdf \
  --print-stats
```

The pipeline:

1. Schedules `episodes × len(envs)` rows in `~/.vlabs-audit/audits.db`
   (SQLite, resumable).
2. Drives a `ThreadPoolExecutor(parallel)` of `verifiable run` subprocesses.
3. Aggregates per-episode rewards into mean ± 95 % bootstrap CI, parse
   failure rate, format validity, and held-out coverage versus
   `1 − alpha`.
4. Renders four PDF figures (reward distribution per env, coverage
   calibration, quality breakdown, cost per correct answer).
5. Renders a five-section LaTeX report (executive summary, methodology,
   results, recommendations, appendix) and compiles it to PDF via
   tectonic.

`--anonymize` swaps the model id with `anonymize_label` (default
`"Frontier Model A"`) throughout the rendered PDF — cover, headers,
tables, citation block. The on-disk audit row keeps the real id for
reproducibility.

## Configuration

Either pass everything on the CLI, or write a YAML config and override
fields at the command line:

```yaml
# configs/sample.yaml
model: claude-haiku-4.5
envs:
  - sparse-fourier-recovery
  - phase-retrieval
  - super-resolution-div2k-x4
episodes: 30
alpha: 0.1
output: report.pdf
parallel: 4
seed_start: 1000
anonymize: false
anonymize_label: "Frontier Model A"
```

```bash
vlabs-audit audit --config configs/sample.yaml --episodes 50 --anonymize
```

CLI flags override YAML values; YAML overrides built-in defaults; passing
the same flag without a value falls back to the YAML/default.

### CLI flags

| flag | meaning |
|---|---|
| `--config FILE` | YAML config (any field below). |
| `--model NAME` | Model id (e.g. `anthropic/claude-haiku-4.5`). |
| `--envs A,B,C` | Comma-separated Verifiable Labs env ids. |
| `--episodes N` | Episodes per env (1–10 000). |
| `--alpha X` | Conformal miscoverage in `(0, 1)`. |
| `--output FILE.pdf` | PDF report path. |
| `--parallel N` | Worker count (1–16). |
| `--seed-start N` | Initial seed; subsequent episodes increment by 1. |
| `--anonymize` | Replace model id in the rendered PDF. |
| `--anonymize-labels "A,B,C"` | Custom anonymisation labels (CSV). |
| `--dry-run` | Print resolved config and exit. |
| `--print-stats` | Print the aggregate stats table after the run. |
| `--figures-dir DIR` | Render PDF figures into `DIR` (implies `--print-stats`). |
| `--resume aud_xxx` | Drain pending + crashed-running rows for an existing audit. |

### Resuming a crashed run

Audit rows live in SQLite (`~/.vlabs-audit/audits.db`). A crashed worker
leaves rows in `running`; `--resume` recovers them:

```bash
vlabs-audit audit --config configs/sample.yaml --resume aud_<id>
```

## Cost estimates

Real measurements from the 17.F sample run (`anthropic/claude-haiku-4.5`
via OpenRouter, 90 episodes across three envs):

| env | cost / episode |
|---|---|
| `sparse-fourier-recovery` | ~$0.0014 |
| `phase-retrieval` | ~$0.0036 |
| `super-resolution-div2k-x4` | ~$0.0119 |

For OpenAI GPT-4o-mini through OpenRouter, expect ~5× lower per episode.
Plan for $0.15 – $0.50 for a typical 90-episode three-env audit on a
small/cheap frontier model; an order of magnitude more for larger
models. The `--dry-run` mode shows the resolved config and total
episode count without spending; always sanity-check before launching.

## Troubleshooting

**`tectonic is not on PATH`** — install per the
[Install](#install) section, or omit `--output` to skip the LaTeX render
and inspect the JSONL traces directly.

**`verifiable run --env X --seed Y failed (exit 1)`** — check that the
SDK is installed (`which verifiable`) and that the relevant API-key env
var is set for the model's provider (see [API keys](#api-keys)).

**Parallel workers race on `~/.verifiable/runs/<env>_<model>_<ts>.jsonl`** —
shouldn't happen: each subprocess is sandboxed under its own temporary
`HOME` for exactly this reason. If you see it, file an issue with the
captured stdout/stderr.

**Cost shows $0.00 / "Cost data not available"** — the SDK writes
`Cost: —` in stdout when the agent didn't surface usage data (typically
free or local models). The audit still runs; the cost figure renders an
informational placeholder.

## Testing

```bash
cd tools/vlabs-audit
pytest         # 83 tests, ~30 s wall (no real network or tectonic invoked)
ruff check src tests
```

The audit suite is also wired into the repo-root pytest config; running
`pytest` from the monorepo root picks up these tests automatically.

## Roadmap

- v0.0.1 (current) — single-model anonymised report.
- v0.0.2 — multi-model side-by-side comparison; one audit row per model.
- v0.0.3 — pluggable nonconformity score per env; per-env target-coverage
  override.

## License

Apache-2.0 — see repo root [`LICENSE`](../../LICENSE).
