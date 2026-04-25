# CLI reference

The repository exposes a few CLI entrypoints under `scripts/`. They
are not packaged as console scripts in v0.1; invoke with
`python scripts/<name>.py`. The naming convention is descriptive
verbs.

## `scripts/create_env.py`

Scaffold a new inverse-problem env from `templates/inverse-problem/`.

```bash
python scripts/create_env.py <env-id> --domain "<domain>" \
    [--target environments/<env-id>] [--force]
```

- `<env-id>` — kebab-case slug (e.g. `seismic-fwi`). Validator rejects
  underscores, uppercase, and consecutive hyphens.
- `--domain` — short label written into the env's `__init__.py`
  (`DOMAIN = "..."`) and README.
- `--target` — output directory; defaults to `environments/<env-id>`.
- `--force` — overwrite the target if it exists.

Walks the template tree, substitutes `__ENV_ID__`, `__ENV_PY__`,
`__ENV_CLASS__`, `__DOMAIN__`, `__DOMAIN_TAG__` placeholders, and
renames the placeholder package directory.

Tutorial: [Creating a custom env](../tutorials/creating-custom-env.md).

## `scripts/validate_env.py`

Run the four-check validator on an env package.

```bash
python scripts/validate_env.py <path-to-env> [options]
```

- `[1/4]` `pytest` on the env's `tests/` — must pass or skip cleanly.
- `[2/4]` Calibration check: empirical coverage on 50 fresh seeds must
  be within ±0.05 of `1 - α`. Skip with `--skip-calibration`.
- `[3/4]` Procedural-regeneration: `EFFECTIVE_INSTANCES > 1e15`.
- `[4/4]` Adapter compatibility:
  `verifiable_labs_envs.load_environment(<env_id>)` round-trip plus a
  `generate_instance(0)` import-time smoke. Skip with
  `--skip-adapter-check`.

Exits non-zero on any failure; prints a per-check summary.

## `scripts/generate_report.py`

Render a Verifiable Labs compliance report from a benchmark CSV.

```bash
python scripts/generate_report.py \
    --benchmark-csv results/complete_matrix_single_turn.csv \
    --model anthropic/claude-haiku-4.5 \
    --output report.md \
    [--pdf report.pdf] \
    [--target-coverage 0.90] [--alpha 0.10]
```

Output: a 7-section Markdown report (Executive Summary, Methodology,
Capability Assessment, Failure Modes, Calibration, Recommendations,
Appendix). With `--pdf` the script invokes `pandoc` (preferred) or
`weasyprint` (fallback). Tutorial:
[Compliance reports](../tutorials/compliance-reports.md).

## `notebooks/training_proof_run.py`

Driver for the training-proof experiment.

```bash
python notebooks/training_proof_run.py \
    [--smoke] [--cap-usd 1.50] [--model anthropic/claude-haiku-4.5]
```

`--smoke` runs a 14-episode tournament (~$0.10) for CI; the default
runs the full 30+5+30-seed experiment (~$0.60). The hard `--cap-usd`
makes the script abort if the cumulative OpenRouter spend exceeds it.
Writes `results/training_proof.csv` and `_summary.json`.

Tutorial: [Training with envs](../tutorials/training-with-envs.md).

## `benchmarks/run_v2_benchmark.py`

Re-run the canonical paper-final benchmark. **This is expensive.**

```bash
python benchmarks/run_v2_benchmark.py --models <model-list> --seeds <count>
```

Reproduces the `results/complete_matrix_*.csv` files. Used to refresh
the leaderboard. Read [`benchmarks/run_v2_benchmark.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/benchmarks/run_v2_benchmark.py)
before running — the defaults exercise all 10 envs across 5 models
and consume ~$5 on OpenRouter at current pricing.

## v0.2 packaging plan

In v0.2 the most-used scripts (`create_env`, `validate_env`,
`generate_report`) will move to console-script entry points so users
can run `vl-create-env <id>` instead of `python scripts/create_env.py <id>`.
Tracked in [Roadmap](../company/roadmap.md).
