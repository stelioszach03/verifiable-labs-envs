# Tutorial: compliance reports

End goal: produce a self-service compliance / capability report for a
single model, suitable for internal review or as an exhibit attached
to an AI-governance framework deliverable. Estimated time: 5 minutes
once you have a benchmark CSV.

## Prerequisites

- A benchmark CSV with at least the columns
  `env, model, seed, turn, reward, components, parse_ok`. The
  platform's `benchmarks/run_v2_benchmark.py` produces exactly this
  shape; the v0.1 paper-final benchmarks are at
  `results/complete_matrix_single_turn.csv` and
  `results/complete_matrix_multi_turn.csv`.
- `pandoc` on your `$PATH` (preferred) or
  `pip install weasyprint markdown` (fallback) for PDF rendering.

## Run

```bash
python scripts/generate_report.py \
    --benchmark-csv results/complete_matrix_single_turn.csv \
    --model anthropic/claude-haiku-4.5 \
    --output haiku-compliance.md \
    --pdf haiku-compliance.pdf
```

Output:

```
Wrote haiku-compliance.md
Wrote haiku-compliance.pdf
```

The PDF is a 7-section typeset report; the Markdown is the same
content for further editing or hosting.

## What's in the report

| § | Section | What it says |
|---|---|---|
| 1 | Executive Summary | model, scope, mean reward, parse-fail rate, coverage; bullet-point findings |
| 2 | Methodology | how the platform generates and scores instances (boilerplate) |
| 3 | Capability Assessment | per-env table sorted by mean reward |
| 4 | Failure Modes | parse-fail count + envs scoring below 0.30 |
| 5 | Calibration | empirical conformal coverage vs target |
| 6 | Recommendations | auto-generated based on the metrics (high parse-fail, miscalibration, weak envs) |
| 7 | Appendix | env list, source-data hash, limitations disclaimer |

## What the report does NOT claim

- **Regulatory compliance.** The platform is an empirical capability
  evaluator. Legal attestation against NIST AI RMF, EU AI Act, ISO
  42001, etc. belongs to your compliance team. The report's
  introduction states this explicitly.
- **Generalisation beyond the CSV's envs.** v0.1 envs are
  inverse-problem-shaped. Conversational, agentic, and open-ended
  generation tasks need different evaluations.
- **Production-readiness.** v0.1.0-alpha. Treat the report as a
  baseline for internal review.

## Customising the template

The template at `templates/compliance-report/report_template.md` uses
`string.Template` (`$var` / `${var}`) substitution from the standard
library. To extend:

1. Add a new `${my_var}` placeholder in the template.
2. Open `scripts/generate_report.py::render_report` and add the
   corresponding key to the `subs` dict.
3. Re-run; the new content appears in the rendered output.

We deliberately avoid Jinja2 / mustache to keep the template
dependency surface zero.

## Running on the published CSV

The repository ships two canonical benchmark CSVs:

```bash
# Single-turn benchmark — 5 models × 5 envs × 3 seeds = 75 rows.
python scripts/generate_report.py \
    --benchmark-csv results/complete_matrix_single_turn.csv \
    --model openai/gpt-5.4 \
    --output gpt54-single.md --pdf gpt54-single.pdf

# Multi-turn benchmark — 5 models × 3 multi-turn envs × 3 seeds.
python scripts/generate_report.py \
    --benchmark-csv results/complete_matrix_multi_turn.csv \
    --model anthropic/claude-opus-4.7 \
    --output opus47-multi.md --pdf opus47-multi.pdf
```

The `templates/compliance-report/example_report.{md,pdf}` example was
generated this way for `anthropic/claude-opus-4.7` — open it to see
the rendered output before running on your own data.

## When to regenerate

Each time you re-run the benchmark (e.g. when a new model becomes
available, when an env's reward function changes, when calibration is
refreshed), regenerate the reports. The report carries the source
CSV's filename and the generation date in the appendix so reviewers
know which benchmark a given report was scored on.

## See also

- [API reference → CLI](../api-reference/cli.md) — full CLI options
- [`templates/compliance-report/README.md`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/templates/compliance-report/README.md) —
  template internals
