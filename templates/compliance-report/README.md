# Compliance report template

Markdown template + Python generator that turn a Verifiable Labs
benchmark CSV into a self-service capability report. Useful for:

- internal model approval gates
- third-party evaluation deliverables
- exhibits attached to AI-governance frameworks (NIST AI RMF, EU AI
  Act, ISO 42001) — note the platform itself does **not** assert legal
  compliance; consult counsel for attestation

## Files

- `report_template.md` — `string.Template` ($-syntax) skeleton with 7
  sections: Executive Summary, Methodology, Capability Assessment,
  Failure Modes, Calibration, Recommendations, Appendix.
- `example_report.md` / `example_report.pdf` — pre-rendered example
  for `anthropic/claude-opus-4.7` against the v0.1 single-turn benchmark.
- `../../scripts/generate_report.py` — fills the template and
  optionally renders to PDF.

## Quick start

```bash
# Markdown only:
python scripts/generate_report.py \
    --benchmark-csv results/complete_matrix_single_turn.csv \
    --model anthropic/claude-opus-4.7 \
    --output report.md

# Markdown + PDF (needs pandoc OR weasyprint):
python scripts/generate_report.py \
    --benchmark-csv results/complete_matrix_single_turn.csv \
    --model anthropic/claude-opus-4.7 \
    --output report.md \
    --pdf report.pdf
```

The script exits non-zero (and prints the available models) if the
model isn't present in the CSV.

## Inputs the generator expects

The CSV must have these columns (the platform's standard
`run_v2_benchmark.py` output has all of them):

| column | type | example |
|---|---|---|
| `env` | str | `sparse-fourier-recovery` |
| `model` | str | `anthropic/claude-haiku-4.5` |
| `seed` | int | `0` |
| `turn` | int | `1` (the script keeps the latest turn per `(env, seed)`) |
| `reward` | float | `0.351` |
| `components` | str | `nmse=0.135, support=0.200, conformal=0.900` |
| `parse_ok` | bool | `True` |

Other columns are ignored.

## Customising

The template uses `string.Template` (`$var` / `${var}`) substitution.
The generator's `subs` dict (in `render_report`) lists every variable;
extending the template is a matter of adding a new key + filling it in
the dict. We deliberately avoid Jinja2 / mustache to keep the
dependency surface zero (stdlib only — `string.Template`, `csv`,
`statistics`).

## PDF rendering chain

1. **pandoc** is preferred. With `tectonic` / `xelatex` / `pdflatex`
   on `$PATH`, pandoc produces a typeset PDF (this is what
   `example_report.pdf` was built with — pandoc 3.x, no LaTeX engine
   needed for the default theme).
2. **weasyprint** is the fallback (HTML → PDF via CSS). Install with
   `pip install weasyprint markdown`.

If neither is present and you pass `--pdf`, the generator exits with
a clear error.

## What the report does NOT claim

- Regulatory compliance with any AI-governance framework. The
  platform is an *empirical* capability evaluator; legal attestation
  belongs to the user's compliance team.
- Generalisation beyond the envs scored in the CSV. The v0.1 envs are
  inverse-problem-shaped; conversational, agentic, and open-ended
  generation tasks are out of scope.
- Production-readiness. v0.1.0-alpha; treat outputs as a baseline for
  internal review, not a sign-off.
