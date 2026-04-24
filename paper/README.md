# Preprint source — Conformal-Calibrated Rewards for Scientific RLVR

LaTeX source and figure-generation scripts for the 4-page OpenReview
preprint:

> **Conformal-Calibrated Rewards for Scientific RLVR: Procedural
> Regeneration Against Benchmark Contamination**
> Stelios Zacharioudakis (National and Kapodistrian University of Athens), 2026

## Files

```
paper/
├── main.tex                          Main LaTeX source (NeurIPS 2023 style)
├── references.bib                    22 verified references
├── neurips_2023.sty                  NeurIPS 2023 style (upstream)
├── figures/
│   ├── fig1_heatmap.pdf              6×5 model×env reward matrix
│   ├── fig2_gap.pdf                  Classical vs LLM 5-env mean
│   ├── fig3_multiturn.pdf            Multi-turn delta per model (appendix)
│   ├── fig4_coverage.pdf             Conformal coverage at N=100/env
│   └── fig5_tools.pdf                Primitive tool-use vs classical vs v0.1 oracle (appendix)
├── scripts/
│   └── generate_figures.py           Reads ../results/*.csv → figures/*.pdf
└── openreview_submission/            Ready-to-upload bundle
    ├── abstract.txt                  189-word abstract
    ├── keywords.txt                  OpenReview keyword list
    └── tldr.txt                      80-char TL;DR
```

The PDF is built as `paper/main.pdf` (6 pages: 4 main + 1 refs + 1 appendix).

## Build

Requires `pdflatex` and `bibtex` (any recent TeX Live).

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
open main.pdf
```

## Regenerate figures

The figure script reads the following CSVs from `../results/` (relative
to `paper/`). The script is deterministic up to matplotlib rendering.

```bash
cd paper
../.venv/bin/python scripts/generate_figures.py
```

Inputs consumed:
- `results/llm_benchmark_v2.csv` — Sprint-1 v2 benchmark (6 envs × 4 models).
- `results/meta_benchmark_v3.csv` — Sprint-giga meta-benchmark (5 envs × 3 cheap models × 2 seeds).
- `results/phase_retrieval_v1_benchmark.csv` — Task-1 phase retrieval bench.
- `results/mri_knee_v1_benchmark.csv` — Task-2 MRI knee bench.
- `results/opus_nano_fill_v2.csv` — Paper-prep Opus 4.7 + GPT-5.4-nano fill-in (2 models × 6 envs × 2 seeds, $0.94 spend).
- `results/coverage_validation.csv` — 10 envs × 100 seeds empirical coverage.
- `results/stat_tests.csv` — 43 paired-bootstrap comparisons.

## Claims ↔ data trace

| claim in paper | source |
|---|---|
| 10 envs across 5 domains | `src/verifiable_labs_envs/envs/`, registry in `__init__.py` |
| 43 paired comparisons, 29 at p<0.05 | `results/stat_tests.csv`, counted by `scripts/statistical_tests.py` |
| mean Δ = +0.213 (classical − LLM) | `scripts/statistical_tests.py` stdout |
| classical 5-env mean 0.630 | `paper/scripts/generate_figures.py fig2_gap stdout` |
| Opus 4.7 5-env mean 0.574 | same |
| coverage 0.901 ± 0.016 | `scripts/coverage_validation.py` stdout + `results/coverage_validation.csv` |
| v0.3 primitive tools 0.40±0.01 | `results/llm_benchmark_tools_v2.csv` |
| v0.1 oracle artefact 0.858 | `results/sparse_fourier_reconciliation.md` |
| multi-turn deltas per model | `scripts/statistical_tests.py` (internal _paired_delta helper) |

## AI-assistance

See the Acknowledgments section in `main.tex`. Briefly: code generation
and drafting help came from Claude (Anthropic) and Codex (OpenAI); all
scientific claims, experimental design, baseline selection, statistical
analysis, and conclusions were developed and verified by the author;
every numerical result in the paper is computed from CSV outputs of
model-API calls without AI intermediation.

## Repository

- Code + envs: https://github.com/stelioszach03/verifiable-labs-envs
- Environments Hub: https://app.primeintellect.ai/dashboard/environments/stelioszach
- Leaderboard Space: https://huggingface.co/spaces/stelioszach03/scientific-rl-benchmark
