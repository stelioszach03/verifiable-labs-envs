# Paper

**Verifiable Labs: Conformal-calibrated reinforcement-learning
environments for scientific reasoning.** OpenReview preprint, draft
v1.0, 4 pages + appendix. Author: Stelios Zacharioudakis (NKUA).

## Abstract

We introduce a benchmark of ten physics-grounded inverse problems for
evaluating frontier large language models on scientific reasoning
tasks. Each environment combines a closed-form forward operator,
procedural problem generation, and a conformal-calibrated reward
function that explicitly scores the model's stated uncertainty
against ground-truth coverage. Across five frontier models we find
substantial cross-environment capability spread (0.18–0.55 mean
reward), measurable miscalibration (empirical coverage 40-95 % vs
target 90 %), and informative multi-turn dynamics on tasks where the
forward operator can be queried.

## Where to find it

- **PDF.** [`paper/main.pdf`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/paper/main.pdf)
  in the repo.
- **Source.** [`paper/`](https://github.com/stelioszach03/verifiable-labs-envs/tree/main/paper) —
  LaTeX, all figure-generation code, `Makefile` for clean rebuilds.
- **OpenReview.** Submitted; venue and ID pending. Update this page
  on acceptance.

## Reproducing the paper's numbers

Every figure and table is reproducible from the repo. The paper's
methodology section names the exact CSV file behind each result:

| paper artifact | source CSV |
|---|---|
| Table 1 (single-turn benchmark) | `results/complete_matrix_single_turn.csv` |
| Table 2 (multi-turn benchmark) | `results/complete_matrix_multi_turn.csv` |
| Figure 2 (env correlation heatmap) | `results/env_correlation_matrix.csv` |
| Figure 3 (calibration validation) | `results/coverage_validation_n200.csv` |
| Figure 4 (failure taxonomy) | `results/failure_taxonomy.csv` |

To regenerate from scratch:

```bash
# Re-run the benchmark (~$5 OpenRouter at current pricing).
python benchmarks/run_v2_benchmark.py

# Rebuild the paper.
cd paper && make
```

## Related Verifiable Labs documents

- [Methodology (long form)](../METHODOLOGY.md) — the platform's
  internal methodology document; covers calibration, regeneration,
  and reward-function design in more detail than the paper.
- [Conformal background](../conformal.md) — references on split-
  conformal prediction, the technique behind the calibrated reward.
- [Contamination check](../CONTAMINATION.md) — empirical audit that
  the procedural-regeneration claim holds in practice.

## Citation

```bibtex
@misc{zacharioudakis2026verifiable,
  title  = {Verifiable Labs: Conformal-calibrated reinforcement-learning
            environments for scientific reasoning},
  author = {Zacharioudakis, Stelios},
  year   = {2026},
  note   = {OpenReview preprint draft v1.0},
  url    = {https://github.com/stelioszach03/verifiable-labs-envs/blob/main/paper/main.pdf}
}
```
