# vlabs-prm-eval

Phase 30.D PRM eval harness CLI. Wraps the three eval surfaces from
[`PHASE_30_PLAN.md`](../../PHASE_30_PLAN.md) §9:

- **D6-A ProcessBench cross-check** — external benchmark for
  step-error detection accuracy. Pass criterion: ≥ 60 %.
- **D6-B Best-of-N reranking lift** — internal eval, PRM lift over
  Phase 29 distilled RM baseline. Pass criterion: ≥ +5 pp on
  held-out math-algebra.
- **D9-C calibration eval** — per-step + aggregate empirical
  coverage. Pass criterion: aggregate within ±5 pp of target.

> **Status:** scaffolding only. The stub PRM returns deterministic
> per-step `0.5 ± 0.1`; real evaluation against trained weights
> arrives in **Phase 30.G** when the trained student is ready
> (gate inherited from 30.F).

## Commands

| Command            | Surface                    | Pass criterion           |
|--------------------|----------------------------|--------------------------|
| `eval-processbench`| D6-A ProcessBench accuracy | ≥ 60 % overall           |
| `bon-rerank`       | D6-B BoN lift vs Phase 29  | ≥ +5 pp                  |
| `calibration`      | D9-C aggregate coverage    | within ±5 pp of target   |
| `card`             | Combined card with all 3   | all three above          |
| `version`          | CLI version                | n/a                      |

## Quickstart

```bash
# Run the combined eval card with the stub PRM (deterministic)
vlabs-prm-eval card \
    --calib-set reports/process_reward/v0.0.1_calib.jsonl \
    --n-processbench 40 \
    --n-bon-problems 10 \
    --output reports/process_reward/eval_card.json

# Just ProcessBench
vlabs-prm-eval eval-processbench --n 40

# Just BoN lift vs Phase 29 baseline (stub for both PRM + RM)
vlabs-prm-eval bon-rerank --n 10 --n-per 4

# Just calibration
vlabs-prm-eval calibration --calib-set reports/process_reward/v0.0.1_calib.jsonl
```

D6-C RL training capability lift is **scaffolded only** (the actual
RL run is gated to Phase 30.G+).
