# Tutorial: training with envs

End goal: use the env's reward signal to drive a search over policy
variants (here, system-prompt variants) — the simplest reproducible
RLVR proxy. Estimated time: 15 minutes setup + 8-12 minutes runtime.
Cost: ~$0.60 on OpenRouter at current pricing.

## What this tutorial does NOT teach

- A full RL training loop (GRPO, PPO, DSPy compile). Those would all
  consume the same `env.run_rollout(...)` reward we use here, but the
  framework wiring is out of scope.
- How to train a model from scratch — we evaluate a frozen frontier
  LLM, not gradients.

The tutorial proves the env's reward signal is **dense enough** to
discriminate policy variants. That's the prerequisite for actual
training.

## Prerequisites

```bash
pip install -e ".[dev]"
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
```

You need ~$1 of OpenRouter credit; the script enforces a $1.50 hard
cap.

## The experiment

Tournament prompt search:

1. Score the env's default system prompt on 30 baseline seeds.
2. Score 4 candidate prompt variants on 5 validation seeds each.
3. Pick the best by mean reward.
4. Re-score default and best on 30 unseen held-out seeds.
5. Compute a paired-bootstrap 95 % CI on the per-seed delta.

The candidates are 4 hand-crafted variations of the default prompt
(terse / verbose-with-strategy / step-numbered / LLM-rewritten), all
reproducible from `notebooks/training_proof_lib.py::DEFAULT_CANDIDATES`.

## Running

```bash
# Smoke (~$0.10, 14 episodes, 90s wall):
python notebooks/training_proof_run.py --smoke --cap-usd 0.30

# Full run (~$0.60, 95 episodes, 8m15s wall):
python notebooks/training_proof_run.py --cap-usd 1.50
```

Or run interactively in `notebooks/training_proof.ipynb` — same code,
plus inline reward-distribution plots.

## Reading the output

The script prints a per-seed log and emits two artifacts:

- `results/training_proof.csv` — one row per seed × split (baseline /
  val per candidate / heldout-default / heldout-best).
- `results/training_proof_summary.json` — aggregate means, deltas,
  paired-bootstrap CI.

Example summary:

```json
{
  "baseline":  {"mean": 0.330, "std": 0.022, "parse_fail_rate": 0.0},
  "candidates": [
    {"name": "terse-baseline",        "mean": 0.331},
    {"name": "verbose-with-strategy", "mean": 0.340},
    {"name": "step-numbered",         "mean": 0.344},
    {"name": "rewritten-directive",   "mean": 0.334}
  ],
  "best": {"name": "step-numbered", "mean_on_val": 0.344},
  "heldout": {
    "default_mean": 0.333, "best_mean": 0.329,
    "delta": -0.0034, "ci_lo": -0.008, "ci_hi": -0.001,
    "n_bootstrap": 5000
  }
}
```

## Interpreting a negative delta

The reference run linked above produced a **negative** held-out delta
(`step-numbered` lost to default by 0.34 pp, CI strictly below zero).
This is informative, not a failure:

- The env's reward signal is **measurable** at sub-1 % effects with
  30 paired seeds. That's the prerequisite for training.
- The optimisation procedure (5-seed val pool, 4 candidates) was too
  noisy to reliably rank — the val winner overfit. A bigger val pool
  or different candidates may flip the result.
- We deliberately do **not** tune the candidate pool until the
  held-out delta is positive; that would be p-hacking.

## How to extend

Three honest ways to make this a real RLVR loop:

1. **Larger candidate pool** — generate 20+ candidates via an LLM
   rewriter, evaluate on a 30-seed val pool. Likely the cheapest path
   to a positive delta on this env.
2. **DSPy `BootstrapFewShot`** — uses the same reward signal but
   compiles a multi-shot prompt program. Requires a LiteLLM shim to
   talk to OpenRouter; the
   [Plan B note](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/notebooks/README.md#why-prompt-search-and-not-dspy)
   explains why we skipped it for v0.1.
3. **Real RL** — `env.run_rollout(solver, instance)` returns a per-
   step reward. Feed it into a PPO / GRPO loop with `transformers` +
   `trl`. The platform supplies the reward; the training framework is
   yours.

## See also

- [Concepts → Conformal rewards](../concepts/conformal-rewards.md) —
  what the reward signal actually measures.
- [`notebooks/README.md`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/notebooks/README.md) —
  the canonical reference for this experiment.
