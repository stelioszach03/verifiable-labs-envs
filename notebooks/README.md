# notebooks/ — runnable demos

Sibling to `examples/` (single-file Python scripts). Notebooks live here
when the demo benefits from inline plots, narrative markdown, or
side-by-side LLM output.

## training_proof.ipynb — Tier-1 Task 4

End-to-end demo that the Verifiable Labs environments are usable as a
**reward signal for prompt-search optimisation**, the simplest reproducible
RLVR proxy.

### What it does

1. Loads `sparse-fourier-recovery-multiturn` (3-turn dialogue env, k-sparse
   Fourier recovery with conformal-calibrated reward).
2. Runs `anthropic/claude-haiku-4.5` baseline on 30 fresh seeds.
3. Evaluates 4 candidate system prompts on a 5-seed validation pool.
4. Picks the best by mean reward.
5. Held-out evaluation: default vs best on 30 unseen seeds with paired
   bootstrap 95 % CI on the per-seed delta.

### Reproducing

```bash
# 1. Install monorepo + the LLM solver bits.
pip install -e ".[dev]"
# 2. Add your OpenRouter key to .env (or export it).
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
# 3. Either run the script directly:
python notebooks/training_proof_run.py --cap-usd 1.50
# 4. Or open the notebook:
jupyter notebook notebooks/training_proof.ipynb
```

### Cost guard

The `BudgetCap` in `training_proof_lib.py` accumulates the `usd_cost`
field from every OpenRouter completion and **raises** when the cap is
hit. Default cap is `$1.50`. Use `--cap-usd 0.30` for a smoke run.

```bash
# Smoke (~$0.10, 14 episodes, 90s wall):
python notebooks/training_proof_run.py --smoke --cap-usd 0.30
```

### Outputs

- `results/training_proof.csv` — one row per seed × split (baseline / val
  per candidate / heldout-default / heldout-best).
- `results/training_proof_summary.json` — aggregate means, deltas, CI.
- (Optional) `notebooks/training_proof_export.html` — `nbconvert`-rendered
  notebook for inclusion in the docs site.

### Why prompt-search and not DSPy?

The original brief mentioned `dspy.BootstrapFewShot`. We deliberately
chose a plain prompt tournament instead because:

- DSPy's default OpenAI client doesn't talk to OpenRouter without a
  custom shim (LiteLLM or per-call HTTP override). The shim adds a
  dependency for a feature we don't strictly need.
- Tournament search is **transparent**: the candidate prompts are 4
  text strings in version control, the search is exhaustive, and the
  selection rule is `max(mean_reward, tie-break on parse-fail)`.
- The headline claim — "the env's reward signal can drive policy
  improvement" — is the same regardless of optimiser. Future versions
  can swap in DSPy / GRPO / PPO without touching the env.

### Limits

- 30 + 5 + 30 = 65 paired episodes per held-out comparison. Smaller-than-
  paper sample size; confidence intervals are wider.
- Single model (Claude Haiku 4.5). Other models may have different
  prompt sensitivities.
- `parse_ok` failures are recorded (`reward = 0`) rather than excluded;
  this is the same convention the v2 benchmark uses.

The full benchmark in `paper/` covers 5 models × 10 envs and is the
authoritative comparison; this notebook is a reproducibility demo.

### Reference run — 2026-04-25, claude-haiku-4.5

One execution of `python notebooks/training_proof_run.py --cap-usd 1.50`
on commit `7ee1684`+. Reproduces from `OPENROUTER_API_KEY` in `.env`.
Recorded in `results/training_proof.csv` and `_summary.json`.

| split | n | mean | notes |
|---|---|---|---|
| baseline (default prompt) | 30 | 0.3299 ± 0.022 | parse-fail = 0 % |
| val: terse-baseline | 5 | 0.3312 | parse-fail = 0 % |
| val: verbose-with-strategy | 5 | 0.3404 | parse-fail = 0 % |
| **val: step-numbered** | **5** | **0.3444** | best by val |
| val: rewritten-directive | 5 | 0.3340 | parse-fail = 0 % |
| held-out: default | 30 | 0.3325 | parse-fail = 0 % |
| held-out: step-numbered | 30 | 0.3291 | parse-fail = 0 % |
| Δ (best − default) | 30 paired | **−0.0034** | 95 % CI **[−0.0080, −0.0008]** |
| total spend | — | **$0.57** | of $1.50 cap, 8m15s wall |

**Honest reading.** The val-picked candidate (`step-numbered`) does
*worse* than default on the held-out pool by 0.34 percentage points,
with a paired-bootstrap CI strictly below zero. Two takeaways:

1. **The env's reward signal is informative at the sub-1 % effect
   scale.** A 95 % CI of width 0.7 % from 30 paired seeds means the
   reward is dense enough to drive optimisation when the candidate
   pool genuinely improves the policy. This is the headline claim
   for using these envs as RLVR signals.
2. **The optimisation procedure overfit.** A 5-seed val pool can't
   reliably rank prompts whose true means cluster within ~0.01.
   `step-numbered` won val by +0.014 over default — within noise —
   and lost the larger held-out comparison. The fix is mechanical
   (more val seeds, regularised selection), not a flaw in the env.

We deliberately do **not** tune the candidate pool until the held-out
delta is positive; that would be p-hacking. The notebook is a
demonstration that the env's reward is a *measurable* signal, not a
claim that prompt search beats the default prompt for this model.
