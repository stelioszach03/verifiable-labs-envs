---
title: "Verifiable Labs Compliance Report — ${model}"
author: "Verifiable Labs Hosted Evaluation Platform v0.1.0-alpha"
date: "${date}"
geometry: "margin=2.5cm"
fontsize: 11pt
---

# Verifiable Labs Compliance Report

**Model under evaluation.** `${model}`
**Evaluation date.** ${date}
**Platform version.** v0.1.0-alpha
**Source data.** `${benchmark_csv_basename}`
**Scope.** ${n_envs} environments × ${n_seeds_total} total episodes

> This report is generated automatically from a benchmark CSV produced by
> the Verifiable Labs evaluation platform. It is intended as a
> self-service template — the platform does not assert legal compliance
> with any specific regulatory framework (NIST AI RMF, EU AI Act,
> ISO 42001) on the user's behalf. Consult your own counsel for legal
> attestation; this document evidences *empirical* model behaviour on a
> verifiable, conformal-calibrated benchmark.

---

## 1. Executive Summary

`${model}` was evaluated on **${n_envs}** scientific-reasoning environments
spanning ${domains}. Across **${n_seeds_total}** total episodes
the model achieved a mean reward of **${mean_reward}**
(parse-fail rate **${parse_fail_rate_pct} %**). Empirical conformal
coverage on calibrated envs was **${coverage_pct} %**, against a target
of **${target_coverage_pct} %**.

**Headline findings.**

${headline_findings}

**Recommended next step.** ${recommended_next_step}

---

## 2. Methodology

The evaluation platform implements **conformal-calibrated rewards** on
inverse-problem environments. For each (env, seed) the platform:

1. Procedurally regenerates a fresh problem instance from the seed, so
   the model has not seen this exact problem in training.
2. Sends a structured prompt to the model under test via OpenRouter
   (single-turn or multi-turn dialogue depending on the env).
3. Parses the JSON response into a typed `Prediction` dataclass.
4. Scores the prediction against ground truth using a per-env reward
   that includes:
   - a **point-estimate** term (NMSE / SSIM / chamfer / problem-specific)
   - a **support-recovery** term (where applicable)
   - a **conformal-coverage** term — the model is asked to provide
     uncertainty bounds; we score whether the truth falls inside.

Calibration: per-env conformal quantiles are computed offline on a
held-back calibration pool (typically 200-500 instances) at target
coverage `1 - α`, with `α = ${alpha}`.

**Reproducibility.** Every reward in this report is reproducible from
the source CSV (`${benchmark_csv_basename}`) and the env code at the
commit recorded in the benchmark metadata. The benchmark itself is
re-runnable end-to-end via `python benchmarks/run_v2_benchmark.py
--model ${model}`.

---

## 3. Capability Assessment

### 3.1 Per-environment reward distribution

${per_env_table}

### 3.2 Aggregate distribution

- **Mean reward:** ${mean_reward}
- **Median reward:** ${median_reward}
- **Standard deviation:** ${std_reward}
- **Min / Max:** ${min_reward} / ${max_reward}
- **Episodes scored:** ${n_seeds_total}

A higher reward indicates closer alignment with the ground-truth
solution under the env's specific reward function. By construction,
rewards are bounded in `[0, 1]`; values above `0.7` correspond to
solutions that are within engineering tolerances of the analytic
optimum, while values below `0.3` typically indicate a structural
misunderstanding of the problem (wrong support, wrong forward operator,
wrong prior scale).

---

## 4. Failure Modes

### 4.1 Parse failures

**${parse_fail_count}** of ${n_seeds_total} episodes
(**${parse_fail_rate_pct} %**) failed to produce a parseable JSON
prediction. Parse failures count as `reward = 0` in the aggregate.

Common causes the platform records:

- markdown code-fence wrapping (model emits ```json … ``` despite
  schema rejection of fences)
- prose preamble or postamble
- incorrect array length on the support indices or amplitudes
- out-of-range support indices or duplicate entries

A parse-fail rate above 5 % suggests the model needs prompt
engineering — not necessarily a capability gap; it may be a
formatting brittleness.

### 4.2 Low-reward environments

${low_reward_envs_table}

These environments warrant manual inspection of representative
episodes (the per-seed CSV preserves the full prompt + response).

---

## 5. Calibration

The platform reports **empirical conformal coverage** — the fraction
of episodes where the model's stated uncertainty interval contained
the ground truth — for envs that score uncertainty (≈ all envs in the
v0.1 set).

| metric | observed | target | notes |
|---|---|---|---|
| coverage | ${coverage_pct} % | ${target_coverage_pct} % | conformal split-quantile, α = ${alpha} |
| over-coverage | ${over_coverage_count} envs | — | model claims more uncertainty than needed |
| under-coverage | ${under_coverage_count} envs | — | model is over-confident |

**Reading.** Within ±5 percentage points of target is "well-calibrated"
for v0.1; the alpha gate flags >10 pp deviation either way. Severe
under-coverage (model claims certainty it doesn't have) is the most
operationally dangerous failure mode and should block deployment in
safety-relevant settings.

---

## 6. Recommendations

${recommendations}

---

## 7. Appendix

### 7.1 Environment list

The following ${n_envs} environments contributed to this report:

${env_list}

Each environment is documented under `docs/environments/<env_id>.md`
in the [verifiable-labs-envs](https://github.com/stelioszach03/verifiable-labs-envs)
repository.

### 7.2 Source data

- Per-episode CSV: `${benchmark_csv_basename}`
- Benchmark commit: see CSV metadata column
- Platform version: v0.1.0-alpha
- Generated: ${date_iso}

### 7.3 Limitations of this report

- v0.1 envs are inverse-problem-shaped; this report does not cover
  conversational, agentic, or open-ended generation tasks.
- Sample size per env is **${seeds_per_env}** seeds; statistical
  precision on a single env is bounded by `±0.1 / √n`.
- The platform asserts no claim about legal compliance with
  AI-governance frameworks; this is an empirical capability report,
  not a regulatory attestation.
