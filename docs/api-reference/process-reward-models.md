# `/v1/process-reward-models` — Distilled PRM service (Phase 30)

> **Status:** v0.0.1-stub. The `/v1/process-reward-models/*` endpoint
> family ships in Phase 30.E of the verifiable-labs-envs roadmap.
> Until the trained student weights land in Phase 30.G, every score
> response carries `schema_version: "v0.1.0-stub"` and a deterministic
> per-step `reward ∈ [0.4, 0.6]` derived from a SHA-256 hash of the
> (prompt, step_index, step_text) tuple. The endpoint contract is
> locked NOW so frontend + SDK integrations can land in parallel.

A *process reward model* is a small open-weights student
(Qwen2.5-1.5B with a per-step regression head per
[PHASE_30_PLAN.md](../../PHASE_30_PLAN.md) §5 D2-A / D3-A) trained
to mimic per-step reward signals from procedurally-decomposable env
verifiers (D2-D), automated rollout propagation (D2-B), and a small
frontier-judgment slice (D2-C). The serving layer emits a per-step
calibrated reward sequence + per-step conformal intervals + an
aggregate score — the same Layer 1 moat that backs `/v1/score` and
`/v1/reward-models/*`, extended to step granularity.

> The split-conformal coverage guarantee behind our calibrated reward intervals is machine-verified in Lean 4 (sorry-free, standard axioms only). The Python implementation is property-tested against the formal specification. See [`formal/VerifiableLabsFormal/ConformalCoverage.lean`](../../formal/VerifiableLabsFormal/ConformalCoverage.lean).

## Endpoint surface

| Method | Path                                                          | Purpose                                          |
|--------|---------------------------------------------------------------|--------------------------------------------------|
| GET    | `/v1/process-reward-models`                                   | Paginated list (filter by `family`/`status`).   |
| GET    | `/v1/process-reward-models/{model_id}`                        | Single-model detail + eval card summary.        |
| POST   | `/v1/process-reward-models/{model_id}/score`                  | Single-call scoring.                            |
| POST   | `/v1/process-reward-models/{model_id}/score/batch`            | Batch scoring (≤ 50 pairs/call).                |
| GET    | `/v1/process-reward-models/{model_id}/evals`                  | Full eval card (held-out env breakdown +        |
|        |                                                               | ProcessBench detail + BoN comparisons +         |
|        |                                                               | calibration trace).                              |

Auth: `X-Vlabs-Key` header (data-plane), same as
[`/v1/score`](score.md), [`/v1/reward-models`](reward-models.md), and
the rest of the data plane.

## `model_id` shape

`vlabs-prm-{family}-v{semver}` per locked D12-B versioning:

- `vlabs-prm-distilled-qwen-1-5b-v0.1.0` (Phase 30.G initial release)
- `vlabs-prm-distilled-qwen-1-5b-v0.1.1` (calibration tweak)
- `vlabs-prm-distilled-qwen-1-5b-v0.2.0` (D5 mix change)

Lifecycle states (`process_reward_models.status`) mirror Phase 29:

| Status      | Customer-visible? | Behaviour                                     |
|-------------|-------------------|-----------------------------------------------|
| `training`  | No (admin only)   | Hidden from list; 404 on detail / score.       |
| `available` | Yes               | Default-routable.                              |
| `deprecated`| Yes               | Still served; flagged in `status` field.       |
| `retired`   | No                | 404 on customer endpoints (admin can re-list). |

## D13-C shared backbone link

When the PRM was trained under the D13-B/C shared-backbone path
(see [`PHASE_30_PLAN.md`](../../PHASE_30_PLAN.md) §5 D13), the
`base_rm_id` field on the detail / list responses surfaces the
parent Phase 29 distilled outcome RM `model_id`:

```json
{
  "model_id": "vlabs-prm-distilled-qwen-1-5b-v0.1.0",
  "base_rm_id": "vlabs-reward-distilled-qwen-1-5b-v0.1.0",
  ...
}
```

`null` indicates the D13-A independent-serving path (v0.0.1 default).

## `POST /v1/process-reward-models/{model_id}/score`

Request body (server segments):

```json
{
  "prompt": "Solve: 2x + 3 = 11",
  "reasoning_trace": "Step 1: Subtract 3 from both sides.\nStep 2: 2x = 8.\nStep 3: Divide by 2.\nStep 4: x = 4.",
  "env_id": "math-algebra",
  "schema_version": "v0.1.0",
  "with_step_rationale": false
}
```

Alternative request (pre-segmented):

```json
{
  "prompt": "Solve: 2x + 3 = 11",
  "reasoning_trace": [
    "Subtract 3 from both sides.",
    "2x = 8.",
    "Divide by 2.",
    "x = 4."
  ],
  "schema_version": "v0.1.0"
}
```

Optional headers:

- `X-Vlabs-Cache: enable` — opt-in Redis cache (D10-B, default-off).
  SHA-256 hashes only; plaintext NEVER persists. Cache TTL 1 h.
- `X-Idempotency-Key: <client-key>` — recorded on the audit row.

Response (`200`):

```json
{
  "step_rewards": [0.92, 0.95, 0.91, 0.98],
  "step_confidence_intervals": [
    [0.84, 1.00], [0.87, 1.00], [0.83, 0.99], [0.90, 1.00]
  ],
  "aggregate_reward": 0.94,
  "aggregate_confidence_interval": [0.86, 1.00],
  "coverage_guarantee": 0.90,
  "step_count": 4,
  "model_id": "vlabs-prm-distilled-qwen-1-5b-v0.1.0",
  "schema_version": "v0.1.0",
  "cache_hit": false,
  "latency_ms": 612,
  "audit_id": "prr_a1b2c3d4e5f6789012345678901234ab",
  "segmentation_warning": null
}
```

Errors:

- `400 process_reward_invalid_trace` — empty prompt / trace, or
  segmentation produced 0 steps.
- `404 process_reward_model_not_found` — unknown `model_id`, OR row
  exists in `training` status (admin-only).
- `410 process_reward_model_retired` — model retired; SDKs should
  switch to a newer version.
- `413 process_reward_trace_too_long` — segmented step count exceeds
  the per-call max (default 32, configurable per `PHASE_30_PLAN.md`
  R15).

## `POST /v1/process-reward-models/{model_id}/score/batch`

Up to 50 (prompt, reasoning_trace) pairs per call (denser per-call
shape than Phase 29's 100-item cap because per-step traces produce
more output dimensions). Same headers as single-score. Each item
gets its own `audit_id` and `step_rewards` array. Idempotent on
`X-Idempotency-Key` — duplicate calls within the 24 h window
short-circuit to the prior response.

```json
{
  "items": [
    {"prompt": "...", "reasoning_trace": "...", "env_id": "math-algebra"},
    {"prompt": "...", "reasoning_trace": ["s1", "s2"]}
  ],
  "schema_version": "v0.1.0"
}
```

## `GET /v1/process-reward-models`

```json
{
  "items": [
    {
      "model_id": "vlabs-prm-distilled-qwen-1-5b-v0.1.0",
      "family": "distilled-qwen-1-5b",
      "version": "0.1.0",
      "status": "available",
      "base_rm_id": "vlabs-reward-distilled-qwen-1-5b-v0.1.0",
      "step_granularity": "per_step",
      "created_at": "2026-07-01T00:00:00Z",
      "eval_summary": {
        "processbench_overall": 0.62,
        "bon_lift_vs_phase29": 0.07,
        "aggregate_calibration_coverage": 0.91
      }
    }
  ],
  "total": 1,
  "limit": 25,
  "offset": 0
}
```

Query params: `limit` (default 25, max 200), `offset`, `family`,
`status`. Sorted `created_at DESC` (newest first).

## `GET /v1/process-reward-models/{model_id}/evals`

Full eval card — per-env step-eval breakdown + ProcessBench-per-subset
scores + BoN reranking comparisons (vs Phase 29 RM baseline) +
per-step + aggregate calibration trace. Inner dicts evolve with
each released model version (D12-B); SDKs parse by key name.

```json
{
  "model_id": "vlabs-prm-distilled-qwen-1-5b-v0.1.0",
  "eval_summary": {
    "processbench_overall": 0.62,
    "bon_lift_vs_phase29": 0.07,
    "aggregate_calibration_coverage": 0.91
  },
  "held_out_envs": {
    "long-context-synthesis": {"per_step_spearman": 0.78, "mae": 0.06},
    "sql-multiturn":          {"per_step_spearman": 0.74, "mae": 0.08},
    "code-mini-repo":         {"per_step_spearman": 0.76, "mae": 0.07}
  },
  "processbench": {"math": 0.62, "gsm8k": 0.71, "olympiadbench": 0.55},
  "bon": {
    "single_accuracy": 0.55,
    "prm_bon_accuracy": 0.66,
    "rm_bon_accuracy": 0.59,
    "prm_bon_lift_vs_rm": 0.07
  },
  "calibration": {
    "step_conformal_quantiles": {
      "range(0, 1)": 0.05,
      "range(1, 3)": 0.07,
      "range(3, 7)": 0.09,
      "range(7, 32)": 0.12
    },
    "aggregate_quantile": 0.087,
    "aggregate_drift": 0.011,
    "n_traces": 1000
  }
}
```

## Audit trail

Every score call writes a `process_reward_model_runs` row carrying:

- SHA-256 hashes of prompt + joined trace (NEVER plaintext).
- `step_count` + per-step rewards + per-step CIs (JSONB arrays).
- Aggregate reward + CI bounds + coverage guarantee.
- Cache hit flag, latency in ms, idempotency key (if supplied).

Customers can verify a row matches their own inputs by re-hashing
locally. Same GDPR-aligned posture as Phases 22 + 29.

## Pricing

Locked at:

- Per-call: $0.005-0.020 depending on step count.
- Batch (≥ 50 calls): $0.003/call.
- Tier-included: free 100/month, pro 10K, team 100K, enterprise
  per-contract. 10× lower per-tier than Phase 29 outcome RM since
  per-step traces are denser per-call.

Tier-cap enforcement lives in 30.G when the trained student is
billing-eligible; the 30.E stub server is free + uncounted.

## Latency budget

| Surface              | p50      | p95      |
|----------------------|----------|----------|
| Stub (30.E)          | < 100 ms | < 200 ms |
| Trained (30.G live)  | < 1 s    | < 4 s    |

## Use cases

1. **Best-of-N reranking** — score N candidate completions, pick the
   highest aggregate. Standard math/code RL pattern. PRM beats
   outcome RM here because per-step granularity catches mid-trace
   divergences that an outcome RM misses (the final answer might
   still be lucky).
2. **RL training reward signal** — train a policy with PRM as
   step-level reward instead of sparse outcome reward. Major
   capability lift in the literature (Lightman et al.,
   "Let's Verify Step by Step", 2023). Gated on D6-C measurement in
   30.G+.
3. **Trace-level audit** — extends Phase 17 `vlabs-audit` with
   per-step confidence scores so customers can debug *where* their
   model's reasoning collapsed. The optional `with_step_rationale`
   request flag enables frontier-judge backfill on borderline steps
   in 30.G+.
