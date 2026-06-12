# Continuous capability monitoring

> Phase 28 — the first **recurring-revenue SaaS surface** in the
> verifiable-labs roadmap.

Continuous monitoring is the answer to *"is my model getting better,
worse, or noisier this week?"* — a temporal signal layered on top of
the env catalogue (Phase 21-27) and the per-call scoring API (Phase 22).

## What it does

Customer registers an OpenAI-compatible model endpoint with us:
URL + auth token + cadence (daily / weekly / monthly) + which envs
to audit on.

We then:

1. Fire a scheduled run at the chosen cadence.
2. Hit the customer endpoint once per `(env_id, seed)` pair in the
   monitor's configuration.
3. Score each completion through the existing
   [`/v1/score`](../api-reference/score.md) kernel.
4. Aggregate the per-episode rewards into a summary; compute a
   regression verdict against the *baseline* run (D6-A first
   successful run, customer-rebaselineable).
5. On `warning` / `regressed` verdicts, dispatch alerts to the
   configured channels (email + optional Slack webhook).
6. Persist the summary, verdict, and a downloadable PDF report at
   `monitor_runs.pdf_url`.

## Why it matters

The env catalogue is a *substrate* — calibrated, contamination-
resistant problem distributions. The per-call API is a *query
surface* — one prompt in, one calibrated reward out. Continuous
monitoring is a *temporal alarm* — the moat-aligned product
position where the conformal calibration gives a statistically
controlled false-positive rate by construction.

A regressed verdict is **not** "the average reward dropped by 5%."
It's "the model's empirical conformal coverage has drifted by more
than 10pp from its calibrated target, *and* a paired-sample
bootstrap confirms a statistically significant negative reward
delta against the baseline." That double-signal lets us alert on
real distribution shifts without tripping on natural variance.

## Architecture (TL;DR)

| Component             | Where                                                             |
|-----------------------|-------------------------------------------------------------------|
| Scheduler tick        | `vlabs_api.monitor_scheduler.scheduler_tick`                      |
| Worker pool           | `vlabs_api.monitor_worker.monitor_worker_loop`                    |
| Episode runner        | `vlabs_api.monitor_episode_runner.run_monitor_episodes`           |
| Regression verdict    | `vlabs_api.monitor_regression.compute_verdict`                    |
| Alert dispatch        | `vlabs_api.monitor_alerts.dispatch_monitor_alerts`                |
| PDF rendering         | `vlabs_api.monitor_pdf.render_monitor_pdf`                        |
| DB                    | `monitors`, `monitor_runs`, `monitor_alerts` (Alembic 0005)       |

The scheduler reads `monitors.next_run_at` every 30 s
(`SELECT … FOR UPDATE SKIP LOCKED`); when a row is due it creates a
`monitor_runs` row in `status='queued'` and pushes the ID onto the
Redis queue (`vlabs:monitor:queue`). Workers BRPOP from the queue,
mark the run `running`, drive the customer endpoint per-episode,
score, aggregate, render a PDF, upload to R2, persist the summary +
verdict, and (if warning/regressed) dispatch alerts.

## Tiers

| Tier        | Monitors | Envs / monitor | Episodes / env |
|-------------|----------|----------------|----------------|
| free        | 1        | 1              | 10             |
| pro         | 3        | 3              | 30             |
| team        | 10       | 5              | 50             |
| enterprise  | ∞        | up to 25       | up to 200      |

Phase 28 enforces the caps at `POST /v1/monitors` create-time;
metered overage billing (D8-D) lands post-Stripe-incorporation.

## See also

- [`/v1/monitors`](../api-reference/monitors.md) — endpoint reference.
- [PHASE_28_PLAN.md](../../PHASE_28_PLAN.md) — full architectural
  decisions (D1-D11), scheduler model, regression statistical
  contract, alert dispatch policy.
- [Conformal calibration](../conformal.md) — the Layer 1 moat
  underwriting the D5-C drift signal.
