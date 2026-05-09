# `vlabs-monitor` scaling runbook

> Operational playbook for the Phase 28 continuous-monitoring worker
> pool. Mirrors the
> [`vlabs-data-scaling`](vlabs-data-scaling.md) cutover doc.

## Default posture (D1-D)

In v0.0.1-alpha the monitor scheduler + worker pool run **in-process**
on the same Fly machine as the FastAPI API. Three tasks are spawned
at lifespan startup:

1. `scheduler_loop` — single instance per machine. Ticks every 30 s,
   reads `monitors WHERE status='active' AND next_run_at <= now()`
   under `SELECT FOR UPDATE SKIP LOCKED`, creates `monitor_runs`
   rows, pushes IDs onto the Redis queue.
2. `monitor_worker_loop × N` — drains
   `vlabs:monitor:queue` via BRPOP. `N` defaults to 2 (configurable
   via `vlabs_data_worker_pool_size` setting; reuses the dataset
   worker pool sizing for now).

This is sufficient up to **~25 active monitors** at default cadences.
Past that, the cutover triggers below fire.

## Cutover triggers (→ D1-B separate Fly app)

When **any** of the following holds, spin up a separate
`vlabs-monitor-worker` Fly app:

- Active-monitor count > 25.
- Scheduler tick latency p95 > 5 s.
- API machine peak RSS > 1.5 GB.
- Sustained queue backlog > 100 runs for > 15 min.

The cutover app uses the **same source tree** but a different
entrypoint:

```dockerfile
CMD ["python", "-m", "vlabs_api.monitor_worker"]
```

Add a thin `__main__` to `vlabs_api/monitor_worker.py` that calls
`asyncio.run(_main())` where `_main` initialises the engine, redis
client, and runs `spawn_monitor_pool` + `await asyncio.Event().wait()`.

The Redis queue is the only shared state — zero schema change.
The original `vlabs-api` Fly machine then disables its own monitor
pool by setting `vlabs_data_worker_pool_size=0` and removing the
`spawn_monitor_pool` call from `lifespan`.

## Restart recovery

The scheduler is **always-fresh** after deploy:

- `monitors.next_run_at` is the single source of truth — process
  restart loses no schedule.
- On worker pool startup, `reset_stale_running` resets any
  `monitor_runs.status='running' AND started_at < now() - 1h` rows
  to `failed` with `error='scheduler_lost_run'`.
- Then `rescue_queued_runs` re-enqueues any `status='queued'` rows
  the previous instance lost on Redis.
- Catch-up after extended downtime fires each monitor **once**, not
  N times — `compute_next_run_at` always returns a future timestamp.

## Common ops

### Pause all schedulers

```sql
UPDATE monitors SET status='paused' WHERE status='active';
```

To resume in bulk:

```sql
UPDATE monitors SET
  status='active',
  next_run_at=now() + interval '1 minute'
WHERE status='paused';
```

### Drain the Redis queue manually

```bash
redis-cli LRANGE vlabs:monitor:queue 0 -1   # peek
redis-cli DEL    vlabs:monitor:queue        # destructive
```

The worker pool will rescue all `status='queued'` runs on the next
restart; deleting the Redis queue is safe but causes a one-time
delay until the rescue loop fires.

### Inspect a stuck run

```sql
SELECT id, monitor_id, status, started_at, error
FROM monitor_runs
WHERE status='running' AND started_at < now() - interval '15 minutes';
```

If there are stuck rows after a deploy, the next restart will
auto-reset them; if you want to force the reset right now, restart
the API machine.

### Tail alert delivery

```sql
SELECT a.id, a.channel, a.delivered_at, a.delivery_error,
       r.scheduled_at, r.regression_verdict, m.name
FROM monitor_alerts a
JOIN monitor_runs r ON r.id = a.monitor_run_id
JOIN monitors m ON m.id = r.monitor_id
ORDER BY r.scheduled_at DESC
LIMIT 50;
```

`delivery_error` field carries the per-channel reason string when
the dispatcher couldn't reach the channel.

## Secrets

- `VLABS_DATA_LLM_KEY_ENCRYPTION` — Fernet key used to encrypt
  customer auth tokens at rest. **Required**. Rotation invalidates
  every existing monitor's token; the customer must re-supply it
  via `PATCH /v1/monitors/{id}`.
- `VLABS_EMAIL_FROM` + `VLABS_EMAIL_API_KEY` — email transport
  (Resend / SendGrid / SES). Optional; absent → LOCAL_FAKE_EMAIL
  mode (writes `.eml` stubs to `/tmp/vlabs-emails/`).
- `VLABS_SLACK_WEBHOOK_DEFAULT` — fallback Slack webhook; optional.

The local helper `_load_phase28_secrets.sh` prompts for these
without exposing them in shell history.
