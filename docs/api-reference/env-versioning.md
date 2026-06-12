# Env versioning

**Every `audit_calls` row pins the `env_version` used to score it.**
Phase 22.

The version comes from `verifiable_labs_envs.__version__` at score
time. Customers can detect catalogue evolution after the fact — if
your local training-loop assumes a certain reward distribution and
the server's `env_version` changes mid-experiment, your downstream
logs flag the discontinuity.

## Bump policy

| Bump  | When                                                    | Examples                                  |
|-------|---------------------------------------------------------|-------------------------------------------|
| MAJOR | Schema-breaking change to `Instance` / `Prediction` dataclass shape | Removing a field; changing units; changing types. |
| MINOR | Reward-distribution-altering change in env code         | Template change; baseline solver change; hyperparam default change. |
| PATCH | Bug fix that does NOT alter reward distribution         | Off-by-one in a comment; performance fix; type-annotation cleanup. |

## What you can rely on

- **Within the same `env_version`**: identical `(env_id, seed,
  difficulty_kwargs, completion)` → identical `reward`,
  identical `components_breakdown`. Bit-for-bit reproducible.
- **Across MINOR bumps**: schema unchanged, but reward distribution
  may shift. Recalibrate any local conformal threshold against a fresh
  pool of baseline runs.
- **Across PATCH bumps**: zero behavioural change. Safe to ignore.

## Detecting drift in your training loop

The server returns `env_version` on every score call. Pin the value
you started training under and alert on mismatch:

```python
INITIAL_VERSION = None

def score(env_id, seed, completion):
    global INITIAL_VERSION
    r = client.post("/v1/score", json={...})
    body = r.json()
    if INITIAL_VERSION is None:
        INITIAL_VERSION = body["env_version"]
    elif body["env_version"] != INITIAL_VERSION:
        log.warning("env_version drift",
                    pinned=INITIAL_VERSION, observed=body["env_version"])
    return body["reward"], body["audit_id"]
```

For audit-trail forensics, query `/v1/score/audit?limit=1000` and
group by `env_version` to see when the cutover happened on the server.

## Idempotency interaction

`(idempotency_key, user_id)` is unique within the 24 h window
regardless of `env_version`. If a server rolls forward to a new MINOR
during your dedup window, a retried score request returns the **original**
audit row — including its **original** `env_version` — not the new
one. This is intentional: idempotency means the answer doesn't change,
even if the underlying engine has.

## Operational policy

We bump `verifiable_labs_envs.__version__` on every release that
ships a behavioural change. The CHANGELOG (in the repo root) names
which bump and which env(s) were affected.
