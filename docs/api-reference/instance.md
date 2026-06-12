# `POST /v1/instance`

**Procedural problem-instance fetch.** Phase 22.B.

Stateless: re-derives the instance deterministically from
`(env_id, seed, difficulty_kwargs)`. No server-side cache. Pair with
[`POST /v1/score`](score.md) to drive a customer training loop.

## Auth

`X-Vlabs-Key` header (data plane). Counts against the per-tier
`scores_per_month` quota — shared with `/v1/score`.

## Request

```http
POST /v1/instance HTTP/1.1
Content-Type: application/json
X-Vlabs-Key: vlk_<32-char>

{
  "env_id": "math-algebra",
  "seed": 42,
  "difficulty_kwargs": {}
}
```

| Field              | Type    | Required | Notes                                       |
|--------------------|---------|----------|---------------------------------------------|
| `env_id`           | string  | yes      | One of the 13 registered envs.              |
| `seed`             | int     | yes      | ≥ 0; deterministic instance generator.      |
| `difficulty_kwargs`| object? | no       | Pass-through to `env.generate_instance()`.  |

## Response — 200 OK

```json
{
  "instance_seed": 42,
  "prompt": "PROBLEM:\nExpand the product (3*x + 2) * (-1*x + 5)...\n\nOUTPUT SCHEMA:\n{...}",
  "metadata": {
    "alpha": 0.1,
    "simplify_timeout_s": 5.0,
    "template": "expand_binomial_product"
  },
  "env_version": "0.0.1"
}
```

| Field           | Type   | Notes                                                              |
|-----------------|--------|--------------------------------------------------------------------|
| `instance_seed` | int    | Echoes the request seed. Use with `/v1/score`.                     |
| `prompt`        | string | LLM-facing problem text via the env's adapter.                     |
| `metadata`      | object | Public env state; oracle fields (e.g. `gold_expr`, `x_true`) excluded. |
| `env_version`   | string | Pinned per fetch.                                                  |

## Per-env prompt-shape contract

| Env family       | `prompt` shape                                                                 |
|------------------|--------------------------------------------------------------------------------|
| symbolic-math    | Short JSON-ish block: `"PROBLEM: <natural language>\n\nOUTPUT SCHEMA: {...}"`. |
| inverse-problem  | JSON object with measurement payload as integer-scaled lists (`y_re_x1000`, `mask`, `sigma_x1000`, …). |

## Error responses

| Status | Code                  | When                                                    |
|--------|-----------------------|---------------------------------------------------------|
| 401    | `invalid_api_key`     | Missing / malformed / revoked `X-Vlabs-Key`.            |
| 402    | `quota_exceeded`      | Tier `scores_per_month` exhausted.                      |
| 404    | `unknown_environment` | `env_id` not in the registered env catalogue (or `difficulty_kwargs` rejected). |
| 422    | (Pydantic)            | Body validation: missing field, wrong type, negative seed. |
| 429    | `rate_limited`        | Tier RPM exceeded.                                      |

## Determinism

Two calls with the same `(env_id, seed, difficulty_kwargs)` return
**identical** `prompt` + `metadata`. This is the bit-for-bit
reproduction property the procedural-regeneration certification
relies on; it's enforced by the env's `generate_instance(seed)`
contract.

## Examples

### Python

```python
import httpx

r = httpx.post(
    "https://api.verifiable-labs.com/v1/instance",
    headers={"X-Vlabs-Key": "vlk_..."},
    json={"env_id": "math-algebra", "seed": 42},
)
inst = r.json()
print(inst["prompt"])
```
