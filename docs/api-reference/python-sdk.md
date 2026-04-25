# Python SDK reference

`pip install verifiable-labs` — sync + async clients for the Hosted
Evaluation API. Pydantic v2 models for every response shape; httpx
under the hood with sane retry/timeout defaults.

## Install

```bash
pip install verifiable-labs              # release
pip install verifiable-labs[dev]         # +pytest, respx for testing
```

The package is namespaced separately from the env code — distribution
name `verifiable-labs`, import name `verifiable_labs`. The env
monorepo is `verifiable-labs-envs` (note the `-envs`); the two are
independent.

## Surface

```python
from verifiable_labs import Client, AsyncClient
from verifiable_labs.models import HealthStatus, EnvironmentList, SubmitResponse
from verifiable_labs.exceptions import VerifiableLabsError, EnvNotFoundError, RateLimitError
```

### `Client(base_url=..., api_key=None, timeout=30.0)`

```python
with Client(base_url="https://api.verifiable-labs.com") as c:
    health = c.health()                  # → HealthStatus
    envs = c.environments()              # → EnvironmentList
    env = c.env("stelioszach/sparse-fourier-recovery")
    result = env.evaluate(seed=42, answer="{...}")  # one-shot
    session = env.start_session(seed=42)            # multi-turn
    lb = c.leaderboard("sparse-fourier-recovery")
```

`api_key` is reserved for v0.2; setting it today is a no-op (no auth
in v0.1) but the SDK forwards it as `X-VL-API-Key` if the server
expects it.

### `Environment.evaluate(seed, answer, env_kwargs=None)`

One-shot: starts a session, submits `answer`, returns the score. For
single-turn envs this is the whole story.

```python
result = env.evaluate(seed=42, answer=my_model_output)
print(f"reward={result.reward:.3f} parse_ok={result.parse_ok}")
print(f"components={result.components}")
print(f"coverage={result.coverage}")  # may be None if env doesn't expose conformal
```

### `Environment.start_session(seed, env_kwargs=None) -> Session`

Multi-turn: returns a `Session` you can call `submit()` on repeatedly.

```python
session = env.start_session(seed=42)
print(session.observation)              # the env's prompt + inputs
session.submit(answer)
print(session.history[-1].reward)       # most recent score
print(session.complete)                 # True once env signals done
```

### `Leaderboard.top_models(n=5)`

```python
lb = c.leaderboard("sparse-fourier-recovery")
for row in lb.top_models(n=5):
    print(f"{row.model:30s} mean={row.mean_reward:.3f}")
```

## Async

`AsyncClient` mirrors the sync API one-to-one with `await`:

```python
import asyncio
from verifiable_labs import AsyncClient

async def main():
    async with AsyncClient(base_url="https://api.verifiable-labs.com") as c:
        envs = await c.environments()
        env = c.env("stelioszach/sparse-fourier-recovery")
        result = await env.evaluate(seed=42, answer="{...}")
        print(result.reward)

asyncio.run(main())
```

## Exception hierarchy

```text
VerifiableLabsError
├── EnvNotFoundError      (404 on /v1/sessions with unknown env_id)
├── SessionExpiredError   (404 on /v1/sessions/{id} when expired)
├── ParseError            (422 — answer_text couldn't be parsed)
├── RateLimitError        (429)
└── ServerError           (5xx)
```

All HTTP errors raise a typed subclass; the original `httpx.Response`
is preserved on `.response` for advanced introspection.

## Testing against a real API in CI

The SDK ships an integration test
([`packages/verifiable-labs/tests/test_integration.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/packages/verifiable-labs/tests/test_integration.py))
that spins up a live `uvicorn` instance from `verifiable_labs_api.app`
and runs the SDK against it. Skipped when the API extras aren't
installed.

For unit tests use `respx` to mock httpx calls — see the SDK test
suite for examples.

## Source

[`packages/verifiable-labs/`](https://github.com/stelioszach03/verifiable-labs-envs/tree/main/packages/verifiable-labs).
