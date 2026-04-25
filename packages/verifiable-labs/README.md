# verifiable-labs

Python SDK for the **Verifiable Labs Hosted Evaluation API** —
evaluate frontier LLMs on conformal-calibrated scientific RL
environments without writing any HTTP plumbing.

> **v0.1.0a1 — alpha.** The Hosted Evaluation API itself is
> v0.1.0-alpha (open / rate-limited / no auth / single-process
> session store). This SDK is a thin httpx wrapper that mirrors the
> 8-endpoint API surface; it'll keep working when we add auth +
> persistence in v0.2.

## Install

```bash
pip install verifiable-labs
```

Python `>=3.11` required.

## Quickstart

### Synchronous

```python
from verifiable_labs import Client

with Client() as client:                                # localhost:8000 by default
    print(client.health().version)                      # "0.1.0-alpha"

    env = client.env("stelioszach/sparse-fourier-recovery")
    result = env.evaluate(
        seed=0,
        answer='{"support_idx": [12, 47, 91], "support_amp_x1000": [800, -300, 1200]}',
        env_kwargs={"calibration_quantile": 2.0},
    )
    print(f"reward={result.reward:.3f}  parse_ok={result.parse_ok}")
```

### Asynchronous

```python
import asyncio
from verifiable_labs import AsyncClient

async def main():
    async with AsyncClient(base_url="https://api.verifiable-labs.com") as client:
        env = client.env("sparse-fourier-recovery")
        # Multi-turn flow: keep submitting until session.complete is True
        session = await env.start_session(seed=42)
        while not session.complete:
            answer = my_agent.solve(session.observation)         # your code
            await session.submit(answer_text=answer)
        print("turns:", len(session.history))

asyncio.run(main())
```

### Leaderboard

```python
lb = client.leaderboard("sparse-fourier-recovery")
for row in lb.top_models(n=3):
    print(f"{row.model:35s}  mean={row.mean_reward:.3f}  n={row.n}")
```

## Public surface

| name | sync / async | purpose |
|---|---|---|
| `Client(api_key=None, base_url=...)` | sync | top-level client |
| `AsyncClient(api_key=None, base_url=...)` | async | top-level client |
| `client.health()` | both | liveness + version |
| `client.environments()` | both | list all 10 envs |
| `client.env(env_id)` | both | returns `Environment` handle |
| `client.leaderboard(env_id)` | both | aggregated benchmark numbers |
| `env.evaluate(seed, answer)` | both | one-shot eval, returns `SubmitResponse` |
| `env.start_session(seed)` | both | returns multi-turn `Session` |
| `session.submit(answer_text=...)` | both | append a turn, returns score |
| `session.history` | sync (property) | list of past `SubmitResponse`s |
| `session.complete` | sync (property) | `bool` — env signalled done |
| `session.refresh()` | both | re-fetch state from the server |

## Exceptions

The SDK raises typed exceptions on non-2xx HTTP status codes; callers
can `except` on the specific failure mode.

```python
from verifiable_labs import (
    VerifiableLabsError,        # base class
    TransportError,             # network / timeout
    InvalidRequestError,        # 400 / 422
    NotFoundError,              # 404
    RateLimitError,             # 429
    ServerError,                # 5xx
)
```

## Configuration

```python
Client(
    api_key=None,               # forward-compat for v0.2; no effect in v0.1
    base_url="http://localhost:8000",
    timeout=30.0,               # httpx total-timeout in seconds
    http_client=None,           # inject your own httpx.Client for custom transport
)
```

`AsyncClient` takes the same args + accepts an `httpx.AsyncClient`.

## What's NOT in v0.1

Same caveats as the Hosted Evaluation API:

- No authentication. `api_key=` is accepted for forward-compat but
  unused.
- Multi-turn sessions don't yet route turns through the env's
  residual-feedback rollout (server records turns but doesn't
  dispatch). The SDK exposes the full `Session` API anyway so the
  shape is stable for v0.2.
- Structured `answer` dicts return HTTP 422; pass strings.
- No persistence — session store is in-memory on the API side.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
