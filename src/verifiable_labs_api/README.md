# `verifiable_labs_api` — Hosted Evaluation API package

FastAPI app for the Verifiable Labs Evaluation API, **v0.1.0-alpha**.
This README is the in-package reference; deployment instructions
live in [`deploy/api/README.md`](../../deploy/api/README.md).

## Module layout

| file | purpose |
|---|---|
| `__init__.py` | package version + `create_app` re-export |
| `app.py` | FastAPI factory + 6 `/v1/*` route handlers |
| `schemas.py` | Pydantic v2 request / response models |
| `sessions.py` | `SessionStore` — in-memory store with TTL eviction |
| `leaderboard.py` | per-env aggregation over `results/*.csv` |
| `registry.py` | env metadata (domain, multi_turn, tool_use flags) |
| `serialization.py` | NumPy + complex → JSON-safe primitives |

The package depends on `verifiable_labs_envs` (the env code) and
the `[api]` extras group from the root `pyproject.toml`.

## Run locally

```bash
pip install -e ".[dev,api]"
uvicorn verifiable_labs_api.app:app --port 8000 --reload
open http://localhost:8000/docs
```

## How a request flows through the code

```
                                                  ┌─ app.create_app
HTTP POST /v1/sessions                            │
   ↓                                              ▼
   ├── slowapi (rate limit)                  ┌──────────────┐
   ├── CORS middleware                       │   FastAPI    │
   ├── Pydantic CreateSessionRequest         │              │
   ↓                                         └─────┬────────┘
   ├── normalize_env_id (strip "owner/")           │
   ├── verifiable_labs_envs.load_environment       │
   ├── env.generate_instance(seed)                 │
   ├── adapter.build_user_prompt(instance) ─→ prompt_text
   ├── instance.as_inputs() ─→ to_json_safe ─→ inputs
   ↓
   SessionStore.make_session  (uuid + TTL)
   ↓
   Pydantic CreateSessionResponse  (201)
```

```
HTTP POST /v1/sessions/{id}/submit
   ↓
   ├── SessionStore.get(id)  (404 if expired/missing)
   ├── adapter.parse_response(answer_text, instance) ─→ Prediction
   ├── env.score(prediction, instance)
   ↓
   Submission appended; SubmitResponse returned (200)
```

## Tests

`tests/api/` (run with `pytest tests/api/`):

- `test_health.py` — 5 tests (200, version label, OpenAPI exposed).
- `test_environments.py` — 6 tests (listing, qualified ids, flags).
- `test_session_lifecycle.py` — 11 tests (create → submit → get,
  including parse-fail, multi-turn, structured-answer rejection).
- `test_leaderboard.py` — 6 tests (rows sorted, qualified id form).
- `test_rate_limit.py` — 5 tests (slowapi cap, TTL eviction).
- `test_serialization.py` — 10 tests (NumPy + complex encoders).

44 tests total at v0.1.0-alpha. All target the existing
`verifiable_labs_envs` envs without mocking, so the suite verifies
the full integration.

## Configuration knobs

| env var | default | meaning |
|---|---|---|
| `VL_API_RATE_LIMIT` | `30/minute` | slowapi rule per IP |
| `VL_API_CORS_ORIGINS` | `*` | CSV of allowed origins, or `*` |

`create_app(rate_limit=..., session_ttl_seconds=...)` is the
test-friendly factory; production uses the env-var path.

## Quality bar (alpha)

✓ FastAPI app boots, all 6 `/v1/*` routes registered, OpenAPI UI live.
✓ Round-trip: `POST /sessions` → `POST /submit` → `GET /sessions/{id}`
  works for sparse-Fourier and multi-turn envs out of the box.
✓ Reads the existing benchmark CSVs for leaderboard aggregation —
  no fabricated rows.
✓ 44 passing tests; existing 254 monorepo tests still green.
✓ Render / Fly / Docker configs ready, no DNS dependency.

✗ No auth — anyone with the URL can submit.
✗ No persistence — session store is in-memory.
✗ Multi-turn submit currently records turns but doesn't dispatch
  the env's residual-feedback rollout.
✗ Structured `answer` payload rejected with HTTP 422.

The four ✗ items are the explicit Tier-2 follow-up.
