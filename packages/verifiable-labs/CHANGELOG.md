# Changelog

All notable changes to **verifiable-labs** (the Python SDK for the
Verifiable Labs Hosted Evaluation API) are documented here.

This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0a1] — 2026-04-25

Initial alpha. Mirrors the Hosted Evaluation API v0.1.0-alpha
endpoints.

### Added

- `Client` (sync) and `AsyncClient` (async) clients backed by `httpx`.
- `Environment` and `AsyncEnvironment` handles with
  `.evaluate(seed, answer)` (one-shot) and
  `.start_session(seed)` (multi-turn) flows.
- `Session` and `AsyncSession` classes with `.submit()`, `.history`,
  `.complete`, `.refresh()`.
- `client.leaderboard(env_id).top_models(n=5)` helper.
- Typed exception hierarchy: `VerifiableLabsError` and the HTTP-status
  subclasses `TransportError`, `InvalidRequestError`, `NotFoundError`,
  `RateLimitError`, `ServerError`.
- Pydantic v2 mirror models for every API schema.

### Known limits (alpha)

- API auth not enforced server-side; `api_key` is accepted but unused.
- Multi-turn turn-dispatch is not yet implemented server-side; the
  SDK exposes the full multi-turn shape for forward-compat with
  v0.2.
- Structured `answer` payloads return HTTP 422; pass strings.

### Compatibility

- Python `>=3.11`.
- httpx `>=0.27`, pydantic `>=2.7`.
