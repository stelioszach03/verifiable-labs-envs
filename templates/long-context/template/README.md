# __ENV_ID__

Verifiable Labs long-context RL environment for **__DOMAIN__**.

This package was scaffolded from
``templates/long-context/`` via
``scripts/create_env.py __ENV_ID__ --template long-context --domain "__DOMAIN__"``.

## What this env does

The env hands the solver a **multi-document corpus + a question**
and expects a JSON envelope with a single ``answer`` field. The
corpus is procedurally generated from a 64-bit seed; one or more
**needles** (distinctive answer-bearing sentences) are injected
at deterministic positions. The solver returns the answer and
is scored by exact match (single-needle), token-F1 (synthesis),
or numeric match (chain reasoning).

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing an `answer` field                    |
| `parse_valid`   | 0.20   | Extracted answer is non-empty                                            |
| `correctness`   | 0.70   | Substring / token-F1 / numeric match against the gold needle             |

The shared corpus generator + needle injector + verification
helpers live at
``verifiable_labs_envs.long_context_primitives``; this env
imports them verbatim — D10-A locked in PHASE_27_PLAN.md.

## Filling in the scaffold

Replace the `NotImplementedError` stubs in:

- ``__ENV_PY__/data.py`` — `generate_problem(seed, **hyperparams)`
  must return a dict with `question`, `corpus`, `needle_text`,
  `needle_anchor`, `position_mode`, and a categorical
  `template_name`. Use `numpy.random.default_rng(seed)` for
  reproducibility; ensure `EFFECTIVE_INSTANCES > 1e15`
  (procedural-regeneration gate).
- ``__ENV_PY__/env.py`` — adjust hyperparams (target_tokens,
  document_count, alpha) if your env needs tighter or looser
  bounds.

The scoring kernel and adapter need no edits in most cases; the
default JSON envelope (``{"answer": "...", "confidence": <float>}``)
and the substring comparator cover the single-needle case.

## Running

```bash
python scripts/validate_env.py environments/__ENV_PY__/   # contract checks
pytest                                                     # unit + reward + corpus tests
```

## Why a separate template family

The inverse-problem template hard-codes forward operators + NMSE;
the symbolic-math template hard-codes SymPy equivalence; the
code-execution template hard-codes a sandboxed pytest runner;
the tool-calling template hard-codes OpenAI function-calling;
the sql-execution template hard-codes a SQLite sandbox. None
applies to long-context retrieval. The long-context family swaps
in:

- A ``NeedleInstance`` shape carrying ``corpus`` (a procedural
  multi-document blob) + ``question`` + ``needle_text`` +
  ``needle_anchor`` (document id + character offset) +
  ``position_mode`` (start / middle / end / random).
- A reward kernel that delegates the heavy lifting to
  ``verifiable_labs_envs.long_context_primitives.exact_match``
  (or ``token_f1`` / ``numeric_match`` for the synthesis /
  reasoning siblings).
- A test-default context cap of 4 000 tokens (D5) so the test
  suite stays fast; production callers scale to 128 000 tokens
  via the ``target_tokens`` hyperparameter.

## Trusted-input scope

This env's verifier runs entirely in-process (no sandbox) — long-
context retrieval scoring is pure string match / numeric
comparison; no untrusted code is executed. The only resource
bound the env enforces is the corpus byte cap
(``DEFAULT_MAX_CORPUS_BYTES = 64 MB``) which protects against
runaway template draws.
