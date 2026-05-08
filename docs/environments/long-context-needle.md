# `long-context-needle`

**Single-turn long-context retrieval (Needle-in-Haystack).** Given
a procedurally generated multi-document corpus + a question, the
model returns the answer-bearing fact in a JSON envelope. The
needle is a distinctive token (e.g., `ABCD-1234`) injected at one
of four deterministic positions; the verifier scores by substring
match.

This is the first member of the **long-context** template family
introduced in Phase 27.

## Problem

| | |
|---|---|
| input | multi-document corpus + question, separated by `---DOCUMENT N: <title>---` headers |
| output | JSON `{"answer": "<extracted text>", "confidence": <float in [0, 1]>}` |
| gold | hidden needle token (e.g., `ABCD-1234`) injected at a deterministic position |
| dialect | UTF-8 plain text (D8-A locked); cl100k_base tokeniser for budget enforcement (D5) |

Ten procedural topic templates spanning the long-context
distribution: bio articles, science abstracts, news reports,
product reviews, technical manuals, legal documents, historical
summaries, recipe collections, travel logs, interview transcripts.
Three needle templates × three question phrasings × four position
modes (`start | middle | end | random`).

`EFFECTIVE_INSTANCES > 7 × 10²⁶`, well above the 1 × 10¹⁵
contamination-resistance gate.

## Variants

- [`long-context-needle`](#) — single-turn, this page.
- [`long-context-synthesis`](long-context-synthesis.md) — 3-5
  needles per instance, 3-turn dialogue with token-F1 feedback.
- [`long-context-reasoning`](long-context-reasoning.md) — 3
  multi-hop chain templates with distractor needles.

## Schema

```json
{
  "answer": "ABCD-1234",
  "confidence": 0.85
}
```

## Reward decomposition

```
reward = 0.10 · format_valid    (output is parseable JSON
                                  with an `answer` field)
       + 0.20 · parse_valid     (extracted answer is non-empty)
       + 0.70 · correctness     (D3-A: substring + case-insensitive
                                  match against the gold needle token)
```

Same weight structure as math / code / tool-calling / SQL —
preserves cross-env reward distribution comparability.

## D5 resource limits (locked)

| Surface              | Default            | Notes                                                  |
|----------------------|--------------------|--------------------------------------------------------|
| Test-default tokens  | 4 000              | Unit-test footprint cap (R7).                          |
| Sandbox-default      | 16 000             | API smoke tests run with this.                         |
| Production max       | 128 000            | D1-C cap; per-env hyperparameter.                      |
| Document count       | 8                  | Configurable via `document_count`.                     |
| Tokeniser            | `cl100k_base`      | Provider-agnostic; rewards are scored on text answers. |
| Corpus byte cap      | 64 MiB             | `DEFAULT_MAX_CORPUS_BYTES`.                            |

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("long-context-needle")
inst = env.generate_instance(seed=42)
print(inst.prompt)
```

## Tests

`tests/test_long_context_corpus.py` and
`tests/test_needle_injection.py` cover the platform-level
primitives; `tests/test_long_context_needle.py` covers this env's
reward kernel and adapter.
