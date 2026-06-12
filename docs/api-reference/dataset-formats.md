# Dataset output formats

Each row in a `/v1/datasets`-generated file is a **scored tuple**.
Job-level metadata — the env id, the storage SHA-256, the originating
`dataset_id` — lives on the `dataset_jobs` row exposed by
`GET /v1/datasets/{dataset_id}`. The per-row schema is:

| Field                | Type    | Notes                                                              |
|----------------------|---------|--------------------------------------------------------------------|
| `format_version`     | string  | Row schema version; bumped on schema changes.                       |
| `env_version`        | string  | Pinned per row (matches the env catalogue at submission time).      |
| `seed`               | int     | Instance generator seed; reproducible via `generate_instance(seed)`.|
| `prompt`             | string  | LLM-facing problem text (rendered through the env's adapter).       |
| `completion`         | string  | Raw LLM response text. UTF-8.                                       |
| `reward`             | float   | Calibrated reward in `[0, 1]`. `0.0` if the LLM call failed.        |
| `components`         | object  | Per-component breakdown (e.g. `format_valid`, `parse_valid`, `correct`). |
| `llm`                | object  | `{prompt_tokens, completion_tokens, cost_usd_estimate, success}`.   |

Parquet flattens `components` and `llm` into top-level columns
(`components_<name>` and `llm_<field>`) so consumers can scan
per-component metrics without unpacking nested types.

## Parquet (default)

- One row group per checkpoint chunk (default 1 000 rows; tunable via
  `VLABS_DATA_CHECKPOINT_EVERY_N`).
- Snappy-compressed.
- Schema is stable for a given `format_version` — new minor versions
  add optional columns; we never silently change types.

Read in Python:

```python
import pyarrow.parquet as pq

table = pq.read_table("vlabs-dataset.parquet")
df    = table.to_pandas()
```

## JSONL

One JSON object per line, newline-delimited. Useful when the consumer
can't pull in pyarrow, or when you want to `grep`/`jq` the file
directly. Same schema as Parquet — every row carries every field.

Read in Python:

```python
import json

rows = [json.loads(line) for line in open("vlabs-dataset.jsonl")]
```

## Reproducibility

Given the same `(env_id, env_version, seed)` tuple, the env's
`generate_instance` is deterministic — anyone holding the dataset
file can re-derive the original problem, regenerate the prompt, and
verify the reward by re-scoring against the same env. This is the
substrate behind the
[procedural-regeneration certification](../concepts/procedural-regeneration.md)
that drives the contamination-proof claim: even after public release,
the held-out test set is the seed range never used in training.

## Integrity

Every dataset file ships with a SHA-256 hash recorded in the
`storage_sha256` column of the job row (and surfaced in the
`/download` JSON response). The hash is computed **before** the file
is uploaded to R2, so transit corruption is detectable without
trusting the storage backend. Re-hash the downloaded file locally to
verify.
