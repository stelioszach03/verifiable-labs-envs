#!/usr/bin/env python3
"""scripts/training/build_prm_dataset_v001.py — package + upload v0.0.1 PRM dataset.

Mirror of ``build_rm_dataset_v001.py`` for the Phase 30 process-reward
dataset. Reads the post-judge JSONL from
``reports/process_reward/v001/judged.jsonl``, normalises to the public
v0.0.1 PRM schema, writes JSONL + parquet, generates a README, and
optionally uploads to ``<owner>/prm-dataset-v0.0.1`` on HF Hub.

The on-disk row shape collapses ``ProcessRewardTraceRow`` to its
public per-trace projection (steps + per-step rewards + per-step
frontier judgments + aggregate reward) so downstream PRM trainers
can consume the dataset without importing the SDK's internal
dataclasses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("reports/process_reward/v001/judged.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/process_reward/v001/dataset")
DEFAULT_REPO_TEMPLATE = "{owner}/prm-dataset-v0.0.1"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key and val and key not in os.environ:
            os.environ[key] = val


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_to_canonical(row: dict[str, Any]) -> dict[str, Any]:
    """Project ProcessRewardTraceRow JSONL → public v0.0.1 PRM schema."""
    return {
        "row_id": row.get("row_id"),
        "env_id": row.get("env_id"),
        "prompt": row.get("prompt"),
        "steps": list(row.get("steps") or []),
        "step_rewards": list(row.get("step_rewards") or []),
        "step_frontier_judgments": list(row.get("step_frontier_judgments") or []),
        "step_consensus_rewards": list(row.get("step_consensus_rewards") or []),
        "aggregate_reward": row.get("aggregate_reward"),
        "source": row.get("source"),
        "schema_version": "v0.0.1",
    }


def package(*, input_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(input_path)
    canonical = [_row_to_canonical(r) for r in rows]

    jsonl_path = output_dir / "prm-dataset-v0.0.1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in canonical:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    parquet_path: Path | None = None
    try:
        import pandas as pd  # type: ignore[import-not-found]

        # Parquet dislikes mixed-type lists; coerce step_rewards etc.
        # into JSON strings to round-trip cleanly.
        df_records = []
        for r in canonical:
            row_copy = dict(r)
            for key in (
                "steps",
                "step_rewards",
                "step_frontier_judgments",
                "step_consensus_rewards",
            ):
                row_copy[key] = json.dumps(row_copy.get(key) or [])
            df_records.append(row_copy)
        parquet_path = output_dir / "prm-dataset-v0.0.1.parquet"
        pd.DataFrame.from_records(df_records).to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None

    n_total = len(canonical)
    n_traces_with_judgments = sum(
        1
        for r in canonical
        if any(j is not None for j in r.get("step_frontier_judgments") or [])
    )
    total_steps = sum(len(r.get("steps") or []) for r in canonical)
    sources: dict[str, int] = {}
    for r in canonical:
        s = r.get("source") or "unknown"
        sources[s] = sources.get(s, 0) + 1

    manifest = {
        "n_traces": n_total,
        "n_steps": total_steps,
        "n_traces_with_step_judgments": n_traces_with_judgments,
        "by_source": sources,
        "schema_version": "v0.0.1",
        "files": {
            "jsonl": str(jsonl_path.relative_to(output_dir.parent)),
            "parquet": (
                str(parquet_path.relative_to(output_dir.parent))
                if parquet_path is not None
                else None
            ),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def render_readme(manifest: dict[str, Any], owner: str) -> str:
    lines = [
        "---",
        "license: apache-2.0",
        "language: en",
        "tags:",
        "- process-reward-modelling",
        "- prm",
        "- verifiable-labs",
        "size_categories:",
        f"- n<{max(manifest['n_traces'] * 2, 100)}",
        "---",
        "",
        "# Verifiable Labs PRM dataset v0.0.1",
        "",
        "Per-step process-reward training data for the Verifiable Labs SDK,",
        "produced by the Phase 30 process-reward pipeline.",
        "",
        "## Stats",
        "",
        f"- Traces: {manifest['n_traces']}",
        f"- Total steps: {manifest['n_steps']}",
        f"- Traces with per-step frontier judgments: "
        f"{manifest['n_traces_with_step_judgments']}",
        "- Frontier judge: ``anthropic/claude-sonnet-4`` (when judged)",
        "- Source mix:",
    ]
    for src, n in sorted(manifest.get("by_source", {}).items()):
        lines.append(f"  - ``{src}``: {n}")
    lines += [
        "",
        "## Schema",
        "",
        "Each row is a JSON object with the following fields:",
        "",
        "| field                          | type            | meaning |",
        "| ------------------------------ | --------------- | ------- |",
        "| ``row_id``                     | str             | unique id |",
        "| ``env_id``                     | str             | env that produced the trace |",
        "| ``prompt``                     | str             | task prompt |",
        "| ``steps``                      | list[str]       | per-step text |",
        "| ``step_rewards``               | list[float]     | env-procedural per-step rewards |",
        "| ``step_frontier_judgments``    | list[float]     | optional Claude-Sonnet per-step |",
        "| ``step_consensus_rewards``     | list[float]     | 70/30 D5-D blend (per-step training target) |",
        "| ``aggregate_reward``           | float           | trace-level score |",
        "| ``source``                     | str             | ``env``/``processbench``/``judgment`` |",
        "| ``schema_version``             | str             | ``v0.0.1`` |",
        "",
        "## License",
        "",
        "Apache-2.0 (matches the upstream SDK).",
        "",
        "## Provenance",
        "",
        "Built by ``scripts/training/build_prm_dataset_v001.py`` from the",
        "post-judge JSONL output of ``vlabs-prm-data extract`` +",
        "``vlabs-prm-data judge-steps``. See the upstream repo at",
        f"https://github.com/{owner}/verifiable-labs-envs for the",
        "full pipeline source.",
    ]
    return "\n".join(lines) + "\n"


def upload_to_hf(
    *,
    repo_id: str,
    output_dir: Path,
    readme_text: str,
    token: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: v0.0.1 README",
    )
    uploads: dict[str, str] = {}
    for f in sorted(output_dir.glob("*")):
        if not f.is_file():
            continue
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"feat: upload {f.name} (v0.0.1)",
        )
        uploads[f.name] = str(f.relative_to(output_dir.parent))
    return {"repo_id": repo_id, "uploaded": uploads}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--no-upload", action="store_true")
    ns = parser.parse_args(argv)

    if not ns.input.exists():
        print(f"ERROR: input does not exist: {ns.input}", file=sys.stderr)
        return 2

    manifest = package(input_path=ns.input, output_dir=ns.output_dir)
    print(
        f"  packaged {manifest['n_traces']} traces "
        f"({manifest['n_steps']} steps) → {ns.output_dir}"
    )
    print(
        f"  with per-step judgments: "
        f"{manifest['n_traces_with_step_judgments']}"
    )

    if ns.no_upload:
        return 0

    _load_env_file(Path.home() / ".vlabs-secrets" / "training-secrets.env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN missing — cannot upload. Use --no-upload to skip.",
            file=sys.stderr,
        )
        return 2

    repo_id = ns.repo
    owner = "anonymous"
    try:
        from huggingface_hub import HfApi

        owner = HfApi(token=token).whoami().get("name", "anonymous")
    except Exception:  # noqa: BLE001
        pass
    if repo_id is None:
        repo_id = DEFAULT_REPO_TEMPLATE.format(owner=owner)

    readme_text = render_readme(manifest, owner=owner)
    out = upload_to_hf(
        repo_id=repo_id,
        output_dir=ns.output_dir,
        readme_text=readme_text,
        token=token,
    )
    print(f"  uploaded to https://huggingface.co/datasets/{out['repo_id']}")
    print(f"  files: {sorted(out['uploaded'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
