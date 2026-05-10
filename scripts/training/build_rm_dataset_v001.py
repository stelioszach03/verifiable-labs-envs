#!/usr/bin/env python3
"""scripts/training/build_rm_dataset_v001.py — package + upload v0.0.1 RM dataset.

Reads the post-judge JSONL output from
``reports/reward_distillation/v001/judged.jsonl`` (or any path
specified via ``--input``), normalises it to the canonical RM
training shape, writes a parquet copy alongside the JSONL, and
optionally uploads both to the HF Hub repo created by
``scripts/training/provision_hf_repos.py``
(``<owner>/rm-dataset-v0.0.1``).

A README.md is auto-generated with the row count, schema, and the
provenance of the judgment slice (live OpenRouter / stub / mixed).

Usage::

    # Local-only — no HF upload, just package the artefact:
    python3 scripts/training/build_rm_dataset_v001.py --no-upload

    # Default — package + upload to HF Hub:
    python3 scripts/training/build_rm_dataset_v001.py

    # Custom input + repo:
    python3 scripts/training/build_rm_dataset_v001.py \\
        --input reports/reward_distillation/v001/judged.jsonl \\
        --repo stelioszach03/rm-dataset-v0.0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("reports/reward_distillation/v001/judged.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/reward_distillation/v001/dataset")
DEFAULT_REPO_TEMPLATE = "{owner}/rm-dataset-v0.0.1"


def _load_env_file(path: Path) -> None:
    """Source a flat ``KEY=value`` env file into ``os.environ``.

    Identical helper to ``provision_hf_repos.py``. Does NOT overwrite
    values already in the env so the shell wins on a deliberate
    export.
    """
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
    """Project a Phase 29 RewardTrainingRow JSONL line down to the
    public v0.0.1 schema.

    Public schema (smaller than the internal RewardTrainingRow shape;
    drops volatile + reproducibility-only fields):

        row_id, env_id, prompt, completion, env_reward,
        frontier_judgment, consensus_reward, conformal_low,
        conformal_high, source, schema_version
    """
    interval = row.get("conformal_interval") or [None, None]
    low, high = (interval + [None, None])[:2]
    return {
        "row_id": row.get("row_id"),
        "env_id": row.get("env_id"),
        "prompt": row.get("prompt"),
        "completion": row.get("completion"),
        "env_reward": row.get("env_reward"),
        "frontier_judgment": row.get("frontier_judgment"),
        "consensus_reward": row.get("consensus_reward"),
        "conformal_low": low,
        "conformal_high": high,
        "source": row.get("source"),
        "schema_version": "v0.0.1",
    }


def package(
    *,
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Read the judged JSONL, project to v0.0.1, and write JSONL +
    optional parquet copies under ``output_dir``. Returns a manifest
    dict with row counts, source-mix, and the file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(input_path)
    canonical = [_row_to_canonical(r) for r in rows]

    jsonl_path = output_dir / "rm-dataset-v0.0.1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in canonical:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    parquet_path: Path | None = None
    try:
        import pandas as pd  # type: ignore[import-not-found]

        parquet_path = output_dir / "rm-dataset-v0.0.1.parquet"
        pd.DataFrame.from_records(canonical).to_parquet(parquet_path, index=False)
    except Exception:
        # Parquet is a nice-to-have; never fail the package step on it.
        parquet_path = None

    n_total = len(canonical)
    n_judged = sum(1 for r in canonical if r.get("frontier_judgment") is not None)
    sources: dict[str, int] = {}
    for r in canonical:
        s = r.get("source") or "unknown"
        sources[s] = sources.get(s, 0) + 1

    manifest = {
        "n_rows": n_total,
        "n_with_frontier_judgment": n_judged,
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
        "- reward-modelling",
        "- preference-learning",
        "- verifiable-labs",
        "size_categories:",
        f"- n<{max(manifest['n_rows'] * 2, 100)}",
        "---",
        "",
        "# Verifiable Labs RM dataset v0.0.1",
        "",
        "Reward-model training data for the Verifiable Labs SDK,",
        "produced by the Phase 29 reward-distillation pipeline.",
        "",
        "## Stats",
        "",
        f"- Rows: {manifest['n_rows']}",
        f"- With frontier judgment: {manifest['n_with_frontier_judgment']}",
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
        "| field                | type            | meaning |",
        "| -------------------- | --------------- | ------- |",
        "| ``row_id``           | str             | unique id |",
        "| ``env_id``           | str             | env that produced the row |",
        "| ``prompt``           | str             | task prompt |",
        "| ``completion``       | str             | candidate completion |",
        "| ``env_reward``       | float \\| null   | env-procedural reward |",
        "| ``frontier_judgment``| float \\| null   | optional Claude-Sonnet score |",
        "| ``consensus_reward`` | float           | 70/30 D5-D blend (the actual training target) |",
        "| ``conformal_low``    | float \\| null   | Phase 22 conformal lower |",
        "| ``conformal_high``   | float \\| null   | Phase 22 conformal upper |",
        "| ``source``           | str             | ``env``/``external``/``judgment`` |",
        "| ``schema_version``   | str             | ``v0.0.1`` |",
        "",
        "## License",
        "",
        "Apache-2.0 (matches the upstream SDK).",
        "",
        "## Provenance",
        "",
        "Built by ``scripts/training/build_rm_dataset_v001.py`` from the",
        "post-judge JSONL output of ``vlabs-reward-data extract`` +",
        "``vlabs-reward-data judge``. See the upstream repo at",
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
    """Upload the JSONL + parquet + README to the HF Hub repo. Idempotent."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", exist_ok=True
    )
    # README first (small + atomic).
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
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSONL input from `vlabs-reward-data judge` (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the packaged dataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help=(
            "HF Hub repo id; defaults to ``<HF_user>/rm-dataset-v0.0.1``."
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the HF Hub upload step (local package only).",
    )
    ns = parser.parse_args(argv)

    if not ns.input.exists():
        print(
            f"ERROR: input file does not exist: {ns.input}",
            file=sys.stderr,
        )
        return 2

    manifest = package(input_path=ns.input, output_dir=ns.output_dir)
    print(f"  packaged {manifest['n_rows']} rows → {ns.output_dir}")
    print(f"  with frontier judgment: {manifest['n_with_frontier_judgment']}")

    if ns.no_upload:
        return 0

    _load_env_file(Path.home() / ".vlabs-secrets" / "training-secrets.env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN missing — cannot upload. Use --no-upload "
            "to skip.",
            file=sys.stderr,
        )
        return 2

    # Resolve owner via whoami if --repo wasn't supplied.
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
