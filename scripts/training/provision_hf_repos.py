#!/usr/bin/env python3
"""scripts/training/provision_hf_repos.py — idempotent HF Hub repo provisioning.

Creates / verifies the four canonical Verifiable Labs Hub repos used by
the Tier 4 reward-model and process-reward-model training pipelines::

    datasets/{owner}/rm-dataset-v0.0.1     (RewardBench-derived RM training)
    datasets/{owner}/prm-dataset-v0.0.1    (ProcessBench-derived PRM training)
    {owner}/rm-qwen-1-5b-v0.0.1            (Qwen2.5-1.5B RM model card)
    {owner}/prm-qwen-1-5b-v0.0.1           (Qwen2.5-1.5B PRM model card)

Owner is derived from the authenticated ``whoami`` call so the script is
maintainer-agnostic — anyone with a valid ``HF_TOKEN`` and the
``write`` scope can run it.

Idempotent: uses ``HfApi.create_repo(exist_ok=True)``. Re-runs are
safe and a no-op on existing repos.

Usage::

    python3 scripts/training/provision_hf_repos.py            # human table
    python3 scripts/training/provision_hf_repos.py --json     # machine
    python3 scripts/training/provision_hf_repos.py --dry-run  # no API
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepoSpec:
    """Declarative spec for one Hub repo."""

    repo_id_template: str  # "{owner}/rm-dataset-v0.0.1"
    repo_type: str  # "dataset" | "model"
    private: bool
    description: str


# The four canonical repos for v0.0.1 Tier 4 datasets + model cards.
REPOS: list[RepoSpec] = [
    RepoSpec(
        repo_id_template="{owner}/rm-dataset-v0.0.1",
        repo_type="dataset",
        private=False,
        description=(
            "Verifiable Labs reward-model training dataset (RewardBench slice)."
        ),
    ),
    RepoSpec(
        repo_id_template="{owner}/prm-dataset-v0.0.1",
        repo_type="dataset",
        private=False,
        description=(
            "Verifiable Labs process-reward-model dataset (ProcessBench slice)."
        ),
    ),
    RepoSpec(
        repo_id_template="{owner}/rm-qwen-1-5b-v0.0.1",
        repo_type="model",
        private=False,
        description=(
            "Qwen2.5-1.5B reward model fine-tuned on Verifiable Labs RM dataset."
        ),
    ),
    RepoSpec(
        repo_id_template="{owner}/prm-qwen-1-5b-v0.0.1",
        repo_type="model",
        private=False,
        description=(
            "Qwen2.5-1.5B process reward model fine-tuned on "
            "Verifiable Labs PRM dataset."
        ),
    ),
]


def _load_env_file(path: Path) -> None:
    """Source a flat ``KEY=value`` env file into ``os.environ``.

    Skips comments + blank lines. Does NOT overwrite values already
    set in the environment so the shell wins on a deliberate export.
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


def provision_one(
    api: Any,  # huggingface_hub.HfApi
    spec: RepoSpec,
    *,
    owner: str,
    dry_run: bool,
) -> dict[str, str]:
    """Create one repo (or noop if it exists). Returns a status dict."""
    repo_id = spec.repo_id_template.format(owner=owner)
    if dry_run:
        return {"repo_id": repo_id, "type": spec.repo_type, "status": "dry-run"}
    api.create_repo(
        repo_id=repo_id,
        repo_type=spec.repo_type,
        private=spec.private,
        exist_ok=True,
    )
    return {"repo_id": repo_id, "type": spec.repo_type, "status": "ok"}


def provision_all(*, token: str, dry_run: bool = False) -> list[dict[str, str]]:
    """Create / verify all four repos. Returns a list of status dicts."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()
    owner = user.get("name")
    if not owner:
        raise RuntimeError(
            "HF whoami returned no 'name' field — cannot derive owner"
        )
    return [
        provision_one(api, spec, owner=owner, dry_run=dry_run) for spec in REPOS
    ]


def render_table(results: list[dict[str, str]]) -> str:
    """Pretty-print a human-readable table."""
    lines = []
    lines.append(f"  {'repo_id':52s}  {'type':8s}  status")
    lines.append(f"  {'-' * 52}  {'-' * 8}  ------")
    for r in results:
        lines.append(
            f"  {r['repo_id']:52s}  {r['type']:8s}  {r['status']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print which repos WOULD be created without calling the API",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of the human table",
    )
    ns = parser.parse_args(argv)

    _load_env_file(Path.home() / ".vlabs-secrets" / "training-secrets.env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not in environment", file=sys.stderr)
        return 2

    try:
        results = provision_all(token=token, dry_run=ns.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if ns.json:
        print(json.dumps(results, indent=2))
    else:
        print(render_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
