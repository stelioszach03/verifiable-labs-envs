#!/usr/bin/env python3
"""scripts/training/provision_wandb_project.py — ensure W&B project exists.

By default tries the modern ``wandb.Api().create_project(name, entity)``
path (W&B SDK ≥ 0.17). On older SDKs that lack ``create_project`` it
falls back to the auto-creation behaviour of ``wandb.init(project=…)``
followed by ``wandb.finish()``, which logs a tiny "provisioning"
marker run.

Idempotent: ``create_project`` raises ``CommError`` ("project already
exists") on re-runs, which we catch and report as ``status=exists``.

Usage::

    python3 scripts/training/provision_wandb_project.py
    python3 scripts/training/provision_wandb_project.py --dry-run
    python3 scripts/training/provision_wandb_project.py --project foo

Returns exit 0 on success, 1 on init failure, 2 on missing token.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = "verifiable-labs"


def _load_env_file(path: Path) -> None:
    """Source a flat ``KEY=value`` env file into ``os.environ``.

    Mirrors the helper in :mod:`provision_hf_repos`; kept local so each
    script is independently runnable. Does NOT overwrite values already
    in the env.
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


def _provision_via_api(
    api_module: Any, project: str, entity: str | None
) -> dict[str, str]:
    """Try the modern ``Api.create_project`` path. Returns the status
    dict on success; raises ``AttributeError`` if ``create_project`` is
    missing (caller falls back). Idempotent on already-exists errors.
    """
    api = api_module.Api()
    create = api.create_project  # AttributeError → caller falls back
    target_entity = entity
    if target_entity is None:
        # Best effort: the authenticated default_entity.
        target_entity = getattr(api, "default_entity", None) or ""
    try:
        create(project, target_entity) if target_entity else create(project)
    except Exception as exc:  # noqa: BLE001
        # CommError or 409-style — treat "exists" as success.
        msg = str(exc).lower()
        if "exist" in msg or "already" in msg or "duplicate" in msg:
            return {"project": project, "entity": target_entity, "status": "exists"}
        raise
    return {"project": project, "entity": target_entity, "status": "created"}


def provision_project(
    project: str,
    *,
    api_key: str,
    entity: str | None = None,
    dry_run: bool = False,
    wandb_module: Any = None,
) -> dict[str, str]:
    """Create / verify the W&B project, return a status dict.

    Parameters
    ----------
    project:
        Project name on W&B.
    api_key:
        Token sourced from ``WANDB_API_KEY``.
    entity:
        Optional entity / username. Defaults to the authenticated
        ``default_entity`` when ``None``.
    dry_run:
        If ``True``, skip the network call and return ``status="dry-run"``.
    wandb_module:
        Injected wandb module — tests pass a stub. Default is the real
        ``wandb`` package import.
    """
    if dry_run:
        return {"project": project, "status": "dry-run"}

    if wandb_module is None:
        import wandb  # type: ignore[import-not-found]

        wandb_module = wandb

    # Tell wandb to use the env API key.
    os.environ["WANDB_API_KEY"] = api_key
    # `wandb.login(anonymous="never")` validates the token and primes
    # the local cache; safe to call repeatedly.
    wandb_module.login(key=api_key, anonymous="never", relogin=True)

    # Modern path — Api.create_project (wandb ≥ 0.17). No marker run.
    try:
        return _provision_via_api(wandb_module, project, entity)
    except AttributeError:
        pass

    # Legacy fallback — init/finish a tiny marker run to force project
    # creation server-side.
    run = wandb_module.init(
        project=project,
        name="provisioning",
        job_type="provision",
        tags=["provisioning", "automated"],
        reinit=True,
    )
    run_entity = getattr(run, "entity", entity or "?")
    run_url = getattr(run, "url", "")
    wandb_module.finish()
    return {
        "project": project,
        "entity": run_entity,
        "run_url": run_url,
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"W&B project name (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip the network call",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of human text",
    )
    ns = parser.parse_args(argv)

    _load_env_file(Path.home() / ".vlabs-secrets" / "training-secrets.env")
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WANDB_API_KEY not in environment", file=sys.stderr)
        return 2

    try:
        result = provision_project(
            ns.project, api_key=api_key, dry_run=ns.dry_run
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if ns.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"  W&B project provisioned: project={result['project']} "
            f"entity={result.get('entity', '?')} status={result['status']}"
        )
        if result.get("run_url"):
            print(f"    marker run: {result['run_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
