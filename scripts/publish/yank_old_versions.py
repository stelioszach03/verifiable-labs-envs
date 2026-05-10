"""scripts/publish/yank_old_versions.py — yank older versions of in-
repo packages on PyPI / Test PyPI.

PyPI **does not allow real deletion** of published versions; you can
only "yank" them, which removes them from default ``pip install``
resolution while keeping them visible in the package's history.
This is intentional PyPI policy to avoid breaking downstream pinned
installs.

This script:
  - enumerates every in-repo package (mirrors list_packages.py),
  - reads the local pyproject version,
  - queries PyPI for all published versions of that package,
  - for any version older than the local one that isn't already
    yanked, sends a yank request via PyPI's authenticated REST API
    (https://docs.pypi.org/legacy-api/).

Run --dry-run first; the prod path is irreversible (un-yanking is
manual via the PyPI UI).

Usage:
    source scripts/publish/_load_pypi_secrets.sh
    python scripts/publish/yank_old_versions.py --dry-run
    python scripts/publish/yank_old_versions.py --test
    python scripts/publish/yank_old_versions.py --prod \
            --package verifiable-labs-envs   # always start narrow
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_pypi_helpers", SCRIPT_DIR / "_pypi_helpers.py"
)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
assert spec.loader is not None
spec.loader.exec_module(helpers)


# Per the PyPI legacy upload API (https://upload.pypi.org/legacy/),
# yank is a POST to the project page with ``:action=yank_release``.
def _yank_endpoint(*, prod: bool) -> str:
    return (
        "https://pypi.org/manage/release/yank/"
        if prod
        else "https://test.pypi.org/manage/release/yank/"
    )


def _semver_lt(a: str, b: str) -> bool:
    def _key(v: str) -> tuple[int, ...]:
        head = v.split("a")[0].split("b")[0].split("rc")[0]
        head = head.split(".dev")[0]
        return tuple(int(p) for p in head.split(".") if p.isdigit())

    return _key(a) < _key(b)


def _yank_one(
    *, name: str, version: str, prod: bool, token: str, reason: str, timeout: float
) -> dict[str, Any]:
    """Send a yank request through PyPI's JSON HTTP API (Warehouse).

    Warehouse's documented yank flow is via the management UI; the
    legacy upload endpoint accepts a ``:action=yank_release`` POST
    with the project name + version + token. Returns a dict with
    ``ok`` + ``status`` + ``detail``.
    """
    import httpx

    upload_url = (
        "https://upload.pypi.org/legacy/"
        if prod
        else "https://test.pypi.org/legacy/"
    )
    try:
        r = httpx.post(
            upload_url,
            data={
                ":action": "yank_release",
                "name": name,
                "version": version,
                "yanked_reason": reason,
            },
            auth=("__token__", token),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": -1, "detail": f"network: {exc!r}"}
    return {
        "ok": r.status_code in (200, 201),
        "status": r.status_code,
        "detail": (r.text or "")[:300],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--test",
        action="store_true",
        help="Yank on test.pypi.org (uses TEST_PYPI_API_TOKEN).",
    )
    grp.add_argument(
        "--prod",
        action="store_true",
        help="Yank on pypi.org (uses PYPI_API_TOKEN). Irreversible.",
    )
    grp.add_argument(
        "--dry-run",
        action="store_true",
        help="List the versions that would be yanked; no API calls.",
    )
    parser.add_argument(
        "--package",
        default=None,
        help="Limit to one package (default: every in-repo package).",
    )
    parser.add_argument(
        "--reason",
        default="superseded by newer release",
        help="Yank reason shown on the PyPI version history page.",
    )
    parser.add_argument(
        "--timeout", type=float, default=20.0,
    )
    parser.add_argument(
        "--require-confirm",
        action="store_true",
        help=(
            "When set, prompt y/N before each yank (recommended on "
            "first --prod run)."
        ),
    )
    ns = parser.parse_args(argv)

    prod = ns.prod
    is_dry = ns.dry_run

    token: str | None = None
    index_label = ""
    if prod:
        token = helpers.get_token(prod=True)
        index_label = "pypi.org"
        if not token:
            helpers.err("ERROR: PYPI_API_TOKEN not set; load via _load_pypi_secrets.sh")
            return 2
    elif ns.test:
        token = helpers.get_token(prod=False)
        index_label = "test.pypi.org"
        if not token:
            helpers.err(
                "ERROR: TEST_PYPI_API_TOKEN not set; load via _load_pypi_secrets.sh"
            )
            return 2
    else:
        index_label = "(dry-run)"

    pkgs = helpers.discover_packages()
    if ns.package:
        pkgs = [p for p in pkgs if p.name == ns.package]
        if not pkgs:
            helpers.err(f"ERROR: --package {ns.package} not found in repo.")
            return 2

    print(f"# yank_old_versions  index={index_label}  packages={len(pkgs)}")
    print()

    total_targeted = 0
    total_yanked = 0
    total_skipped = 0
    total_failed = 0

    for p in pkgs:
        try:
            published = helpers.query_pypi_versions(
                p.name,
                index=helpers.PYPI_INDEX_URL if prod else helpers.TEST_PYPI_INDEX_URL,
            )
            yanked = helpers.query_pypi_yanked_state(
                p.name,
                index=helpers.PYPI_INDEX_URL if prod else helpers.TEST_PYPI_INDEX_URL,
            )
        except RuntimeError as exc:
            print(f"  ⚠ {p.name}: query failed — {exc}")
            continue

        if not published:
            print(f"  · {p.name}: not on {index_label}; nothing to yank")
            continue

        targets = [
            v for v in published
            if _semver_lt(v, p.version) and not yanked.get(v, False)
        ]
        already_yanked = sum(1 for v in published if yanked.get(v, False))

        print(
            f"  · {p.name} (local={p.version}; "
            f"{len(published)} published, {already_yanked} already yanked) "
            f"→ targets: {targets or 'none'}"
        )
        total_targeted += len(targets)

        if not targets:
            continue

        for v in targets:
            label = f"{p.name}=={v}"
            if is_dry:
                print(f"      [dry] would yank {label}")
                continue

            if ns.require_confirm:
                resp = input(f"      yank {label}? [y/N] ").strip().lower()
                if resp != "y":
                    print(f"      [skipped by user] {label}")
                    total_skipped += 1
                    continue

            res = _yank_one(
                name=p.name,
                version=v,
                prod=prod,
                token=token,  # type: ignore[arg-type]
                reason=ns.reason,
                timeout=ns.timeout,
            )
            if res["ok"]:
                print(f"      ✓ yanked {label}")
                total_yanked += 1
            else:
                print(
                    f"      ⚠ yank {label} failed "
                    f"(status={res['status']}, detail={res['detail'][:120]!r})"
                )
                total_failed += 1

    print()
    print(
        f"# summary  targeted={total_targeted}  yanked={total_yanked}  "
        f"skipped_by_user={total_skipped}  failed={total_failed}  "
        f"dry_run={is_dry}"
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
