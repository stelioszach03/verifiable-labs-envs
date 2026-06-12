"""scripts/publish/list_packages.py — enumerate in-repo packages +
their PyPI vs Test-PyPI state.

Output is a fixed-width table. No tokens needed; all queries are
to the public JSON API.

Usage:
    python scripts/publish/list_packages.py            # both indexes
    python scripts/publish/list_packages.py --prod-only
    python scripts/publish/list_packages.py --test-only
    python scripts/publish/list_packages.py --json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Local helper import (avoids the package needing to be installed).
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_pypi_helpers", SCRIPT_DIR / "_pypi_helpers.py"
)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
assert spec.loader is not None
spec.loader.exec_module(helpers)


def _classify(local: str, latest: str | None, yanked_map: dict[str, bool]) -> str:
    """Return a short status word."""
    if latest is None:
        return "missing"
    if yanked_map.get(local, False):
        return "local-yanked"
    if local == latest:
        return "matched"
    if _semver_gt(local, latest):
        return "ahead"
    return "behind"


def _semver_gt(a: str, b: str) -> bool:
    """Best-effort version compare (PEP 440 isn't fully implemented;
    we do a lexicographic numeric tuple comparison after dropping
    pre-release suffixes for the major/minor/patch sort)."""

    def _key(v: str) -> tuple[int, ...]:
        # Strip ``aN`` / ``bN`` / ``rcN`` / ``.devN`` so the numeric
        # tuple is comparable. Pre-release versions sort below the
        # release version in this approximation.
        head = v.split("a")[0].split("b")[0].split("rc")[0]
        head = head.split(".dev")[0]
        return tuple(int(p) for p in head.split(".") if p.isdigit())

    return _key(a) > _key(b)


def _build_row(pkg, *, check_prod: bool, check_test: bool) -> dict:
    row: dict = {
        "name": pkg.name,
        "local_version": pkg.version,
        "package_dir": str(pkg.package_dir.relative_to(helpers.REPO_ROOT)),
    }
    if check_prod:
        try:
            versions = helpers.query_pypi_versions(
                pkg.name, index=helpers.PYPI_INDEX_URL
            )
            yanked = helpers.query_pypi_yanked_state(
                pkg.name, index=helpers.PYPI_INDEX_URL
            )
            row["pypi_versions"] = versions
            row["pypi_latest"] = versions[-1] if versions else None
            row["pypi_status"] = _classify(
                pkg.version, row["pypi_latest"], yanked
            )
            row["pypi_yanked_count"] = sum(1 for v in yanked.values() if v)
        except RuntimeError as exc:
            row["pypi_error"] = str(exc)
    if check_test:
        try:
            versions = helpers.query_pypi_versions(
                pkg.name, index=helpers.TEST_PYPI_INDEX_URL
            )
            yanked = helpers.query_pypi_yanked_state(
                pkg.name, index=helpers.TEST_PYPI_INDEX_URL
            )
            row["test_pypi_versions"] = versions
            row["test_pypi_latest"] = versions[-1] if versions else None
            row["test_pypi_status"] = _classify(
                pkg.version, row["test_pypi_latest"], yanked
            )
        except RuntimeError as exc:
            row["test_pypi_error"] = str(exc)
    return row


def _render_table(rows: list[dict], *, check_prod: bool, check_test: bool) -> str:
    cols = ["name", "local"]
    if check_prod:
        cols += ["pypi_latest", "pypi_status"]
    if check_test:
        cols += ["test_latest", "test_status"]
    widths = {"name": 44, "local": 12, "pypi_latest": 14, "pypi_status": 13,
              "test_latest": 14, "test_status": 13}
    lines: list[str] = []
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        cells = {
            "name": r["name"][: widths["name"]],
            "local": r["local_version"][: widths["local"]],
            "pypi_latest": (r.get("pypi_latest") or "—")[: widths["pypi_latest"]],
            "pypi_status": r.get("pypi_status", "—")[: widths["pypi_status"]],
            "test_latest": (r.get("test_pypi_latest") or "—")[: widths["test_latest"]],
            "test_status": r.get("test_pypi_status", "—")[: widths["test_status"]],
        }
        lines.append("  ".join(cells[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    ns = parser.parse_args(argv)

    check_prod = not ns.test_only
    check_test = not ns.prod_only

    pkgs = helpers.discover_packages()
    rows = [_build_row(p, check_prod=check_prod, check_test=check_test) for p in pkgs]

    if ns.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(f"# in-repo packages (excluding {sorted(helpers.EXCLUDED_PACKAGES)}):")
        print(f"# total: {len(pkgs)}")
        print()
        print(_render_table(rows, check_prod=check_prod, check_test=check_test))
        # Suggested actions for each status.
        if check_prod:
            ahead = [r for r in rows if r.get("pypi_status") == "ahead"]
            missing = [r for r in rows if r.get("pypi_status") == "missing"]
            print()
            if missing:
                print(
                    f"# {len(missing)} packages missing on prod PyPI: "
                    + ", ".join(r["name"] for r in missing)
                )
            if ahead:
                print(
                    f"# {len(ahead)} packages ahead of prod PyPI: "
                    + ", ".join(r["name"] for r in ahead)
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
