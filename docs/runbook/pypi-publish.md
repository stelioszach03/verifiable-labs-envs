# PyPI publishing runbook

Procedure for refreshing every in-repo package on PyPI / Test PyPI.
Mirrors the deploy runbook's "discovery → silent-prompt → execute"
shape: tokens never appear in chat, no plaintext on the command line,
and the prod path is gated behind an explicit `--prod` flag because
**PyPI uploads are irreversible** (you can yank, you cannot delete).

## Pre-reqs

- A PyPI account at https://pypi.org/account/register/ (Stelios →
  `stelioszach03`, owner of the existing `vlabs-calibrate==0.1.0a1`).
- A Test PyPI account at https://test.pypi.org/account/register/.
  The same email is fine; **the accounts are separate** even if the
  username matches — register once on each.
- Two API tokens, generated with the **"upload"** scope:
  - https://pypi.org/manage/account/token/
  - https://test.pypi.org/manage/account/token/
  Pick scope = "Entire account" the first time; you can narrow to
  per-project tokens once each package exists on each index.

Tokens look like `pypi-AgEIcHlwaS5vcmcCJ...` (long base64). They're
write-once: the only thing PyPI shows you is the create-time view.
Lost tokens are rotated by deleting + re-creating.

## Toolkit overview

```
scripts/publish/
├── _load_pypi_secrets.sh      # silent-prompt loader (source it)
├── _pypi_helpers.py           # shared discovery + REST helpers
├── list_packages.py           # tabulate local vs PyPI vs Test PyPI
├── bump_versions.py           # patch-bump every pyproject.toml
├── publish.sh                 # build + upload (--list / --test / --prod / --dry-run)
└── yank_old_versions.py       # yank older versions on PyPI / Test PyPI
```

## Step-by-step

### 1. Inventory

```bash
python3 scripts/publish/list_packages.py
```

Output is a table of every in-repo package with three columns:
`local`, `pypi_latest`, `test_latest`. Status words:

- `matched` — local version equals the index's latest. Re-upload
  rejected; yank older versions or bump to a new version first.
- `ahead` — local is newer; safe to upload.
- `behind` — index has a newer version (rare; usually a manual mistake).
- `missing` — package never published to that index.

### 2. Bump versions (optional)

If most packages are at `matched` and you want a clean refresh, bump
the patch component on every package's `pyproject.toml` first:

```bash
# Show what would change.
python3 scripts/publish/bump_versions.py --dry-run

# Write to disk.
python3 scripts/publish/bump_versions.py --apply

# Or set every selected package to a specific version.
python3 scripts/publish/bump_versions.py --set 0.1.0a2 --apply

# Or limit to one package.
python3 scripts/publish/bump_versions.py \
    --package verifiable-labs-envs --set 0.2.0 --apply
```

The bump heuristic preserves pre-release suffixes:
- `0.1.0` → `0.1.1`
- `0.1.0a1` → `0.1.0a2`
- `1.0.0` → `1.0.1`

Commit the version bumps before publishing so the git tag matches
what's on PyPI.

### 3. Load tokens (silent prompts)

```bash
source scripts/publish/_load_pypi_secrets.sh
# Two prompts: PYPI_API_TOKEN, then TEST_PYPI_API_TOKEN.
# Press Enter on either to skip that one.
```

Both tokens stay in the current shell only. The loader writes a
chmod-600 stamp file at `~/.vlabs-secrets/last_pypi_load.txt` listing
which keys are present (NEVER the values).

To reload only one:

```bash
source scripts/publish/_load_pypi_secrets.sh --only test
source scripts/publish/_load_pypi_secrets.sh --only prod
```

### 4. Test PyPI dry-run (always do this first)

```bash
# Build only — no upload.
bash scripts/publish/publish.sh --dry-run --all

# Upload to test.pypi.org.
bash scripts/publish/publish.sh --test --all
```

Verify on https://test.pypi.org/project/<name>/ that each package
appears with the correct version. The script skips already-uploaded
versions automatically (`twine --skip-existing`).

To narrow to one package:

```bash
bash scripts/publish/publish.sh --test --package verifiable-labs-envs
```

Build artifacts land under `dist/publish/<name>/`. Build / twine logs
are next to them (`build.log`, `twine.log`).

### 5. Production upload

```bash
bash scripts/publish/publish.sh --prod --all
```

This is irreversible. After upload, each package shows the new
version on https://pypi.org/project/<name>/. Pinned downstream
installs see the new version on next `pip install --upgrade`.

### 6. Yank old versions (optional)

PyPI doesn't allow real deletion; yanking removes a version from
default `pip install` resolution while keeping it visible in history
+ accessible if explicitly pinned. Run `--dry-run` first; the prod
yank is irreversible from CLI (un-yank requires the PyPI UI).

```bash
# What would be yanked? Same view on test.pypi.
python3 scripts/publish/yank_old_versions.py --dry-run

# Yank older versions on test.pypi.
python3 scripts/publish/yank_old_versions.py --test

# Yank older versions on prod PyPI — start narrow, with confirms.
python3 scripts/publish/yank_old_versions.py --prod \
    --package verifiable-labs-envs --require-confirm
```

The `--require-confirm` flag adds a y/N prompt per version — use it
the first time you run a `--prod` yank.

## Safety guarantees

- Tokens never appear in chat, terminal echo, logs, or commit
  messages. All input via `read -srp`; passed to `twine` via the
  `TWINE_PASSWORD` env var.
- `~/.vlabs-secrets/last_pypi_load.txt` is chmod 600 and only lists
  WHICH tokens are loaded — never the values.
- `dist/publish/` is built fresh per package; not committed
  (gitignored under `dist/`).
- Production uploads always require an explicit `--prod` flag;
  default is `--test`.
- The yank script always defaults to `--dry-run`.

## Out of scope

- `vlabs-api` is excluded from publication — it's the deployed
  FastAPI service, not a redistributable library.
- Real deletion of a PyPI version (PyPI policy: yank-only).
- Per-project token issuance (use account-wide tokens until each
  package exists on each index).
