# How to contribute to Verifiable Labs

Thanks for your interest in `verifiable-labs-envs`. This project is in alpha and we welcome bug reports, environment-correctness fixes, new scientific environments, paper-reproduction patches, and SDK / CLI improvements. Please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before participating.

## Quick setup

```bash
git clone https://github.com/stelioszach03/verifiable-labs-envs.git
cd verifiable-labs-envs

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

If you also want the optional extras (FastAPI service, real LoDoPaB-CT data, baseline torch models, docs), install them too:

```bash
pip install -e ".[dev,api,baselines,docs]"
# omit ct-real unless you actually need real-patient CT — its dependency
# pins (dival + odl<1.0 + setuptools<70) are tight.
```

> **macOS note:** if your repo lives under `~/Documents` (iCloud Drive), put the venv outside of iCloud (e.g. `~/.venvs/verifiable-labs`) to avoid duplicate-resolution artifacts. See `docs/MACOS_ICLOUD_VENV.md`.

## Running tests and lint

```bash
# Full suite — must be green before you open a PR.
pytest -q

# Lint — must be clean before you open a PR.
ruff check .
```

The full suite finishes in well under a minute on a recent laptop. CI runs the same two commands on Ubuntu and macOS — match locally before pushing.

## Code style

We use **`ruff` defaults** (with a small repo-level config in `pyproject.toml`: `line-length = 100`, `target-version = "py311"`, lint rule sets `E`, `F`, `I`, `UP`, `B`, `SIM`). No custom formatters, no pre-commit hooks. Run `ruff check .` and `ruff check . --fix` to auto-fix import order and obvious issues.

Type hints are encouraged but not enforced. `mypy` is configured but not strict; failing strict-mode checks won't block a PR.

## Pull-request process

1. **Branch from `main`.** Keep one logical change per PR.
2. **Write a test.** Bug fixes need a regression test; new features need coverage of the happy path and at least one failure mode. New scientific environments must include a classical-baseline numerical sanity check.
3. **Run the gate locally:** `pytest -q && ruff check .` — both must pass.
4. **Open the PR** against `main`. Use the PR template that auto-loads. Reference the issue number if one exists.
5. **CI must be green** before merge. CI runs the same `pytest` + `ruff` on Ubuntu and macOS with Python 3.11.
6. **No `--no-verify` on commits.** No force-push to `main`. Tag releases will be cut by maintainers.

Small-PR norms: a focused diff is much easier to review than a sprawling refactor. If your change touches more than ~400 lines of non-generated code, consider splitting it.

## Issue triage

We use GitHub issues for everything except security vulnerabilities (see [`SECURITY.md`](SECURITY.md)).

- **Bug reports** — please use the bug-report issue template. Include a minimal reproduction, the package version (`pip show verifiable-labs-envs`) or commit SHA, your OS, and the full traceback if any.
- **Feature requests** — please use the feature-request template. Describe the *problem* you're hitting, not just the solution you have in mind. Include links to relevant prior art (papers, datasets, other RL benchmarks) where helpful.
- **Reproducibility issues with the paper** — open a regular bug report and tag it `paper`. Include the figure / table number and the seed you used.

Maintainers triage on a best-effort cadence; bumping a stale issue with new information is welcome.

## Contributor agreement

By submitting a contribution to this repository you agree to license your work under the terms of the [Apache License 2.0](LICENSE), the same license as the rest of the project. We do not require a separate CLA — opening a PR is taken as your assent. Please keep the project license in mind if you import third-party code; pull requests that introduce a more restrictive or incompatible license cannot be accepted.

## Contact

For coordination that doesn't fit a public issue (e.g., security disclosure, paper-collaboration discussions, downstream-use reports), email **stelios@stelioszach.com**.
