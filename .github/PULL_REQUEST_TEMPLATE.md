<!--
Thanks for sending a PR! Please complete the sections below.
For non-trivial changes, link the issue this PR closes (e.g. "Closes #123").
-->

## Summary

<!-- One-paragraph description of what this PR does and why. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature / new environment (non-breaking)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Docs / paper / CI only (no functional code change)

## Checklist

- [ ] Tests added or updated for the change.
- [ ] `pytest -q` passes locally.
- [ ] `ruff check .` passes locally with no new warnings.
- [ ] Docs / `README.md` / `CHANGELOG.md` updated if user-visible behavior changed.
- [ ] No secrets, API keys, or local paths committed (`git diff --stat` reviewed).
- [ ] Branched from latest `main`; commits are focused and squashable.
- [ ] If this PR adds a new environment: it ships with a classical-baseline numerical sanity test and a forward-operator round-trip test.
- [ ] If this PR touches the paper or paper figures: regenerated artifacts are committed and the PDF builds cleanly.

## How to test

<!--
The exact commands a reviewer can run to verify the change end-to-end.
"Run pytest" is not enough — point at the specific test file or CLI invocation.
-->

## Additional notes

<!-- Anything reviewers should know up front: trade-offs, follow-ups, related issues. -->
