# Security Policy

## Supported versions

`verifiable-labs-envs` is in **alpha** (`v0.1.x`). Security fixes are issued on a best-effort basis to the latest `0.1.x` release only. There is no long-term support track yet; users on older alpha tags should upgrade to receive fixes.

| Version       | Supported            |
| ------------- | -------------------- |
| `0.1.x`       | ✅ best-effort       |
| `< 0.1.0`     | ❌ no                |

## Reporting a vulnerability

**Please do not open a public GitHub issue, pull request, or discussion to report a security vulnerability.** Public reports give attackers a head start and put downstream users at risk.

Instead, email **stelios@stelioszach.com** with:

- A description of the issue and the affected component (`src/verifiable_labs_envs`, `src/verifiable_labs_api`, `packages/verifiable-labs/`, hosted API endpoints, CI workflows, etc.).
- Steps to reproduce — minimal, deterministic, ideally a single Python snippet or `curl` invocation.
- Affected version (`pip show verifiable-labs-envs` or commit SHA).
- Your assessment of impact (confidentiality / integrity / availability).
- Whether you intend to publish a CVE or write-up, and on what timeline.

You should receive an acknowledgment within **72 hours**. We will work with you to confirm the issue, develop a fix, and coordinate disclosure. Credit is given to reporters in the release notes unless you ask to remain anonymous.

## Scope

In scope:
- The Python packages `verifiable_labs_envs`, `verifiable_labs_api`, and `verifiable-labs` (SDK).
- The `verifiable` CLI.
- Default configurations of the hosted FastAPI evaluation service.
- GitHub Actions workflows in this repository.

Out of scope:
- Issues that require physical access to a developer machine or compromised credentials.
- Vulnerabilities in third-party dataset providers (e.g., LoDoPaB-CT, DIV2K) unless triggered by a flaw in our integration code.
- Reproduction-quality issues in the paper or paper figures (please use regular GitHub issues for those).
- Denial-of-service achievable only through unbounded user-supplied tensor sizes against a self-hosted instance — operators are expected to apply their own rate limits.

## Disclosure timeline

Our default coordinated-disclosure window is **90 days** from the acknowledgment of a confirmed vulnerability, or until a fix ships in a tagged release, whichever is sooner. We are happy to negotiate an extension if a fix is genuinely complex; we will not extend silently.
