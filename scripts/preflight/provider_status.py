"""scripts/preflight/provider_status.py — multi-provider liveness probe.

Pings each configured provider's auth endpoint, parses the response,
emits one CSV row per provider:

    provider,status,balance_usd,gpu_available

Status values: ``ok`` (auth + balance ≥ 0), ``no_credit`` (auth OK but
no remaining balance), ``unauth`` (token missing or rejected), ``error``
(network or unexpected response).

Each provider has its own ``_probe_*`` function so adding a new
backend is one PR. Probes are kept synchronous + ``httpx.Client`` so
the script is testable via :func:`respx` mocks without an event loop.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    status: str  # ok | no_credit | unauth | error
    balance_usd: float | None
    gpu_available: bool | None
    detail: str = ""

    def to_csv_row(self) -> list[str]:
        return [
            self.provider,
            self.status,
            "" if self.balance_usd is None else f"{self.balance_usd:.2f}",
            "" if self.gpu_available is None
            else ("true" if self.gpu_available else "false"),
        ]


# ── individual provider probes ─────────────────────────────────────


def _http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict | str]:
    """Single sync GET — returns (status_code, parsed_body_or_text).

    Lifted out of the probe functions so the test harness can monkey-
    patch this single seam instead of patching each `httpx.get`."""
    import httpx

    r = httpx.get(url, headers=headers, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def probe_vultr(token: str, timeout: float = DEFAULT_TIMEOUT) -> ProviderResult:
    code, body = _http_get(
        "https://api.vultr.com/v2/account",
        {"Authorization": f"Bearer {token}"},
        timeout,
    )
    if code == 401 or code == 403:
        return ProviderResult("vultr", "unauth", None, None, str(body))
    if code != 200:
        return ProviderResult("vultr", "error", None, None, f"http {code}")
    if not isinstance(body, dict):
        return ProviderResult("vultr", "error", None, None, "non-JSON")
    bal = body.get("account", {}).get("balance")
    if bal is None:
        return ProviderResult("vultr", "error", None, None, "missing balance")
    bal = float(bal)
    status = "ok" if bal > 0 else "no_credit"
    return ProviderResult("vultr", status, bal, None)


def probe_runpod(token: str, timeout: float = DEFAULT_TIMEOUT) -> ProviderResult:
    code, body = _http_get(
        "https://api.runpod.io/graphql",
        {"Authorization": f"Bearer {token}"},
        timeout,
    )
    if code in (401, 403):
        return ProviderResult("runpod", "unauth", None, None, str(body))
    if code >= 500:
        return ProviderResult("runpod", "error", None, None, f"http {code}")
    if isinstance(body, dict) and body.get("data"):
        bal = body["data"].get("myself", {}).get("balance")
        bal_f = float(bal) if bal is not None else None
        status = "ok" if bal_f and bal_f > 0 else "no_credit"
        return ProviderResult("runpod", status, bal_f, True)
    return ProviderResult("runpod", "ok", None, True)


def probe_digitalocean(
    token: str, timeout: float = DEFAULT_TIMEOUT
) -> ProviderResult:
    code, body = _http_get(
        "https://api.digitalocean.com/v2/account",
        {"Authorization": f"Bearer {token}"},
        timeout,
    )
    if code == 401:
        return ProviderResult("digitalocean", "unauth", None, None, str(body))
    if code != 200:
        return ProviderResult("digitalocean", "error", None, None, f"http {code}")
    if not isinstance(body, dict):
        return ProviderResult("digitalocean", "error", None, None, "non-JSON")
    return ProviderResult("digitalocean", "ok", None, True)


def probe_hf(token: str, timeout: float = DEFAULT_TIMEOUT) -> ProviderResult:
    code, body = _http_get(
        "https://huggingface.co/api/whoami-v2",
        {"Authorization": f"Bearer {token}"},
        timeout,
    )
    if code == 401 or code == 403:
        return ProviderResult("hf", "unauth", None, None, str(body))
    if code != 200:
        return ProviderResult("hf", "error", None, None, f"http {code}")
    return ProviderResult("hf", "ok", None, False)


def probe_wandb(token: str, timeout: float = DEFAULT_TIMEOUT) -> ProviderResult:
    code, body = _http_get(
        "https://api.wandb.ai/graphql",
        {"Authorization": f"Bearer api:{token}"},
        timeout,
    )
    if code in (401, 403):
        return ProviderResult("wandb", "unauth", None, None, str(body))
    if code >= 500:
        return ProviderResult("wandb", "error", None, None, f"http {code}")
    return ProviderResult("wandb", "ok", None, False)


def probe_openrouter(token: str, timeout: float = DEFAULT_TIMEOUT) -> ProviderResult:
    code, body = _http_get(
        "https://openrouter.ai/api/v1/auth/key",
        {"Authorization": f"Bearer {token}"},
        timeout,
    )
    if code == 401:
        return ProviderResult("openrouter", "unauth", None, None, str(body))
    if code != 200:
        return ProviderResult("openrouter", "error", None, None, f"http {code}")
    if not isinstance(body, dict):
        return ProviderResult("openrouter", "error", None, None, "non-JSON")
    data = body.get("data", body)
    bal = data.get("usage")
    limit = data.get("limit")
    if isinstance(bal, (int, float)) and isinstance(limit, (int, float)):
        remaining = max(0.0, float(limit) - float(bal))
        status = "ok" if remaining > 0 else "no_credit"
        return ProviderResult("openrouter", status, remaining, False)
    return ProviderResult("openrouter", "ok", None, False)


# ── registry ──────────────────────────────────────────────────────


PROBES: dict[str, tuple[str, Callable[[str, float], ProviderResult]]] = {
    "vultr": ("VULTR_API_KEY", probe_vultr),
    "runpod": ("RUNPOD_API_KEY", probe_runpod),
    "digitalocean": ("DIGITALOCEAN_API_TOKEN", probe_digitalocean),
    "hf": ("HF_TOKEN", probe_hf),
    "wandb": ("WANDB_API_KEY", probe_wandb),
    "openrouter": ("OPENROUTER_API_KEY", probe_openrouter),
}


def probe_all(
    *,
    env: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    only: list[str] | None = None,
) -> list[ProviderResult]:
    """Probe every configured provider whose env var is set.

    ``env`` defaults to ``os.environ``; tests pass a dict to keep the
    real shell environment out of the way.
    """
    env = env if env is not None else dict(os.environ)
    out: list[ProviderResult] = []
    for name, (env_var, fn) in PROBES.items():
        if only and name not in only:
            continue
        token = env.get(env_var)
        if not token:
            out.append(ProviderResult(name, "unauth", None, None, "no token"))
            continue
        try:
            out.append(fn(token, timeout))
        except Exception as exc:  # noqa: BLE001
            out.append(
                ProviderResult(name, "error", None, None, f"{type(exc).__name__}")
            )
    return out


def render_csv(results: list[ProviderResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["provider", "status", "balance_usd", "gpu_available"])
    for r in results:
        writer.writerow(r.to_csv_row())
    return buf.getvalue()


def render_json(results: list[ProviderResult]) -> str:
    return json.dumps(
        [
            {
                "provider": r.provider,
                "status": r.status,
                "balance_usd": r.balance_usd,
                "gpu_available": r.gpu_available,
                "detail": r.detail,
            }
            for r in results
        ],
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated provider names")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ns = parser.parse_args(argv)

    only = [s.strip() for s in ns.only.split(",")] if ns.only else None
    results = probe_all(timeout=ns.timeout, only=only)

    sys.stdout.write(
        render_json(results) + "\n" if ns.json else render_csv(results)
    )
    fail = sum(1 for r in results if r.status == "error")
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
