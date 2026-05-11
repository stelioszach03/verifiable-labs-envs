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
from typing import Any

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


def _http_post(
    url: str, headers: dict[str, str], json_body: dict, timeout: float
) -> tuple[int, dict | str]:
    """Single sync POST — returns (status_code, parsed_body_or_text).

    Used by the W&B probe (``/graphql`` is POST-only). Mirror seam
    of :func:`_http_get` for the test harness."""
    import httpx

    r = httpx.post(url, headers=headers, json=json_body, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def probe_vultr(token: str, timeout: float = DEFAULT_TIMEOUT) -> ProviderResult:
    """Vultr's accounting uses a SIGNED ``balance`` field:

    - ``balance < 0``: user has credit. The absolute value is the
      remaining USD credit. (Vultr models this as "negative debt" so
      negative numbers === money the platform owes the user.)
    - ``balance == 0``: account paid in full, no remaining credit.
    - ``balance > 0``: user owes Vultr money (charges minus payments
      exceed credit). No credit available for new provisioning.

    The previous probe inverted this and reported ``no_credit`` for
    accounts that actually had $255 of credit available. Now we
    report ``balance_usd`` as the positive remaining-credit number
    (signed in: ``-bal`` when ``bal < 0``).
    """
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
    # Negative = credit available; positive = debt owed → no credit.
    remaining_credit = -bal if bal < 0 else 0.0
    status = "ok" if remaining_credit > 0 else "no_credit"
    return ProviderResult("vultr", status, remaining_credit, None)


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
    """Auth-ping wandb.ai. The ``/graphql`` endpoint is POST-only and
    expects a query body, so a bare GET returns 4xx regardless of the
    auth header. Sends a minimal ``{ viewer { username } }`` GraphQL
    POST and tries both the modern ``Bearer {token}`` scheme (used by
    Personal Access Tokens that start with ``wandb_``) and the legacy
    ``Bearer api:{token}`` scheme (40-char hex API keys)."""
    last_body: dict | str = ""
    for header_value in (f"Bearer {token}", f"Bearer api:{token}"):
        code, body = _http_post(
            "https://api.wandb.ai/graphql",
            {"Authorization": header_value},
            {"query": "{ viewer { username } }"},
            timeout,
        )
        last_body = body
        if code == 200:
            return ProviderResult("wandb", "ok", None, False)
        if code >= 500:
            return ProviderResult("wandb", "error", None, None, f"http {code}")
    # Both schemes returned 4xx other than 200 — auth failed.
    return ProviderResult("wandb", "unauth", None, None, str(last_body))


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


_ORACLE_GPU_LIMIT_NAMES: tuple[str, ...] = (
    "bm-gpu2-2-count",          # 2× P100 (Pascal, legacy)
    "bm-gpu3-8-count",          # 8× V100 (Volta)
    "bm-gpu4-8-count",          # 8× A100 40 GB
    "bm-gpu-a100-v2-count",     # 8× A100 80 GB BM
    "bm-gpu-h100-8-count",      # 8× H100
    "bm-gpu-l40s-4-count",      # 4× L40S
    "vm-gpu-a10-1-count",       # 1× A10 24 GB
    "vm-gpu-a10-2-count",       # 2× A10
    "vm-gpu-a100-1-count",      # 1× A100 single-VM
)


def _oracle_gpu_quota_summary(
    oci_module: Any,
    config: dict[str, str],
    timeout: float,
) -> tuple[bool, str]:
    """Query the OCI limits service for known GPU shape counts.

    Returns ``(any_available, detail_string)``. ``any_available`` is
    True iff at least one known GPU shape has ``available > 0`` in any
    availability domain. The detail string lists the non-zero quotas
    (when present) or is empty (when all zero).

    Wrapped in a helper so tests can mock the limits client without
    touching the auth path above it.
    """
    identity = oci_module.identity.IdentityClient(config)
    identity.base_client.timeout = timeout
    ads = identity.list_availability_domains(config["tenancy"]).data
    limits_client = oci_module.limits.LimitsClient(config)
    limits_client.base_client.timeout = timeout

    found: dict[str, int] = {}
    for limit_name in _ORACLE_GPU_LIMIT_NAMES:
        for ad in ads:
            try:
                v = limits_client.get_resource_availability(
                    service_name="compute",
                    limit_name=limit_name,
                    compartment_id=config["tenancy"],
                    availability_domain=ad.name,
                )
                avail = int(v.data.available or 0)
                if avail > 0:
                    found[f"{limit_name}@{ad.name.split(':')[-1]}"] = avail
            except oci_module.exceptions.ServiceError as exc:
                # InvalidParameter = OCI doesn't recognize the shape for
                # this region/tenancy. 404 = the limit isn't applicable.
                # Both are "no quota" — keep iterating.
                if exc.status in (400, 404):
                    continue
                # Other errors shouldn't block the auth-ok signal.
                continue

    if found:
        return True, "GPU quota OK: " + str(found)
    return False, (
        "auth ok but ZERO GPU quota across known shapes — "
        "file Service Limit Increase request via OCI Console "
        "(see docs/operational/oracle-sli-request.md)"
    )


def probe_oracle(token: str, timeout: float = DEFAULT_TIMEOUT) -> ProviderResult:
    """Oracle Cloud Infrastructure (OCI) probe.

    Two-stage check:

      1. **Auth** — signed-request `get_user(user_ocid)` confirms the
         private key + fingerprint + OCID quintet are valid. A 200
         here means we CAN talk to OCI; nothing more.

      2. **GPU quota** — query the limits service for known GPU shape
         counts. A trial account with valid auth still has every GPU
         quota at zero by default — the user has to file a Service
         Limit Increase (SLI) request and wait 24-72 h. The probe
         now distinguishes:

           gpu_available=True   → at least one GPU shape has avail>0
           gpu_available=False  → auth OK but zero quota anywhere

         This catches the "looked-fine-but-actually-no-GPU" trap.
    """
    del token  # not used directly — the real auth pulls OCID+keypair from env

    tenancy = os.environ.get("ORACLE_TENANCY_OCID", "").strip()
    user = os.environ.get("ORACLE_USER_OCID", "").strip()
    fingerprint = os.environ.get("ORACLE_FINGERPRINT", "").strip()
    key_path = os.environ.get("ORACLE_PRIVATE_KEY_PATH", "").strip()
    region = os.environ.get("ORACLE_REGION", "us-ashburn-1").strip() or "us-ashburn-1"

    have_full_keypair = all([tenancy, user, fingerprint, key_path])
    if not have_full_keypair:
        return ProviderResult(
            "oracle",
            "unauth",
            None,
            None,
            "missing one or more of ORACLE_{TENANCY,USER,FINGERPRINT,PRIVATE_KEY_PATH}",
        )

    # ── try the real signed-request path ──────────────────────────
    try:
        import oci  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        return ProviderResult(
            "oracle",
            "ok",
            None,
            True,  # OCI does offer GPU instances
            "config-shape OK; install ``oci`` python sdk for live verification",
        )

    try:
        from pathlib import Path as _Path  # noqa: PLC0415

        key_file = _Path(key_path).expanduser()
        if not key_file.exists():
            return ProviderResult(
                "oracle",
                "unauth",
                None,
                None,
                f"private key not found at {key_path}",
            )
        config = {
            "user": user,
            "fingerprint": fingerprint,
            "tenancy": tenancy,
            "region": region,
            "key_file": str(key_file),
        }
        oci.config.validate_config(config)
        client = oci.identity.IdentityClient(config)
        client.base_client.timeout = timeout
        resp = client.get_user(user)
        if resp.status != 200:
            return ProviderResult(
                "oracle", "error", None, None, f"identity api http {resp.status}"
            )

        # Auth OK — now check GPU quota. Without this distinction the
        # probe lied for weeks: trial accounts with valid keypairs but
        # zero GPU allowance still got reported as ``ok, gpu_available
        # = True``, masking the SLI requirement.
        try:
            has_quota, detail = _oracle_gpu_quota_summary(oci, config, timeout)
        except Exception as exc:  # noqa: BLE001
            # Quota probe is best-effort: if it crashes (network blip,
            # SDK quirk), fall back to "auth ok, gpu unknown" rather
            # than turning the whole probe red.
            return ProviderResult(
                "oracle",
                "ok",
                None,
                None,
                f"auth ok; gpu quota probe failed: {type(exc).__name__}",
            )
        return ProviderResult("oracle", "ok", None, has_quota, detail)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "authentication" in msg or "401" in msg or "not authorized" in msg:
            return ProviderResult(
                "oracle", "unauth", None, None, f"{type(exc).__name__}: {exc}"
            )
        return ProviderResult(
            "oracle", "error", None, None, f"{type(exc).__name__}: {exc}"
        )


# ── registry ──────────────────────────────────────────────────────


PROBES: dict[str, tuple[str, Callable[[str, float], ProviderResult]]] = {
    "vultr": ("VULTR_API_KEY", probe_vultr),
    "runpod": ("RUNPOD_API_KEY", probe_runpod),
    "digitalocean": ("DIGITALOCEAN_API_TOKEN", probe_digitalocean),
    "hf": ("HF_TOKEN", probe_hf),
    "wandb": ("WANDB_API_KEY", probe_wandb),
    "openrouter": ("OPENROUTER_API_KEY", probe_openrouter),
    # Oracle's auth ladder is OCID + signed keypair, not a single
    # bearer token. The probe reads the 5 OCID env vars directly from
    # os.environ. We gate on ``ORACLE_TENANCY_OCID`` (the most
    # commonly-set OCID field) rather than ``ORACLE_CLI_AUTH_TOKEN``
    # — auth tokens are an alternative path and most users skip them,
    # whereas the OCID quintet is what every real-world OCI setup
    # provides. The actual auth check happens inside probe_oracle().
    "oracle": ("ORACLE_TENANCY_OCID", probe_oracle),
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
