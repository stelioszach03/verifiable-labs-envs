"""Monthly Stripe metered-billing reconciliation.

Reads the diff between ``usage_counters.traces_count`` and the last
reported value, then reports the delta to Stripe via
``stripe.SubscriptionItem.create_usage_record`` for each Pro/Team
subscription whose tier has a metered overage price.

Stage C ships this as a **stub**: while ``VLABS_BILLING_ENABLED=false``
the job logs the deferral and exits cleanly. Once the Delaware C-corp
+ Stripe live mode are activated, fill in the body below and uncomment
the schedule entry in ``deploy/fly.toml``.

Run manually for a dry test::

    python -m vlabs_api.jobs.reconcile_overage
"""
from __future__ import annotations

import sys

import structlog

from vlabs_api.config import get_settings

log = structlog.get_logger(__name__)


def main() -> int:
    settings = get_settings()
    if not settings.vlabs_billing_enabled:
        log.info(
            "reconcile_overage.skipped",
            reason="billing deferred (VLABS_BILLING_ENABLED=false)",
            note=(
                "Re-enable in deploy/fly.toml + flip VLABS_BILLING_ENABLED=true "
                "via fly secrets set, after Delaware C-corp + Stripe live mode."
            ),
        )
        return 0

    log.warning(
        "reconcile_overage.not_implemented",
        note="Stage C ships a stub; metered overage reporting lands in Phase 17.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
