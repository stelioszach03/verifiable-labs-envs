"""Idempotently create the Verifiable Labs Stripe test-mode products.

Run **once** with your test-mode secret key in env::

    cd services/api
    STRIPE_SECRET_KEY=sk_test_... python scripts/create_stripe_products.py

The script:
  1. Refuses to run unless ``STRIPE_SECRET_KEY`` starts with ``sk_test_``.
  2. Creates (or finds, by ``metadata.vlabs_product``) two recurring
     subscription products: ``vlabs_pro`` ($99/mo) and ``vlabs_team``
     ($499/mo).
  3. Creates two metered overage prices for traces beyond the included
     quota — $1.00 per 10K (Pro) and $0.40 per 10K (Team).
  4. Prints the resulting price IDs in ``.env``-paste-ready form.

Re-running is safe: existing products/prices are reused.
"""
from __future__ import annotations

import os
import sys

import stripe


def _require_test_mode() -> None:
    key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not key.startswith("sk_test_"):
        sys.exit(
            "STRIPE_SECRET_KEY must be a test-mode key (sk_test_...). "
            "Live mode is blocked until the Delaware C-corp registration."
        )
    stripe.api_key = key


def _find_or_create_product(slug: str, name: str, description: str) -> stripe.Product:
    existing = stripe.Product.list(active=True, limit=100)
    for prod in existing.auto_paging_iter():
        if prod.metadata.get("vlabs_product") == slug:
            return prod
    return stripe.Product.create(
        name=name,
        description=description,
        metadata={"vlabs_product": slug},
    )


def _find_or_create_price(
    product: str,
    slug: str,
    *,
    amount_cents: int,
    interval: str = "month",
    metered: bool = False,
) -> stripe.Price:
    existing = stripe.Price.list(product=product, active=True, limit=100)
    for price in existing.auto_paging_iter():
        if price.metadata.get("vlabs_price") == slug:
            return price
    kwargs: dict[str, object] = {
        "product": product,
        "currency": "usd",
        "unit_amount": amount_cents,
        "recurring": {"interval": interval},
        "metadata": {"vlabs_price": slug},
    }
    if metered:
        kwargs["recurring"] = {"interval": interval, "usage_type": "metered"}
        kwargs["billing_scheme"] = "per_unit"
    return stripe.Price.create(**kwargs)


def main() -> int:
    _require_test_mode()

    pro_prod = _find_or_create_product(
        slug="vlabs_pro",
        name="Verifiable Labs — Pro",
        description="1,000,000 traces/month + standard support.",
    )
    team_prod = _find_or_create_product(
        slug="vlabs_team",
        name="Verifiable Labs — Team",
        description="10,000,000 traces/month + priority support.",
    )

    pro_base = _find_or_create_price(
        pro_prod.id, "vlabs_pro_base", amount_cents=9900
    )
    team_base = _find_or_create_price(
        team_prod.id, "vlabs_team_base", amount_cents=49900
    )
    # Overage: $0.10 per 10K traces for Pro = 1 cent per 1K -> 1 cent per 1K
    pro_over = _find_or_create_price(
        pro_prod.id,
        "vlabs_pro_overage",
        amount_cents=10,
        metered=True,
    )
    team_over = _find_or_create_price(
        team_prod.id,
        "vlabs_team_overage",
        amount_cents=4,
        metered=True,
    )

    print("# Paste these into services/api/.env.local")
    print(f"STRIPE_PRICE_ID_PRO={pro_base.id}")
    print(f"STRIPE_PRICE_ID_TEAM={team_base.id}")
    print(f"STRIPE_PRICE_ID_PRO_OVERAGE={pro_over.id}")
    print(f"STRIPE_PRICE_ID_TEAM_OVERAGE={team_over.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
