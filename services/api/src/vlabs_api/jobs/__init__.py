"""Scheduled jobs invoked by Fly.io scheduled-machines.

Each module under :mod:`vlabs_api.jobs` is run as ``python -m
vlabs_api.jobs.<name>`` on a schedule defined in ``deploy/fly.toml``.
Stage C ships with one job (``reconcile_overage``) stubbed while
Stripe is deferred.
"""
