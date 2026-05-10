"""Tests for ``verifiable_labs_envs.process_reward.wandb_integration``."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from verifiable_labs_envs.process_reward.wandb_integration import (
    DEFAULT_PRM_PROJECT,
    WandbHandle,
    aggregate_step_loss_decomposition,
    init_prm_run,
    log_multi_task_balance,
    log_per_step_metrics,
    log_run_card,
)


def test_default_project_locked() -> None:
    """W&B project mirrors the docs: vlabs-prm-distillation."""
    assert DEFAULT_PRM_PROJECT == "vlabs-prm-distillation"


def test_init_prm_run_falls_back_to_noop_when_wandb_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("forced unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    handle = init_prm_run(mode="offline")
    assert handle.is_real is False


def test_log_per_step_metrics_noop_handle_safe() -> None:
    handle = WandbHandle(run=None)
    log_per_step_metrics(handle, step=0, per_step_loss=0.42)


def test_log_per_step_metrics_calls_wandb_with_prefix() -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_per_step_metrics(
        handle,
        step=10,
        per_step_loss=0.5,
        per_step_calibration_coverage=0.91,
        bon_lift_vs_phase29=0.07,
    )
    fake_run.log.assert_called_once()
    payload = fake_run.log.call_args[0][0]
    assert payload["prm/per_step_loss"] == pytest.approx(0.5)
    assert payload["prm/per_step_calibration_coverage"] == pytest.approx(0.91)
    assert payload["prm/bon_lift_vs_phase29"] == pytest.approx(0.07)
    assert payload["step"] == 10


def test_log_per_step_metrics_omits_optional_fields_when_none() -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_per_step_metrics(handle, step=0, per_step_loss=0.5)
    payload = fake_run.log.call_args[0][0]
    assert "prm/per_step_calibration_coverage" not in payload
    assert "prm/bon_lift_vs_phase29" not in payload


def test_log_per_step_metrics_passes_extras() -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_per_step_metrics(
        handle, step=0, per_step_loss=0.5, extra={"something_else": 0.9}
    )
    payload = fake_run.log.call_args[0][0]
    assert payload["prm/something_else"] == pytest.approx(0.9)


def test_log_multi_task_balance_aggregates_components() -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_multi_task_balance(
        handle,
        step=5,
        step_loss_component=0.4,
        outcome_loss_component=0.1,
        per_step_weight=0.7,
        outcome_weight=0.3,
    )
    payload = fake_run.log.call_args[0][0]
    assert payload["prm/multi_task/step_loss_component"] == pytest.approx(0.4)
    assert payload["prm/multi_task/total_loss"] == pytest.approx(0.5)
    assert payload["prm/multi_task/step_share"] == pytest.approx(0.8)


def test_log_multi_task_balance_handles_zero_total() -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_multi_task_balance(
        handle,
        step=0,
        step_loss_component=0.0,
        outcome_loss_component=0.0,
        per_step_weight=0.7,
        outcome_weight=0.3,
    )
    payload = fake_run.log.call_args[0][0]
    assert payload["prm/multi_task/step_share"] == pytest.approx(0.0)


def test_log_run_card_noop_handle_safe() -> None:
    handle = WandbHandle(run=None)
    log_run_card(
        handle,
        config={"lr": 1e-4},
        dependencies={"is_satisfied": True},
        multi_task={"enable": False},
    )


def test_log_run_card_flattens_keys() -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_run_card(
        handle,
        config={"lr": 1e-4, "epochs": 3},
        dependencies={"is_satisfied": True},
        multi_task={"enable": True, "outcome_weight": 0.3},
    )
    payload = fake_run.log.call_args[0][0]
    assert "config/lr" in payload
    assert "deps/is_satisfied" in payload
    assert "multi_task/enable" in payload


# ── aggregate_step_loss_decomposition ──────────────────────────────


def test_aggregate_step_loss_decomposition_basic() -> None:
    summary = aggregate_step_loss_decomposition([0.1, 0.5, 0.2])
    assert summary["mean"] == pytest.approx((0.1 + 0.5 + 0.2) / 3)
    assert summary["max"] == pytest.approx(0.5)
    assert summary["argmax"] == 1.0
    assert summary["n"] == 3.0


def test_aggregate_step_loss_decomposition_empty() -> None:
    summary = aggregate_step_loss_decomposition([])
    assert summary == {"mean": 0.0, "max": 0.0, "argmax": 0.0, "n": 0.0}


def test_aggregate_step_loss_decomposition_single() -> None:
    summary = aggregate_step_loss_decomposition([0.7])
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["argmax"] == 0.0
