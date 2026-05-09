"""Tests for ``vlabs_reward_train.wandb_callback``."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from vlabs_reward_train.wandb_callback import (
    DEFAULT_MODE,
    DEFAULT_PROJECT,
    WandbHandle,
    has_wandb_credentials,
    init_wandb_run,
    is_wandb_available,
    log_calibration_card,
    log_metrics,
    wandb_run,
)


def test_default_mode_offline() -> None:
    """29.C contract: default mode is offline so CI doesn't need W&B."""
    assert DEFAULT_MODE == "offline"
    assert DEFAULT_PROJECT == "vlabs-reward-distillation"


def test_init_wandb_run_disabled_returns_noop() -> None:
    handle = init_wandb_run(mode="disabled")
    assert handle.is_real is False
    assert handle.mode == "disabled"


def test_init_wandb_run_falls_back_when_wandb_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When wandb isn't importable, the helper returns a no-op handle."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("forced unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    handle = init_wandb_run(mode="offline")
    assert handle.is_real is False


def test_init_wandb_run_raises_when_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="wandb not installed"):
        init_wandb_run(mode="offline", fallback_to_noop=False)


def test_log_metrics_noop_handle_is_safe() -> None:
    handle = WandbHandle(run=None)
    log_metrics(handle, step=0, metrics={"loss": 0.5})


def test_log_calibration_card_noop_handle_is_safe() -> None:
    handle = WandbHandle(run=None)
    log_calibration_card(handle, {"quantile": 0.087, "drift": 0.01})


def test_init_wandb_run_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """When wandb IS importable, the helper passes config straight through."""
    fake_run = MagicMock()
    fake_module = types.ModuleType("wandb")
    fake_module.init = MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    handle = init_wandb_run(project="my-project", config={"lr": 1e-4})
    assert handle.is_real is True
    fake_module.init.assert_called_once()  # type: ignore[attr-defined]
    handle.log({"loss": 0.42})
    fake_run.log.assert_called_once_with({"loss": 0.42})


def test_init_wandb_run_falls_back_when_init_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.ModuleType("wandb")

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    fake_module.init = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    handle = init_wandb_run()
    assert handle.is_real is False


def test_wandb_run_context_manager_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run = MagicMock()
    fake_module = types.ModuleType("wandb")
    fake_module.init = MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    with wandb_run(project="x") as handle:
        assert handle.is_real is True
    fake_run.finish.assert_called_once()


def test_log_metrics_calls_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_metrics(handle, step=10, metrics={"loss": 0.5})
    fake_run.log.assert_called_once_with({"step": 10, "loss": 0.5})


def test_log_calibration_card_prefixes_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run = MagicMock()
    handle = WandbHandle(run=fake_run)
    log_calibration_card(handle, {"quantile": 0.1, "drift": -0.02})
    fake_run.log.assert_called_once_with(
        {"calibration/quantile": 0.1, "calibration/drift": -0.02}
    )


def test_finish_swallows_errors() -> None:
    fake_run = MagicMock()
    fake_run.finish.side_effect = RuntimeError("boom")
    handle = WandbHandle(run=fake_run)
    handle.finish()  # must not raise


def test_is_wandb_available_default(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert is_wandb_available() is False


def test_has_wandb_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert has_wandb_credentials() is False
    monkeypatch.setenv("WANDB_API_KEY", "x")
    assert has_wandb_credentials() is True
    monkeypatch.setenv("WANDB_API_KEY", "  ")
    assert has_wandb_credentials() is False
