"""Schema-shape tests for Phase 30.E Alembic migration 0007."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "services" / "api" / "migrations" / "versions"


def _load_migration():
    path = MIGRATIONS / "0007_add_process_reward_models.py"
    spec = importlib.util.spec_from_file_location(
        "_alembic_07_process_reward_models", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_revision_chain_from_0006() -> None:
    m = _load_migration()
    assert m.revision == "0007"
    assert m.down_revision == "0006"
    assert m.branch_labels is None
    assert m.depends_on is None


def test_upgrade_downgrade_callable_present() -> None:
    m = _load_migration()
    assert callable(m.upgrade)
    assert callable(m.downgrade)


def test_migration_creates_two_tables() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    for tbl in ("process_reward_models", "process_reward_model_runs"):
        assert f'create_table(\n        "{tbl}"' in text


def test_migration_extends_usage_counters() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    assert 'add_column(\n        "usage_counters"' in text
    assert "process_reward_scores_count" in text


def test_migration_creates_expected_indexes() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    for idx in (
        "process_reward_models_family_idx",
        "process_reward_models_status_idx",
        "process_reward_models_base_rm_idx",
        "process_reward_model_runs_user_idx",
        "process_reward_model_runs_model_idx",
        "process_reward_model_runs_idempotency_idx",
    ):
        assert idx in text


def test_migration_creates_status_check_constraint() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    assert "process_reward_models_status_check" in text
    assert "'training','available','deprecated','retired'" in text


def test_migration_creates_step_granularity_check() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    assert "process_reward_models_granularity_check" in text
    assert "'per_step','per_token','per_stage'" in text


def test_migration_links_base_rm_id_to_reward_models() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    assert '"base_rm_id"' in text
    assert 'sa.ForeignKey("reward_models.id"' in text


def test_downgrade_drops_in_reverse_order() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    drop_col = text.index('drop_column("usage_counters"')
    drop_runs = text.index('drop_table("process_reward_model_runs"')
    drop_models = text.index('drop_table("process_reward_models"')
    assert drop_col < drop_runs < drop_models


def test_orm_classes_match_migration() -> None:
    from vlabs_api import db as orm

    assert orm.ProcessRewardModel.__tablename__ == "process_reward_models"
    assert orm.ProcessRewardModelRun.__tablename__ == "process_reward_model_runs"
    assert "ProcessRewardModel" in orm.__all__
    assert "ProcessRewardModelRun" in orm.__all__


def test_orm_usage_counter_has_process_reward_scores_count() -> None:
    from vlabs_api.db import UsageCounter

    cols = {c.name for c in UsageCounter.__table__.columns}
    assert "process_reward_scores_count" in cols


def test_migration_uses_jsonb_for_step_columns() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    assert '"step_conformal_quantiles",\n            postgresql.JSONB()' in text
    assert '"step_rewards", postgresql.JSONB()' in text
    assert '"step_cis", postgresql.JSONB()' in text


def test_migration_uses_uuid_pk() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    assert text.count("gen_random_uuid()") == 2  # one per table


def test_migration_partial_unique_idempotency_index() -> None:
    text = (MIGRATIONS / "0007_add_process_reward_models.py").read_text(encoding="utf-8")
    assert (
        'postgresql_where=sa.text("idempotency_key IS NOT NULL")'
        in text
    )
