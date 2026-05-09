"""Schema-shape tests for Phase 29.E Alembic migration 0006.

Verifies the migration script is well-formed: revision identifiers,
table creation, indexes, check constraints, downgrade reversal, and
ORM-class alignment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "services" / "api" / "migrations" / "versions"


def _load_migration():
    path = MIGRATIONS / "0006_add_reward_models.py"
    spec = importlib.util.spec_from_file_location(
        "_alembic_06_reward_models", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_revision_chain_from_0005() -> None:
    m = _load_migration()
    assert m.revision == "0006"
    assert m.down_revision == "0005"
    assert m.branch_labels is None
    assert m.depends_on is None


def test_upgrade_downgrade_callable_present() -> None:
    m = _load_migration()
    assert callable(m.upgrade)
    assert callable(m.downgrade)


def test_migration_creates_two_tables() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    for tbl in ("reward_models", "reward_model_runs"):
        assert f'create_table(\n        "{tbl}"' in text, (
            f"missing create_table for {tbl}"
        )


def test_migration_extends_usage_counters() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    assert 'add_column(\n        "usage_counters"' in text
    assert "reward_scores_count" in text


def test_migration_creates_expected_indexes() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    for idx in (
        "reward_models_family_idx",
        "reward_models_status_idx",
        "reward_model_runs_user_idx",
        "reward_model_runs_model_idx",
        "reward_model_runs_idempotency_idx",
    ):
        assert idx in text, f"missing index {idx}"


def test_migration_creates_status_check_constraint() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    assert "reward_models_status_check" in text
    assert "'training','available','deprecated','retired'" in text


def test_downgrade_drops_in_reverse_dependency_order() -> None:
    """Downgrade order: column → indexes → reward_model_runs → indexes →
    reward_models. Reverse of upgrade so FKs unwind cleanly."""
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    drop_col_pos = text.index('drop_column("usage_counters"')
    drop_runs_pos = text.index('drop_table("reward_model_runs"')
    drop_models_pos = text.index('drop_table("reward_models"')
    assert drop_col_pos < drop_runs_pos < drop_models_pos


def test_orm_classes_match_migration() -> None:
    """ORM classes :class:`RewardModel` and :class:`RewardModelRun` are
    importable + listed in ``__all__``."""
    from vlabs_api import db as orm

    assert orm.RewardModel.__tablename__ == "reward_models"
    assert orm.RewardModelRun.__tablename__ == "reward_model_runs"
    assert "RewardModel" in orm.__all__
    assert "RewardModelRun" in orm.__all__


def test_orm_usage_counter_has_reward_scores_count() -> None:
    from vlabs_api.db import UsageCounter

    cols = {c.name for c in UsageCounter.__table__.columns}
    assert "reward_scores_count" in cols


def test_migration_uses_jsonb_for_eval_metrics() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    # eval_metrics + training_config both use postgresql.JSONB().
    assert '"eval_metrics", postgresql.JSONB()' in text
    assert '"training_config",\n            postgresql.JSONB()' in text


def test_migration_uses_uuid_pk_with_gen_random_uuid() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    assert text.count("gen_random_uuid()") == 2  # one per table


def test_migration_partial_unique_idempotency_index() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    assert (
        'postgresql_where=sa.text("idempotency_key IS NOT NULL")'
        in text
    )


def test_status_check_constraint_includes_all_four_states() -> None:
    text = (MIGRATIONS / "0006_add_reward_models.py").read_text(encoding="utf-8")
    for status in ("'training'", "'available'", "'deprecated'", "'retired'"):
        assert status in text
