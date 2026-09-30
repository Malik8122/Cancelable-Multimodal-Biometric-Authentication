"""Minimal schema migration helper.

A capstone-scope SQLite backend doesn't warrant a full migration framework
(Alembic) - this is a deliberate scope reduction, not an oversight. What's
here covers the two things that actually matter for this project's lifetime:
creating the schema on first run (`ensure_schema`, a thin wrapper around
`backend.database.session.init_db`) and a defensive check that a freshly
created database's tables have the columns the current `models.py` expects,
which catches "someone changed models.py but didn't delete their local
biometric.db" during development.

If this project ever needs real migrations (e.g. a Postgres deployment with
existing production data), that's the point to adopt Alembic rather than
extend this file.
"""

from __future__ import annotations

from sqlalchemy import inspect

from backend.database.models import Base
from backend.database.session import get_engine, init_db


def ensure_schema() -> None:
    """Create any missing tables. Safe to call on every app startup."""
    init_db()


def schema_matches_models() -> bool:
    """True if every table/column `models.py` declares actually exists in the DB.

    Does not check for extra columns (a superset is fine) or type mismatches -
    just presence, which is enough to catch the common "stale local .db file"
    development mistake early with a clear error instead of a confusing
    SQLAlchemy exception mid-request.
    """
    inspector = inspect(get_engine())
    existing_tables = set(inspector.get_table_names())

    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            return False
        existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
        expected_columns = {column.name for column in table.columns}
        if not expected_columns.issubset(existing_columns):
            return False
    return True


# --- Multi-template architecture upgrade (docs/MULTI_TEMPLATE_ARCHITECTURE.md) -------------------
#
# `Base.metadata.create_all` never alters an existing table, so a database
# created before the template pool existed (a real Render/PostgreSQL or a
# local SQLite file) lacks the new columns. `upgrade_schema` adds them with
# plain `ALTER TABLE ... ADD COLUMN` (portable across SQLite and PostgreSQL)
# and backfills legacy rows. It is idempotent and runs on every startup.

from sqlalchemy import Engine, text  # noqa: E402


def _add_missing_columns(engine: Engine) -> list[str]:
    inspector = inspect(engine)
    added: list[str] = []
    with engine.begin() as connection:
        for table_name, table in Base.metadata.tables.items():
            if table_name not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                column_type = column.type.compile(dialect=engine.dialect)
                connection.execute(text(f'ALTER TABLE {table_name} ADD COLUMN {column.name} {column_type}'))
                added.append(f"{table_name}.{column.name}")
    return added


def _backfill_legacy_templates(engine: Engine) -> int:
    """Give pre-pool rows a lifecycle status. Returns how many rows were updated.

    A legacy row is one whose `template_status` is still NULL (the column was
    just added). The single active template becomes ACTIVE at pool position
    T1; an old deactivated one becomes REVOKED. No STANDBY templates can be
    invented here - generating one needs the user's embedding, which is
    never stored - so a legacy user has a pool of one until
    `POST /templates/{user_id}/replenish` (authenticated by a fresh capture)
    fills it.
    """
    with engine.begin() as connection:
        result = connection.execute(
            text(
                "UPDATE protected_templates SET "
                "template_status = CASE WHEN is_active THEN 'ACTIVE' ELSE 'REVOKED' END, "
                "template_index = COALESCE(template_index, 1), "
                "template_group_id = COALESCE(template_group_id, template_id), "
                "activation_time = CASE WHEN is_active THEN COALESCE(activation_time, created_at) ELSE activation_time END, "
                "revoked_time = CASE WHEN is_active THEN revoked_time ELSE COALESCE(revoked_time, created_at) END, "
                "revoked_reason = CASE WHEN is_active THEN revoked_reason ELSE COALESCE(revoked_reason, 'legacy: replaced') END "
                "WHERE template_status IS NULL"
            )
        )
        return result.rowcount or 0


def _backfill_template_sets(engine: Engine) -> int:
    """Group legacy rows into template sets and give them set-level status. Returns rows updated.

    Handles both older shapes (`template_set_version IS NULL`):

    - pre-pool single templates: `template_index` is 1, so every user's
      active templates form set v1;
    - the per-modality pools of the previous phase: `template_index` (T1..TN)
      becomes the set version. If modalities had drifted apart (one revoked
      further than another), the ACTIVE set is the highest version any
      modality was ACTIVE at; each modality's row at that version becomes
      ACTIVE, its older live rows are REVOKED (`migrated to set-level
      activation`), and newer live rows are STANDBY. A modality with no live
      row at that version keeps its own ACTIVE row and `exactly_one_active_set`
      security validation will flag the context rather than silently mixing sets.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from backend.database.models import STATUS_ACTIVE, STATUS_REVOKED, STATUS_STANDBY, ProtectedTemplate

    updated = 0
    with Session(engine) as db:
        legacy = list(db.execute(select(ProtectedTemplate).where(ProtectedTemplate.template_set_version.is_(None))).scalars())
        contexts = {(row.user_id, row.application_id) for row in legacy}
        for user_id, application_id in contexts:
            rows = [r for r in legacy if r.user_id == user_id and r.application_id == application_id]
            for row in rows:
                row.template_set_version = row.template_index or 1
            active_versions = [r.template_set_version for r in rows if r.template_status == STATUS_ACTIVE]
            active_version = max(active_versions) if active_versions else None
            for row in rows:
                if active_version is not None and row.template_status != STATUS_REVOKED:
                    if row.template_set_version == active_version:
                        row.template_status, row.is_active = STATUS_ACTIVE, True
                    elif row.template_set_version < active_version:
                        row.template_status, row.is_active = STATUS_REVOKED, False
                        row.revoked_reason = row.revoked_reason or "migrated to set-level activation"
                        row.revoked_time = row.revoked_time or row.created_at
                    else:
                        row.template_status, row.is_active = STATUS_STANDBY, False
            by_version: dict[int, list] = {}
            for row in rows:
                by_version.setdefault(row.template_set_version, []).append(row)
            for version, members in by_version.items():
                live = [m for m in members if m.template_status != STATUS_REVOKED]
                status = (
                    STATUS_REVOKED
                    if not live
                    else STATUS_ACTIVE if any(m.template_status == STATUS_ACTIVE for m in live) else STATUS_STANDBY
                )
                created = min(m.created_at for m in members)
                for member in members:
                    member.template_index = version
                    member.template_set_status = status
                    member.template_set_created_at = created
                    member.template_set_activated_at = (
                        next((m.activation_time for m in members if m.activation_time), created) if status != STATUS_STANDBY else None
                    )
                    member.template_set_revoked_at = (
                        max((m.revoked_time or created) for m in members) if status == STATUS_REVOKED else None
                    )
            updated += len(rows)
        db.commit()
    return updated


def _ensure_indexes(engine: Engine) -> None:
    """`create_all` never adds an index to an already-existing table - add the set index ourselves."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_one_live_template_per_set_modality "
                "ON protected_templates (user_id, application_id, template_set_version, modality) "
                "WHERE template_status <> 'REVOKED'"
            )
        )


#: Modalities removed from the system (iris and fingerprint). Their templates, enrollment events and the
#: audit column that held fingerprint scores are deleted by `_remove_retired_modalities`.
RETIRED_MODALITIES = ("fingerprint", "iris")
_RETIRED_COLUMNS = (("audit_logs", "fingerprint_similarity"),)


def _remove_retired_modalities(engine: Engine) -> dict[str, object]:
    """Delete data that belongs only to a retired modality. Face and voice rows are never touched.

    Removes the retired modalities' protected templates and enrollment events, and drops the audit-log
    column that stored fingerprint similarity (`ALTER TABLE ... DROP COLUMN` works on SQLite >= 3.35
    and PostgreSQL). Existing audit rows are kept: their JSON fields are a historical record of what
    was attempted. Idempotent.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    placeholders = ", ".join(f":m{i}" for i in range(len(RETIRED_MODALITIES)))
    params = {f"m{i}": modality for i, modality in enumerate(RETIRED_MODALITIES)}
    deleted: dict[str, int] = {}
    dropped: list[str] = []
    with engine.begin() as connection:
        for table_name in ("protected_templates", "enrollment_events"):
            if table_name in tables:
                result = connection.execute(
                    text(f"DELETE FROM {table_name} WHERE modality IN ({placeholders})"), params
                )
                deleted[table_name] = result.rowcount or 0
        for table_name, column_name in _RETIRED_COLUMNS:
            if table_name in tables and column_name in {c["name"] for c in inspector.get_columns(table_name)}:
                connection.execute(text(f"ALTER TABLE {table_name} DROP COLUMN {column_name}"))
                dropped.append(f"{table_name}.{column_name}")
    return {"rows_deleted": deleted, "columns_dropped": dropped}


def upgrade_schema(engine: Engine | None = None) -> dict[str, object]:
    """Bring an existing database up to the template-set schema. Safe to re-run."""
    engine = engine or get_engine()
    Base.metadata.create_all(bind=engine)
    retired = _remove_retired_modalities(engine)
    added = _add_missing_columns(engine)
    backfilled = _backfill_legacy_templates(engine)
    sets_backfilled = _backfill_template_sets(engine)
    _ensure_indexes(engine)
    return {
        "columns_added": added,
        "legacy_templates_backfilled": backfilled,
        "template_set_rows_backfilled": sets_backfilled,
        "retired_modalities_removed": retired,
    }
