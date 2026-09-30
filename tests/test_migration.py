"""Offline tests for backend/database/migration.py.

Uses backend.database.session's lazily-created engine, pointed at a
temporary SQLite file via DATABASE_URL, so it exercises the real
get_engine()/init_db() path (unlike test_database.py's fully independent
in-memory engine).
"""

from __future__ import annotations

from pathlib import Path

from backend.database import migration
from backend.database.session import get_engine, get_session_factory


def _reset_caches():
    get_engine.cache_clear()
    get_session_factory.cache_clear()


def test_ensure_schema_creates_tables_and_matches_models(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    from backend.config import get_settings

    get_settings.cache_clear()
    _reset_caches()

    try:
        assert migration.schema_matches_models() is False
        migration.ensure_schema()
        assert migration.schema_matches_models() is True
    finally:
        get_settings.cache_clear()
        _reset_caches()
