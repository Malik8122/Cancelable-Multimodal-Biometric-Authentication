"""End-to-end tests: does the real FastAPI app (`backend.main.app`, via its `lifespan()` hook)
actually refuse to start in States B/D, and actually start + report `key_continuity` correctly in
States A/C - against a real, file-backed SQLite database, not just the isolated functions.

Mirrors `tests/test_system_health.py`'s `client` fixture pattern (fresh `tmp_path` database per
test, engine/settings caches reset around it) rather than sharing it, since these tests need to
seed the database file *before* `TestClient(app)` triggers startup - something that fixture's
"always start empty" shape doesn't support directly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, MasterSecretFingerprint, ProtectedTemplate, User
from backend.secret_fingerprint import fingerprint_master_secret


def _reset_caches():
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


def _seed(db_path, *, with_template: bool, fingerprint_for_secret: str | None):
    """Create the schema and pre-populate it, independent of the app's own engine/session."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db:
        if with_template:
            db.add(User(id="U001"))
            db.add(
                ProtectedTemplate(
                    user_id="U001",
                    modality="face",
                    application_id="test-app",
                    template_version=1,
                    key_version=1,
                    output_bits=256,
                    protected_template=b"\x00" * 32,
                )
            )
        if fingerprint_for_secret is not None:
            db.add(MasterSecretFingerprint(id=1, fingerprint=fingerprint_master_secret(fingerprint_for_secret)))
        db.commit()
    engine.dispose()


@pytest.fixture
def app_client_factory(tmp_path, monkeypatch):
    """Returns a callable that starts `TestClient(app)` against a freshly-imported app pointed at
    `tmp_path`'s database, with the given `MASTER_SECRET`. Caller enters/exits the returned
    context manager so a startup failure (State B/D) surfaces at `__enter__`, exactly like it
    would for a real `uvicorn` process."""
    db_path = tmp_path / "test_key_continuity.db"

    def factory(master_secret: str):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        monkeypatch.setenv("MASTER_SECRET", master_secret)
        _reset_caches()

        import importlib

        import backend.main as main_module

        importlib.reload(main_module)
        return TestClient(main_module.app)

    yield db_path, factory
    _reset_caches()


def test_state_a_fresh_database_starts_and_reports_ok(app_client_factory):
    db_path, factory = app_client_factory
    _seed(db_path, with_template=False, fingerprint_for_secret=None)

    with factory("this-sessions-secret") as client:
        response = client.get("/system/health")
        assert response.status_code == 200
        assert response.json()["key_continuity"] == "ok"


def test_state_b_existing_templates_no_fingerprint_refuses_to_start(app_client_factory):
    """The exact real-world incident this mechanism exists for: a database with pre-existing
    protected templates and no recorded fingerprint. The app must fail to start rather than come
    up "healthy" and silently authenticate against the wrong key."""
    from backend.key_continuity import KeyContinuityError

    db_path, factory = app_client_factory
    _seed(db_path, with_template=True, fingerprint_for_secret=None)

    with pytest.raises(KeyContinuityError):
        with factory("a-freshly-generated-secret-that-may-or-may-not-be-the-original"):
            pytest.fail("TestClient should never have finished entering - startup must fail first")


def test_state_c_matching_fingerprint_starts_and_reports_ok(app_client_factory):
    db_path, factory = app_client_factory
    _seed(db_path, with_template=True, fingerprint_for_secret="the-real-secret")

    with factory("the-real-secret") as client:
        response = client.get("/system/health")
        assert response.status_code == 200
        assert response.json()["key_continuity"] == "ok"


def test_state_d_mismatched_fingerprint_refuses_to_start(app_client_factory):
    from backend.key_continuity import KeyContinuityError

    db_path, factory = app_client_factory
    _seed(db_path, with_template=True, fingerprint_for_secret="the-real-secret")

    with pytest.raises(KeyContinuityError):
        with factory("a-different-secret"):
            pytest.fail("TestClient should never have finished entering - startup must fail first")


def test_key_continuity_response_never_contains_secret_or_fingerprint_material(app_client_factory):
    """Requirement: never expose MASTER_SECRET or the fingerprint through the API. `key_continuity`
    must be one of the four short enum strings - nothing derived from the actual bytes."""
    db_path, factory = app_client_factory
    _seed(db_path, with_template=True, fingerprint_for_secret="the-real-secret")

    with factory("the-real-secret") as client:
        body = client.get("/system/health").json()
        assert body["key_continuity"] in {"ok", "no_fingerprint_recorded", "mismatch", "no_templates_yet"}
        # The full response body, serialized, must never contain the configured secret.
        assert "the-real-secret" not in str(body)
