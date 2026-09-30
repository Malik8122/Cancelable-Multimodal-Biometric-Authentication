"""Synthetic-image fixtures.

None of these tests require a GPU, a trained checkpoint, or a downloaded
dataset - they exist to prove the preprocessing -> embedding interface is
correctly wired for every modality using generated placeholder images,
per the "local dummy verification" step described in docs/ROADMAP.md.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def test_master_secret(monkeypatch):
    """Every test runs with a fixed, obviously-fake MASTER_SECRET set.

    backend/config.py::Settings.master_secret has no default (a real deployment
    must supply its own secret) - this fixture is what lets `pytest` run with
    zero manual `.env` setup without hardcoding a secret anywhere in the
    importable source, only in test fixtures. autouse=True because any test
    that imports backend.config (even transitively) needs this set before
    Settings() is constructed.
    """
    monkeypatch.setenv("MASTER_SECRET", "unit-test-master-secret-not-for-production")
    # Per-modality scores are internal-only in production (DEBUG_SCORES unset).
    # Tests that assert on them opt in here; tests/test_multi_template.py
    # overrides this to prove the production (scores hidden) behaviour.
    monkeypatch.setenv("DEBUG_SCORES", "true")
    from backend.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def random_rgb_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def db_session():
    """A fresh, isolated in-memory SQLite session for one test.

    Independent of backend/database/session.py's cached, settings-derived
    engine - tests/test_database.py, test_services.py, and test_backend_api.py
    all use this instead so nothing here ever touches a real biometric.db
    file or leaks state between tests.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.database.models import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
