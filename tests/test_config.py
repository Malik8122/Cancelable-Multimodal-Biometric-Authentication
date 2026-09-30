"""Offline tests for backend/config.py.

Relies on the autouse `test_master_secret` fixture in tests/conftest.py to
supply MASTER_SECRET via the environment (never hardcoded in source).
"""

from __future__ import annotations

import pytest

from backend.config import Settings, get_settings


def test_settings_loads_master_secret_from_environment():
    settings = get_settings()
    assert settings.master_secret == "unit-test-master-secret-not-for-production"


def test_settings_has_sensible_defaults():
    settings = get_settings()
    assert settings.database_url == "sqlite:///./biometric.db"
    assert settings.application_id == "capstone-demo"
    assert settings.template_bits == 256
    assert 0.0 < settings.match_threshold <= 1.0


def test_settings_model_paths_default_to_embeddings_constants():
    from embeddings.constants import DEFAULT_CHECKPOINTS

    settings = get_settings()
    assert settings.face_model_path == DEFAULT_CHECKPOINTS["face"]
    assert settings.voice_model_path == DEFAULT_CHECKPOINTS["voice"]
    assert not hasattr(settings, "fingerprint_model_path") and not hasattr(settings, "iris_model_path")


def test_settings_raises_without_master_secret(monkeypatch):
    monkeypatch.delenv("MASTER_SECRET", raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_resolved_database_url_defaults_to_database_url():
    settings = Settings(_env_file=None, master_secret="x")
    assert settings.database_path is None
    assert settings.resolved_database_url == settings.database_url == "sqlite:///./biometric.db"


def test_resolved_database_url_prefers_database_path_when_set():
    """Deployment sprint: DATABASE_PATH (a bare file path, e.g. Render's
    persistent disk mount) must take priority over DATABASE_URL, without
    changing DATABASE_URL's own default/behavior when DATABASE_PATH is
    unset (see the test above)."""
    settings = Settings(_env_file=None, master_secret="x", database_path="/var/data/biometric.db")
    assert settings.resolved_database_url == "sqlite:////var/data/biometric.db"
    assert settings.database_url == "sqlite:///./biometric.db"  # unaffected


def test_environment_defaults_to_development_and_is_not_production():
    settings = Settings(_env_file=None, master_secret="x")
    assert settings.environment == "development"
    assert settings.is_production is False


def test_environment_reads_from_env_variable_not_environment(monkeypatch):
    """Deployment sprint: the env var is named ENV (matching Render's
    convention), not ENVIRONMENT (pydantic-settings' default case-insensitive
    field-name match) - this pins the explicit `validation_alias="ENV"`."""
    monkeypatch.setenv("ENV", "production")
    settings = Settings(_env_file=None, master_secret="x")
    assert settings.environment == "production"
    assert settings.is_production is True
