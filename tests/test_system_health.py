"""Offline tests for GET /system/health.

The whole point of this endpoint is to be safe to poll frequently (the
frontend does, every 15s - frontend/src/hooks/useAuthSession.ts) on a
memory-constrained deployment. Before the fix this pass, it called
every modality's service getter (get_face_service() etc.)
unconditionally, forcing every real checkpoint (100+ MB each) to load
into memory on the very first poll - see backend/api/system.py's docstring
and docs/AUTHENTICATION_RELIABILITY_REPORT.md's Render OOM section. These
tests pin the fix: the endpoint must report status without ever calling a
not-yet-loaded modality's service getter.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from fastapi.testclient import TestClient


def _reset_caches():
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_backend.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    _reset_caches()

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client

    _reset_caches()


def test_system_health_never_instantiates_an_unloaded_model(client, monkeypatch):
    """The real regression test: swap in spy getters that raise if called,
    with cache_info().currsize == 0 (never-loaded), matching the real
    get_face_service/etc.'s state on a fresh process. If _model_status ever
    calls the getter for a not-yet-loaded modality, this fails loudly."""
    import backend.api.system as system_module

    calls = {"face": 0, "voice": 0}

    def _make_spy(modality: str):
        @lru_cache
        def spy():
            calls[modality] += 1
            raise AssertionError(f"{modality}'s model getter was called - /system/health must not instantiate models")

        return spy

    monkeypatch.setattr(
        system_module,
        "_MODALITY_GETTERS",
        {
            "face": (_make_spy("face"), "face_model_path"),
            "voice": (_make_spy("voice"), "voice_model_path"),
        },
    )

    response = client.get("/system/health")

    assert response.status_code == 200
    assert calls == {"face": 0, "voice": 0}


def test_system_health_reports_already_loaded_models_without_reloading(client, monkeypatch):
    """If a modality's service was already loaded by an earlier real request
    (cache_info().currsize == 1), the endpoint may report its real status -
    that's a cache hit, not a fresh load - and must do so without raising."""
    import backend.api.system as system_module

    class _FakePipeline:
        is_mock = False

    class _FakeService:
        pipeline = _FakePipeline()

    @lru_cache
    def already_loaded_getter():
        return _FakeService()

    already_loaded_getter()  # populate the cache, simulating a prior real request

    monkeypatch.setattr(
        system_module,
        "_MODALITY_GETTERS",
        {
            "face": (already_loaded_getter, "face_model_path"),
            "voice": (already_loaded_getter, "voice_model_path"),
        },
    )

    response = client.get("/system/health")

    assert response.status_code == 200
    body = response.json()
    assert body["face_model"] == "loaded"
    assert body["voice_model"] == "loaded"


def test_system_health_returns_the_expected_response_shape(client):
    """No model getter is monkeypatched here - this exercises the real,
    unmodified _MODALITY_GETTERS, proving the endpoint still returns 200 with
    every field the frontend (BackendStatusBanner, useAuthSession) expects.

    Doesn't assert "not loaded" specifically: get_face_service/etc.'s
    lru_cache is process-global, and other test modules (e.g.
    test_voice_backend_integration.py) legitimately call the real getters -
    whichever runs first in a given pytest session leaves that cache warm for
    every test after it. The two tests above already pin the not-loaded and
    already-loaded behaviors precisely, with the cache state controlled
    explicitly; this one only checks the response shape and values that hold
    regardless of load state.
    """
    response = client.get("/system/health")

    assert response.status_code == 200
    body = response.json()
    for field in (
        "backend",
        "database",
        "face_model",
        "voice_model",
        "template_protection",
        "fusion_policy",
        "thresholds_loaded",
        "audit_logging",
    ):
        assert field in body, f"missing expected field {field!r}"

    assert "fingerprint_model" not in body and "iris_model" not in body  # removed modalities
    assert body["backend"] == "online"
    assert body["database"] == "connected"
    valid_statuses = {"available (not loaded)", "mock (no checkpoint)", "loaded", "mock"}
    for modality_field in ("face_model", "voice_model"):
        assert body[modality_field] in valid_statuses, f"unexpected {modality_field} value: {body[modality_field]!r}"
