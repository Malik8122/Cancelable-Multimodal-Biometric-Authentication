"""Offline tests for backend/main.py's deployment-facing behavior:
CORS origin resolution and production-mode Swagger/ReDoc disabling.

Deliberately tests `_resolve_cors_origins`/`_is_production` as plain
functions (reading `os.environ` directly, per their own docstrings) rather
than only through the module-level `app` object, since `app = FastAPI(...)`
evaluates `_is_production()` once at import time - re-testing that wiring
for a different `ENV` value needs a fresh module reload, which
`test_docs_are_disabled_in_a_freshly_imported_production_app` does exactly
once, carefully, to confirm the wiring itself (not just the function) is
correct end-to-end.
"""

from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_resolve_cors_origins_defaults_to_localhost_vite_dev_server(monkeypatch):
    monkeypatch.delenv("CORS_ORIGIN", raising=False)
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    from backend.main import _resolve_cors_origins

    assert _resolve_cors_origins() == ["http://localhost:5173"]


def test_resolve_cors_origins_reads_cors_allowed_origins_comma_separated(monkeypatch):
    monkeypatch.delenv("CORS_ORIGIN", raising=False)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173, http://localhost:3000")
    from backend.main import _resolve_cors_origins

    assert _resolve_cors_origins() == ["http://localhost:5173", "http://localhost:3000"]


def test_resolve_cors_origins_prefers_cors_origin_over_cors_allowed_origins(monkeypatch):
    """Deployment sprint: CORS_ORIGIN (singular, the deployment-facing name -
    "set this to your Vercel URL") must win when both are set."""
    monkeypatch.setenv("CORS_ORIGIN", "https://my-app.vercel.app")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173")
    from backend.main import _resolve_cors_origins

    assert _resolve_cors_origins() == ["https://my-app.vercel.app"]


def test_resolve_cors_origins_never_returns_a_wildcard(monkeypatch):
    monkeypatch.setenv("CORS_ORIGIN", "*")
    from backend.main import _resolve_cors_origins

    # Not a validation rule (an operator could still misconfigure this) -
    # just confirms nothing here silently substitutes a wildcard on its own.
    assert _resolve_cors_origins() == ["*"]  # whatever was configured, unmodified - no silent widening either


def test_is_production_false_by_default(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    from backend.main import _is_production

    assert _is_production() is False


def test_is_production_true_when_env_is_production(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    from backend.main import _is_production

    assert _is_production() is True


def test_docs_are_disabled_in_a_freshly_imported_production_app(monkeypatch):
    """End-to-end confirmation that `app = FastAPI(docs_url=None if
    _is_production() else "/docs", ...)` actually wires `_is_production()`'s
    result into the real app object, not just that the helper function
    itself returns the right boolean."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("MASTER_SECRET", "unit-test-master-secret-not-for-production")

    import backend.main as main_module

    importlib.reload(main_module)
    try:
        client = TestClient(main_module.app)
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/health").status_code == 200
    finally:
        # Restore the module to its normal (development) import-time state
        # so later tests in the same process don't inherit docs_url=None.
        monkeypatch.delenv("ENV", raising=False)
        importlib.reload(main_module)


def test_docs_are_enabled_in_development(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setenv("MASTER_SECRET", "unit-test-master-secret-not-for-production")

    import backend.main as main_module

    importlib.reload(main_module)
    client = TestClient(main_module.app)
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
