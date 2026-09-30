"""FastAPI application entrypoint.

Run locally with:

    uvicorn backend.main:app --reload

See docs/BACKEND_API.md for the full endpoint reference.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import audit, authenticate, buildings, enroll, fusion, hand, metrics, revoke, system, templates, user, verify
from backend.config import get_settings
from backend.database.crud import ConcurrentEnrollmentError
from backend.database.session import get_session_factory, init_db
from backend.key_continuity import KeyContinuityError, enforce_key_continuity_at_startup
from backend.security_validation import SecurityValidationError

logging.basicConfig(level=logging.INFO)


def _resolve_cors_origins() -> list[str]:
    """Read the allowed CORS origin(s) directly from the environment rather
    than via `backend.config.get_settings()`.

    CORS middleware has to be registered at import time - Starlette builds
    its middleware stack on the app's first request and raises if
    `add_middleware` is called afterwards, so this can't be deferred into
    the `lifespan` startup hook either. `get_settings()` requires
    `MASTER_SECRET` to be set; calling it here, at module-import time, would
    make importing this module fail during pytest collection (which happens
    before the `test_master_secret` autouse fixture in `tests/conftest.py`
    has a chance to set it). Every other setting is still read the normal
    lazy way, via `Depends(get_settings)` inside request handlers - this is
    the one deliberate exception, for this one structural reason.

    `CORS_ORIGIN` (singular, one origin) is the production/deployment-facing
    name - "set this to your deployed Vercel URL" is simpler to document and
    configure than a comma-separated list. `CORS_ALLOWED_ORIGINS` (plural,
    comma-separated) remains supported for local/multi-origin development.
    `CORS_ORIGIN` wins if both are set. Never `"*"` either way - this API
    handles biometric templates.
    """
    single_origin = os.environ.get("CORS_ORIGIN")
    if single_origin and single_origin.strip():
        return [single_origin.strip()]
    raw = os.environ.get("CORS_ALLOWED_ORIGINS")
    if not raw:
        return ["http://localhost:5173"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _is_production() -> bool:
    """Mirrors `backend.config.Settings.is_production` without needing
    `get_settings()` at import time - see `_resolve_cors_origins` for why
    that's unsafe here. `ENV=production` disables interactive Swagger/ReDoc
    docs below; it never changes any authentication or model behavior."""
    return os.environ.get("ENV", "development").strip().lower() == "production"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = get_session_factory()()
    try:
        enforce_key_continuity_at_startup(db, get_settings())
    except KeyContinuityError as exc:
        # Never the secret or the fingerprint - see backend/key_continuity.py's own messages,
        # which are already safe to log/print in full.
        logging.getLogger("backend").error("Startup refused: %s", exc)
        raise
    finally:
        db.close()
    yield


app = FastAPI(
    title="Cancelable Multimodal Biometric Authentication API",
    description=(
        "Enrollment, authentication, and revocation over cancelable "
        "(non-invertible-by-design) biometric templates. Only protected "
        "templates are ever persisted - see docs/TEMPLATE_PROTECTION.md."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # Interactive Swagger/ReDoc/raw-OpenAPI-JSON are convenient in
    # development but needlessly expose the full API surface publicly in
    # production - disabled when ENV=production (see _is_production()).
    # This changes nothing about how any endpoint behaves, only whether its
    # documentation is servable.
    docs_url=None if _is_production() else "/docs",
    redoc_url=None if _is_production() else "/redoc",
    openapi_url=None if _is_production() else "/openapi.json",
)

# Phase 3A dashboard runs on a different origin (Vite dev server) than this
# API - explicit allow-list, never "*", since this API handles biometric
# templates. See backend/config.py::Settings.cors_allowed_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(enroll.router, tags=["enrollment"])
app.include_router(hand.router, tags=["enrollment"])
app.include_router(authenticate.router, tags=["authentication"])
app.include_router(fusion.router, tags=["authentication"])
app.include_router(verify.router, tags=["verification"])
app.include_router(revoke.router, tags=["revocation"])
app.include_router(templates.router, tags=["templates"])
app.include_router(user.router, tags=["user"])
app.include_router(buildings.router, tags=["buildings"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(audit.router, tags=["audit"])
app.include_router(system.router, tags=["system"])


@app.exception_handler(ConcurrentEnrollmentError)
def handle_concurrent_enrollment(_request: Request, exc: ConcurrentEnrollmentError) -> JSONResponse:
    """Two racing enroll/revoke requests for the same context -> 409, not a bare 500.

    See backend/database/crud.py::ConcurrentEnrollmentError and
    backend/database/models.py's partial unique index for why this can
    happen at all under FastAPI's threadpooled sync routes.
    """
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})


@app.exception_handler(SecurityValidationError)
def handle_security_validation_failure(_request: Request, exc: SecurityValidationError) -> JSONResponse:
    """A structural security invariant broke (backend/security_validation.py) -
    always a 500, never a normal `authenticated: false`, and never echoes the
    exception's own detail (which names which checks failed - useful in
    server logs, not something to hand back to an untrusted caller)."""
    logging.getLogger("backend").error("Security validation failure: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Authentication could not be completed due to an internal security validation failure."},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
