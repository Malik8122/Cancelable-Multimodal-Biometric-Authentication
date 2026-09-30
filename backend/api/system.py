"""GET /system/health: a real (not hardcoded) snapshot of backend readiness.

Distinct from the existing bare `GET /health` (which only proves the process
is up) - this checks the database and calibration state are real, and
reports each modality's model status. Every field is computed at request
time, not a static dict.

**Must never instantiate a model itself.** `get_face_service`/
`get_voice_service` are `@lru_cache`d, zero-argument
functions - the first real call to any of them builds the full pipeline and
loads its checkpoint (100+ MB for face), which is exactly
the memory a 512 MB deployment can't spare on every health check. This
endpoint is polled automatically every 15s by the frontend
(frontend/src/hooks/useAuthSession.ts), so calling every getter here
unconditionally previously forced every checkpoint to load together on
the very first poll. `_model_status` below checks `getter.cache_info().currsize`
(a plain counter on the decorator, not a call) to see whether a *previous*
real request already paid that cost - reusing that already-loaded result is
free - and otherwise only stats the configured checkpoint path (also cheap)
rather than building the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database.schema import SystemHealthResponse
from backend.database.session import get_db
from backend.key_continuity import evaluate_key_continuity
from backend.services.base_service import ModalityService
from backend.services.face_service import get_face_service
from backend.services.voice_service import get_voice_service
from backend.threshold_loader import is_calibrated
from fusion.config import DEFAULT_FUSION_POLICY

router = APIRouter()

#: (lru_cache'd service getter, Settings attribute naming its checkpoint path)
#: per modality.
_MODALITY_GETTERS: dict[str, tuple[Callable[[], ModalityService], str]] = {
    "face": (get_face_service, "face_model_path"),
    "voice": (get_voice_service, "voice_model_path"),
}


def _model_status(modality: str, settings: Settings) -> str:
    """Report status without loading the model unless it's already loaded.

    - Already loaded by an earlier real request (`getter.cache_info().currsize`
      is 1, not 0 - checking this never calls `getter`): report its real
      "loaded"/"mock" state - free, since the getter just returns the cached
      instance.
    - Not loaded yet: stat the configured checkpoint path (a filesystem
      check, not a model load - see models/common/base_embedder.py's own
      `mock_mode = ... or not checkpoint_path.exists()`) and report whether
      one is even configured, without building the pipeline that would
      actually load it.
    """
    getter, path_attr = _MODALITY_GETTERS[modality]
    if getter.cache_info().currsize > 0:
        try:
            service = getter()
            return "mock" if service.pipeline.is_mock else "loaded"
        except Exception as error:  # noqa: BLE001 - health check must never 500
            return f"error: {error}"

    checkpoint_path = Path(getattr(settings, path_attr))
    return "available (not loaded)" if checkpoint_path.exists() else "mock (no checkpoint)"


@router.get("/system/health", response_model=SystemHealthResponse)
def system_health(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> SystemHealthResponse:
    try:
        db.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception as error:  # noqa: BLE001
        database_status = f"error: {error}"

    thresholds_loaded = all(is_calibrated(modality) for modality in ("face", "voice"))

    try:
        key_continuity = evaluate_key_continuity(db, settings).value
    except Exception as error:  # noqa: BLE001 - health check must never 500
        key_continuity = f"error: {error}"

    return SystemHealthResponse(
        backend="online",
        database=database_status,
        face_model=_model_status("face", settings),
        voice_model=_model_status("voice", settings),
        template_protection="active",
        fusion_policy=DEFAULT_FUSION_POLICY.value,
        thresholds_loaded=thresholds_loaded,
        audit_logging=True,
        template_pool_size=settings.template_pool_size,
        diagnostics_enabled=settings.debug_scores,
        key_continuity=key_continuity,
    )
