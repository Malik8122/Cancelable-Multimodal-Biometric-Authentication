"""POST /verify/{face,voice,hand}: modality-only verification.

Identical to POST /authenticate except the modality comes from the URL
instead of a form field - each route is a thin wrapper around `_verify`,
which uses the same shared authentication path and returns the same single
fused decision.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database.schema import AuthenticateResponse
from backend.database.session import get_db
from backend.services.authentication import authenticate_samples, respond
from fusion.config import FusionPolicy

router = APIRouter()


def _verify(
    modality: str,
    user_id: str,
    application_id: str | None,
    building_id: str | None,
    image: UploadFile,
    db: Session,
    settings: Settings,
):
    outcome = authenticate_samples(
        db,
        settings,
        user_id=user_id,
        application_id=application_id or settings.application_id,
        building_id=building_id,
        uploads={modality: image},
        policy=FusionPolicy.ALL_REQUIRED,
        started_at=time.perf_counter(),
    )
    return respond(outcome, settings)


@router.post("/verify/face", response_model=AuthenticateResponse, response_model_exclude_none=True)
def verify_face(
    user_id: str = Form(...),
    application_id: str | None = Form(None),
    building_id: str | None = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    return _verify("face", user_id, application_id, building_id, image, db, settings)


@router.post("/verify/voice", response_model=AuthenticateResponse, response_model_exclude_none=True)
def verify_voice(
    user_id: str = Form(...),
    application_id: str | None = Form(None),
    building_id: str | None = Form(None),
    image: UploadFile = File(..., description="WAV audio file"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthenticateResponse:
    """Named `image` for consistency with the other `/verify/*` routes - it is
    actually a WAV audio upload (see `decode_biometric_sample`'s voice branch)."""
    return _verify("voice", user_id, application_id, building_id, image, db, settings)


@router.post("/verify/hand", response_model=AuthenticateResponse, response_model_exclude_none=True)
def verify_hand(
    user_id: str = Form(...),
    application_id: str | None = Form(None),
    building_id: str | None = Form(None),
    gesture: UploadFile = File(..., description="ONE execution of the enrolled gesture: a JSON MediaPipe landmark sequence."),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthenticateResponse:
    """One gesture execution, compared by DTW with the enrolled executions of the ACTIVE set (hand_service.py). A capture
    problem is CAPTURE_ERROR / QUALITY_INSUFFICIENT with its code in `hand_gesture.decision`, never a mismatch."""
    return _verify("hand", user_id, application_id, building_id, gesture, db, settings)
