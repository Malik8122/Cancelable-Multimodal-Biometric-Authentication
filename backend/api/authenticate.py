"""POST /authenticate: one modality -> the same single fused decision as /authenticate/fusion.

Compares only against the ACTIVE template set. If the modality is not enrolled the answer is HTTP 409
ENROLLMENT_REQUIRED (nothing is verified). The response carries one fusion similarity (for a single modality, that
modality's similarity is the fused value); per-modality fields appear only under DEBUG_SCORES=true.
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


@router.post("/authenticate", response_model=AuthenticateResponse, response_model_exclude_none=True)
def authenticate(
    user_id: str = Form(...),
    modality: str = Form(...),
    application_id: str | None = Form(None),
    building_id: str | None = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    started_at = time.perf_counter()
    outcome = authenticate_samples(
        db,
        settings,
        user_id=user_id,
        application_id=application_id or settings.application_id,
        building_id=building_id,
        uploads={modality: image},
        policy=FusionPolicy.ALL_REQUIRED,
        started_at=started_at,
    )
    return respond(outcome, settings)
