"""Template-set management: GET /templates/{user_id}, POST .../activate/{version}, POST .../generate.

Nothing here returns template bytes - only lifecycle metadata. `version` is the
template SET version. Every mutating call needs biometric authorization
against the ACTIVE set (a capture of each modality it contains; HTTP 403 on
failure) - see `backend/services/template_sets.py`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.api.revoke import collect_uploads
from backend.config import Settings, get_settings
from backend.database import crud
from backend.database.models import STATUS_ACTIVE, STATUS_STANDBY
from backend.database.schema import ActivateResponse, GenerateSetResponse, TemplateSetInfo, TemplateSetPoolResponse
from backend.database.session import get_db
from backend.services import template_sets

logger = logging.getLogger("backend.api.templates")

router = APIRouter()


@router.get("/templates/{user_id}", response_model=TemplateSetPoolResponse)
def get_template_sets(
    user_id: str,
    application_id: str | None = Query(None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TemplateSetPoolResponse:
    """The template set pool: every set, its status and the modalities it contains (metadata only)."""
    if crud.get_user(db, user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No such user: {user_id!r}")
    resolved_application_id = application_id or settings.application_id
    sets = crud.get_template_sets(db, user_id, resolved_application_id)
    if not sets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No template sets for user_id={user_id!r}.")
    active = next((s for s in sets if s.status == STATUS_ACTIVE), None)
    return TemplateSetPoolResponse(
        user_id=user_id,
        application_id=resolved_application_id,
        pool_size=settings.template_pool_size,
        active_template_set_version=active.version if active else None,
        standby_count=sum(1 for s in sets if s.status == STATUS_STANDBY),
        sets=[
            TemplateSetInfo(
                template_set_version=s.version,
                status=s.status,
                modalities=s.modalities,
                key_versions=s.key_versions,
                template_group_id=s.group_id,
                created_at=s.created_at,
                activated_at=s.activated_at,
                revoked_at=s.revoked_at,
                revoked_reason=s.revoked_reason,
            )
            for s in sets
        ],
    )


@router.post("/templates/{user_id}/activate/{version}", response_model=ActivateResponse)
def activate_template_set(
    user_id: str,
    version: int,
    application_id: str | None = Form(None),
    face_image: UploadFile | None = File(None),
    voice_audio: UploadFile | None = File(None),
    hand_gesture: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ActivateResponse:
    """Promote STANDBY set `version` to ACTIVE (the previous ACTIVE set becomes REVOKED)."""
    try:
        previous, promoted, remaining = template_sets.activate_set(
            db,
            settings,
            user_id=user_id,
            application_id=application_id or settings.application_id,
            uploads=collect_uploads(face_image, voice_audio, hand_gesture),
            version=version,
        )
    except crud.TemplateNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return ActivateResponse(
        success=True,
        user_id=user_id,
        previous_active_template_set_version=previous,
        new_active_template_set_version=promoted,
        remaining_standby_template_sets=remaining,
    )


@router.post("/templates/{user_id}/generate", response_model=GenerateSetResponse)
@router.post("/templates/{user_id}/replenish", response_model=GenerateSetResponse, include_in_schema=False)
def generate_template_set(
    user_id: str,
    application_id: str | None = Form(None),
    face_image: UploadFile | None = File(None),
    voice_audio: UploadFile | None = File(None),
    hand_gesture: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> GenerateSetResponse:
    """Add one new STANDBY template set (all modalities of the ACTIVE set) from the authorization captures.

    409 when the pool already holds `TEMPLATE_POOL_SIZE` live sets. The old
    `/replenish` path is kept as a hidden alias.
    """
    resolved_application_id = application_id or settings.application_id
    try:
        version = template_sets.generate_template_set(
            db,
            settings,
            user_id=user_id,
            application_id=resolved_application_id,
            uploads=collect_uploads(face_image, voice_audio, hand_gesture),
        )
    except crud.TemplateNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except crud.TemplatePoolExhaustedError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    active = crud.get_active_set_rows(db, user_id, resolved_application_id)
    logger.info("Generated template set v%d for user_id=%s", version, user_id)
    return GenerateSetResponse(
        success=True,
        user_id=user_id,
        new_template_set_version=version,
        standby_template_set_versions=crud.standby_set_versions(db, user_id, resolved_application_id),
        active_template_set_version=active[0].template_set_version if active else 0,
    )
