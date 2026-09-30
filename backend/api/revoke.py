"""POST /revoke-template: revoke the ACTIVE template set and activate the next STANDBY set.

Revocation is per SET: the face and voice templates of the ACTIVE set
become REVOKED together and the oldest STANDBY set becomes ACTIVE for every
modality at once. No login exists, so the caller must first authenticate
against the ACTIVE set - a fresh capture of every modality it contains
(`face_image`, `voice_audio`, `hand_gesture`); failure is
HTTP 403. With no STANDBY set left: 409 "Template set pool exhausted.
Re-enrollment required." (nothing changes).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database import crud
from backend.database.schema import RevokeResponse
from backend.database.session import get_db
from backend.services import template_sets

logger = logging.getLogger("backend.api.revoke")

router = APIRouter()


def collect_uploads(
    face_image: UploadFile | None,
    voice_audio: UploadFile | None,
    hand_gesture: UploadFile | None = None,
) -> dict[str, UploadFile]:
    return {
        modality: upload
        for modality, upload in (
            ("face", face_image),
            ("voice", voice_audio),
            ("hand", hand_gesture),
        )
        if upload is not None
    }


@router.post("/revoke-template", response_model=RevokeResponse)
def revoke_template_endpoint(
    user_id: str = Form(...),
    application_id: str | None = Form(None),
    reason: str | None = Form(None),
    face_image: UploadFile | None = File(None),
    voice_audio: UploadFile | None = File(None),
    hand_gesture: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RevokeResponse:
    resolved_application_id = application_id or settings.application_id
    try:
        revoked, promoted, remaining = template_sets.revoke_active_set(
            db,
            settings,
            user_id=user_id,
            application_id=resolved_application_id,
            uploads=collect_uploads(face_image, voice_audio, hand_gesture),
            reason=reason,
        )
    except crud.TemplateNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except crud.TemplatePoolExhaustedError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    logger.info("Revoked user_id=%s set v%d -> v%d remaining_standby=%d", user_id, revoked, promoted, remaining)
    return RevokeResponse(
        success=True,
        user_id=user_id,
        revoked_template_set_version=revoked,
        new_active_template_set_version=promoted,
        remaining_standby_template_sets=remaining,
    )
