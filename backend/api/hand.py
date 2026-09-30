"""Hand gesture enrollment: POST /enroll/hand (three independent Z samples) and POST /enroll/hand/check (one sample's verdict).

The client runs MediaPipe Hands and uploads each execution as a JSON landmark sequence (content type
`application/json`, format in preprocessing/hand_gesture.py::parse_capture) - never video. Authentication uses the
shared routes: `POST /verify/hand` (field `gesture`) and `POST /authenticate/fusion` (field `hand_gesture`).

Capture failures (NO_HAND_DETECTED, INSUFFICIENT_FRAMES, CAPTURE_QUALITY_FAILURE, INVALID_TRAJECTORY - incl. not a Z -,
MALFORMED_CAPTURE) are 422 `HAND_CAPTURE_REJECTED` with the failing attempt number - nothing is stored. They are never
reported as a biometric mismatch.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database import crud
from backend.database.schema import EnrollResponse, HandGestureCheckResponse
from backend.database.session import get_db
from backend.services.hand_service import HandEnrollmentIncomplete, get_hand_service
from backend.utils import decode_biometric_sample
from preprocessing.hand_gesture import MALFORMED_CAPTURE, REASONS, HandCaptureRejected

logger = logging.getLogger("backend.api.hand")

router = APIRouter()

HAND_CAPTURE_REJECTED = "HAND_CAPTURE_REJECTED"


def _decode(upload: UploadFile, settings: Settings, attempt: int):
    try:
        return decode_biometric_sample("hand", upload, settings)
    except HTTPException as error:
        if error.status_code != status.HTTP_400_BAD_REQUEST:
            raise  # wrong content type / too large: a request error
        raise HandCaptureRejected(MALFORMED_CAPTURE, str(error.detail), attempt) from error


def _rejected_body(rejected: HandCaptureRejected) -> dict:
    return {"status": HAND_CAPTURE_REJECTED, "modality": "hand", "verdict": rejected.code, "attempt": rejected.attempt,
            "capture_error": rejected.is_capture_error, "detail": str(rejected)}


@router.post("/enroll/hand", response_model=EnrollResponse, response_model_exclude_none=True)
def enroll_hand(
    user_id: str = Form(...),
    application_id: str | None = Form(None),
    attempts: list[UploadFile] = File(..., description="The independent gesture samples (JSON landmark sequences), one file per sample."),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Enroll the dynamic hand gesture from `Settings.hand_gesture_enrollment_attempts` (default 3) independent Z samples.

    Each execution is validated (capture gate) and turned into a feature sequence; all of them are kept - transformed
    under each template set's key - so authentication can compare one live execution with every enrolled one.
    """
    resolved_application_id = application_id or settings.application_id
    service = get_hand_service()
    try:
        captures = [_decode(upload, settings, n) for n, upload in enumerate(attempts, start=1)]
        rows, summary = service.enroll_attempts(db, captures, user_id=user_id, application_id=resolved_application_id)
    except HandCaptureRejected as rejected:
        logger.info("Hand enrollment capture rejected user_id=%s attempt=%s verdict=%s - nothing stored",
                    user_id, rejected.attempt, rejected.code)
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=_rejected_body(rejected))
    except HandEnrollmentIncomplete as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    crud.record_enrollment_event(db, user_id=user_id, application_id=resolved_application_id, modality="hand", outcome="ENROLLED")
    active = next((row for row in rows if row.is_active), rows[0])
    standby = [row.template_set_version for row in rows if row.template_set_status == "STANDBY"]
    logger.info("Enrolled user_id=%s modality=hand gesture=%s attempts=%d sets=%d active_set=v%d",
                user_id, summary["gesture_type"], summary["attempts"], len(rows), active.template_set_version)
    public = {k: summary[k] for k in ("gesture_type", "attempts", "attempt_frames_detected", "attempt_durations_s",
                                      "template_format")}
    if settings.debug_scores:
        public.update(intra_distance_median=summary["intra_distance_median"], intra_distance_max=summary["intra_distance_max"])
    return EnrollResponse(
        success=True,
        user_id=user_id,
        modality="hand",
        templates_created=len(rows),
        active_template_set_version=active.template_set_version,
        standby_template_set_versions=standby,
        template_version=active.template_version,
        key_version=active.key_version,
        template_id=active.template_id,
        hand_gesture=public,
    )


@router.post("/enroll/hand/check", response_model=HandGestureCheckResponse, response_model_exclude_none=True)
def check_hand_gesture(
    gesture: UploadFile = File(...),
    attempt: int | None = Form(None),
    settings: Settings = Depends(get_settings),
) -> HandGestureCheckResponse:
    """Capture-gate verdict for ONE execution so the UI can ask for an immediate retake of just that attempt.

    The same gate enrollment and authentication apply. Stores nothing, writes no audit row.
    """
    try:
        processed = get_hand_service().process(_decode(gesture, settings, attempt), attempt=attempt)
    except HandCaptureRejected as rejected:
        if settings.debug_scores:
            logger.info("HAND-CHECK-DEBUG sample=%s verdict=%s detail=%s", attempt, rejected.code, rejected)
        return HandGestureCheckResponse(attempt=attempt, verdict=rejected.code, passed=False, message=str(rejected))
    metrics = {k: v for k, v in processed.metadata.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if settings.debug_scores:
        logger.info("HAND-CHECK-DEBUG sample=%s verdict=OK metrics=%s", attempt, metrics)
    return HandGestureCheckResponse(attempt=attempt, verdict="OK", passed=True, message="Gesture captured.", metrics=metrics)


__all__ = ["router", "REASONS"]
