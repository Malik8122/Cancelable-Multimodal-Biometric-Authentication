"""POST /enroll: preprocess -> embed -> one protected template per template set -> store.

One enrollment yields a template in each of `TEMPLATE_POOL_SIZE` (default 4) template SETS (set 1 ACTIVE, the rest
STANDBY). Enroll each modality in turn and every set ends up holding all of them. Never returns template bytes.

Face is a ONE-TIME five-pose enrollment: `pose_front`, `pose_left`, `pose_right`, `pose_up`, `pose_down`. Each pose is
detected (MTCNN), aligned and embedded; only blurry or faceless poses are rejected. The valid embeddings are averaged into
a centroid, the temporary embeddings are discarded, and T1-T4 are generated from the centroid alone. (A single `image`
still enrolls a face from one capture.) `POST /enroll/face/check-pose` gives the UI an immediate verdict per pose.

Declared as a plain `def` route: the service call runs CPU-bound OpenCV and
PyTorch inference, which FastAPI runs in a threadpool automatically.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database import crud
from backend.database.schema import EnrollResponse, FacePoseCheckResponse, VoiceQualityCheckResponse
from backend.database.session import get_db
from backend.services import get_service_for_modality, voice_quality
from backend.services.base_service import EnrollmentInconsistent, LowQualityWarning
from backend.services.face_enrollment import FACE_POSES, POSE_HINTS, VALID, FaceCaptureRejected
from backend.services.recording_quality import FAIR, POOR
from backend.states import ENROLLMENT_INCONSISTENT, LOW_QUALITY_WARNING, RETRY_REQUIRED
from backend.utils import call_modality_service, decode_biometric_sample

logger = logging.getLogger("backend.api.enroll")

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse, response_model_exclude_none=True)
def enroll(
    user_id: str = Form(...),
    modality: str = Form(...),
    application_id: str | None = Form(None),
    image: UploadFile | None = File(None, description="The sample (required unless the face poses are sent)."),
    pose_front: UploadFile | None = File(None, description="Face: front pose."),
    pose_left: UploadFile | None = File(None, description="Face: head turned slightly left."),
    pose_right: UploadFile | None = File(None, description="Face: head turned slightly right."),
    pose_up: UploadFile | None = File(None, description="Face: chin slightly up."),
    pose_down: UploadFile | None = File(None, description="Face: chin slightly down."),
    confirm_image: UploadFile | None = File(
        None,
        description="Voice: the second recording. The two are compared by ECAPA embedding cosine similarity BEFORE "
        "anything is stored: >= 0.75 enrolled (Excellent/Good); 0.60-0.74 -> 409 LOW_QUALITY_WARNING (nothing stored "
        "unless accept_low_quality=true); < 0.60 -> 422 ENROLLMENT_INCONSISTENT (nothing stored).",
    ),
    accept_low_quality: bool = Form(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    if modality == "hand":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="The hand gesture is enrolled with POST /enroll/hand (three Z gesture samples).")
    if confirm_image is not None and modality != "voice":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="`confirm_image` is only used for voice enrollment.")
    pose_uploads = {
        pose: upload
        for pose, upload in zip(FACE_POSES, (pose_front, pose_left, pose_right, pose_up, pose_down))
        if upload is not None
    }
    if pose_uploads and modality != "face":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The pose_* captures are only used for face enrollment.")
    if not pose_uploads and image is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="`image` is required (or, for face, the pose_* captures).")

    raw_image = decode_biometric_sample(modality, image, settings) if image is not None and not pose_uploads else None
    confirm_raw = decode_biometric_sample(modality, confirm_image, settings) if confirm_image is not None else None

    resolved_application_id = application_id or settings.application_id
    service = get_service_for_modality(modality)

    if service.pipeline.is_mock:
        logger.warning("Enrolling user_id=%s modality=%s against a MOCK embedding - not biometrically meaningful", user_id, modality)

    recording_quality: str | None = None
    pose_report: list[dict] | None = None
    try:
        if pose_uploads:
            captures = [(pose, decode_biometric_sample(modality, upload, settings)) for pose, upload in pose_uploads.items()]
            pool, pose_report = call_modality_service(
                service.enroll_poses, modality, db, captures, user_id=user_id, application_id=resolved_application_id
            )
        elif confirm_raw is not None:
            pool, recording_quality = call_modality_service(
                service.enroll_confirmed,
                modality,
                db,
                raw_image,
                confirm_raw,
                user_id=user_id,
                application_id=resolved_application_id,
                accept_low_quality=accept_low_quality,
            )
        else:
            pool = call_modality_service(
                service.enroll, modality, db, raw_image, user_id=user_id, application_id=resolved_application_id
            )
    except voice_quality.VoiceCaptureRejected as rejected:
        # A recording failed the capture-quality gate: nothing stored, nothing recorded as an enrollment outcome.
        logger.info("Voice enrollment capture-quality failure user_id=%s sentence=%s verdict=%s - nothing stored",
                    user_id, rejected.sentence, rejected.report.verdict)
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=_voice_quality_body(rejected, modality))
    except FaceCaptureRejected as rejected:
        # Too few usable poses: nothing stored. The per-pose verdicts tell the UI which poses to retake.
        logger.info("Face enrollment rejected user_id=%s - nothing stored: %s", user_id, rejected)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"status": "FACE_CAPTURE_REJECTED", "detail": str(rejected), "modality": modality, "pose_results": rejected.report},
        )
    except LowQualityWarning as warning:
        # FAIR: usable but lower quality. Nothing is stored; the client may continue (accept_low_quality) or re-record.
        logger.info("Enrollment low-quality warning user_id=%s modality=%s - nothing stored yet", user_id, modality)
        body = {"status": LOW_QUALITY_WARNING, "recording_quality": FAIR, "detail": str(warning), "modality": modality}
        if settings.debug_scores:
            body.update(similarity=warning.similarity)
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=body)
    except EnrollmentInconsistent as error:
        crud.record_enrollment_event(
            db, user_id=user_id, application_id=resolved_application_id, modality=modality, outcome=ENROLLMENT_INCONSISTENT
        )
        logger.info("Enrollment rejected (inconsistent recordings) user_id=%s modality=%s - nothing stored", user_id, modality)
        body = {
            "status": ENROLLMENT_INCONSISTENT,
            "recording_quality": POOR,
            "detail": str(error),
            "modality": modality,
            "enrollment_status": RETRY_REQUIRED,
        }
        if settings.debug_scores:
            body.update(similarity=error.similarity)
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body)

    crud.record_enrollment_event(db, user_id=user_id, application_id=resolved_application_id, modality=modality, outcome="ENROLLED")
    active = next((row for row in pool if row.is_active), pool[0])
    standby = [row.template_set_version for row in pool if row.template_set_status == "STANDBY"]
    logger.info(
        "Enrolled user_id=%s modality=%s sets=%d active_set=v%d", user_id, modality, len(pool), active.template_set_version
    )

    return EnrollResponse(
        success=True,
        user_id=user_id,
        modality=modality,
        templates_created=len(pool),
        active_template_set_version=active.template_set_version,
        standby_template_set_versions=standby,
        template_version=active.template_version,
        key_version=active.key_version,
        template_id=active.template_id,
        recording_quality=recording_quality,
        poses_valid=sum(1 for r in pose_report if r["status"] == VALID) if pose_report else None,
        pose_results=pose_report,
    )


VOICE_CAPTURE_QUALITY_FAILURE = "VOICE_CAPTURE_QUALITY_FAILURE"


def _voice_quality_body(rejected: voice_quality.VoiceCaptureRejected, modality: str) -> dict:
    return {"status": VOICE_CAPTURE_QUALITY_FAILURE, "detail": str(rejected), "modality": modality,
            "sentence": rejected.sentence, "verdict": rejected.report.verdict, "metrics": rejected.report.metrics}


@router.post("/voice/check-quality", response_model=VoiceQualityCheckResponse, response_model_exclude_none=True)
def check_voice_quality(
    audio: UploadFile = File(...),
    sentence: int | None = Form(None),
    settings: Settings = Depends(get_settings),
) -> VoiceQualityCheckResponse:
    """Capture-quality verdict for ONE recording, so the UI can ask for a retake of just that sentence.

    The same gate enrollment and authentication apply (backend/services/voice_quality.py). Stores nothing and embeds
    nothing - no audio, no embedding, no audit row; the backend re-checks every recording when it is submitted.
    """
    report = voice_quality.assess(decode_biometric_sample("voice", audio, settings), settings)
    return VoiceQualityCheckResponse(sentence=sentence, verdict=report.verdict, passed=report.passed,
                                     message=report.message, metrics=report.metrics, diagnostics=report.diagnostics)


@router.post("/enroll/face/check-pose", response_model=FacePoseCheckResponse, response_model_exclude_none=True)
def check_face_pose(
    pose: str = Form(...),
    image: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> FacePoseCheckResponse:
    """Verdict for ONE captured pose (VALID / NO_FACE / BLURRY) so the guided UI can ask for an immediate retake.

    Runs the same MTCNN detection + alignment + blur check as enrollment and stores nothing - not the image, not an embedding.
    """
    if pose not in FACE_POSES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"pose must be one of {FACE_POSES}")
    raw = decode_biometric_sample("face", image, settings)
    verdict = call_modality_service(get_service_for_modality("face").check_capture, "face", raw)
    return FacePoseCheckResponse(pose=pose, status=verdict, valid=verdict == VALID, detail=POSE_HINTS.get(verdict))
