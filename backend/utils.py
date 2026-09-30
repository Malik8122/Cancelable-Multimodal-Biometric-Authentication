"""Upload validation and logging setup shared across backend/api/*.py.

Nothing in this module - or anywhere else in backend/ - ever logs a template,
an embedding, or raw image bytes: log statements are limited to identifiers
(user_id, modality, application_id) and outcomes (accepted/rejected,
counts), per the spec's "never log templates, never print embeddings."
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.config import Settings

logger = logging.getLogger("backend")


def validate_upload(
    file: UploadFile,
    contents: bytes,
    settings: Settings,
    allowed_content_types: tuple[str, ...] | None = None,
) -> None:
    """Reject an upload that fails content-type or size checks.

    `contents` is passed in (already read by the caller) rather than read
    again here, so a single `UploadFile.read()` call is the only I/O -
    reading twice would either double the work or require seeking, and
    `UploadFile`'s underlying stream isn't guaranteed seekable in every
    deployment.

    `allowed_content_types` defaults to `settings.allowed_content_types`
    (images); voice's route passes `settings.allowed_audio_content_types`
    instead - same validation, different acceptable MIME set.
    """
    allowed_content_types = allowed_content_types or settings.allowed_content_types
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}; expected one of {allowed_content_types}",
        )
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload exceeds the {settings.max_upload_size_bytes}-byte limit",
        )


def decode_image(contents: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an RGB `np.ndarray` (H, W, 3), uint8.

    OpenCV decodes to BGR by default; every `preprocessing/*.py` module
    (and thus `embeddings.pipelines.ModalityPipeline.embed`) expects RGB,
    matching `models/common/base_embedder.py`'s documented input contract.
    """
    buffer = np.frombuffer(contents, dtype=np.uint8)
    bgr_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode the uploaded file as an image")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def call_modality_service(operation, modality: str, *args, **kwargs):
    """Run a `ModalityService.enroll/authenticate/revoke` call, converting a
    real preprocessing failure into a clean 422 instead of an unhandled 500.

    E.g. face's real detector (MTCNN, via facenet-pytorch) raises
    `ValueError("No face detected in the provided image.")` for a sample
    with no detectable face - a genuinely bad biometric sample, not a server
    error, so every route that reaches a `ModalityService` method goes
    through this one place rather than each re-implementing the same
    try/except.
    """
    try:
        return operation(*args, **kwargs)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process the {modality} sample: {error}",
        ) from error


def decode_biometric_sample(modality: str, file: UploadFile, settings: Settings):
    """Read, validate, and decode one uploaded biometric sample for `modality`.

    Voice needs different upload validation (audio MIME types, via
    `decode_audio`) than the image modality (face,
    via `decode_image`) - this is the one place that branches on modality,
    so `backend/api/enroll.py`, `authenticate.py`, and `verify.py` share it
    instead of each re-implementing the same `if modality == "voice"` check.

    Returns whatever `ModalityService.enroll`/`.authenticate` expect as their
    `raw_image` argument for that modality: an `(H, W, 3)` array for the
    image modalities, or a `(waveform, sample_rate)` tuple for voice (see
    `backend/services/voice_service.py::_VoicePipelineAdapter`).
    """
    contents = file.file.read()
    if modality == "hand":
        # A MediaPipe landmark sequence (JSON), parsed but not yet validated - the capture gate
        # (backend/services/hand_service.py) decides whether it is usable. Unreadable JSON is a decode failure.
        from preprocessing.hand_gesture import HandCaptureRejected, parse_capture

        validate_upload(file, contents, settings, allowed_content_types=settings.allowed_gesture_content_types)
        try:
            return parse_capture(contents)
        except HandCaptureRejected as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    if modality == "voice":
        validate_upload(file, contents, settings, allowed_content_types=settings.allowed_audio_content_types)
        return decode_audio(contents)
    validate_upload(file, contents, settings)
    return decode_image(contents)


def record_authentication_audit(
    db: Session,
    *,
    user_id: str,
    modality_list: list[str],
    similarity_scores: dict[str, float],
    thresholds_used: dict[str, float],
    authenticated: bool,
    started_at: float,
    template_versions: dict[str, int],
    key_versions: dict[str, int],
    building_id: str | None = None,
    fusion_score: float | None = None,
    fusion_policy: str | None = None,
    fusion_similarity: float | None = None,
    template_set_version: int | None = None,
    template_set_status: str | None = None,
    authentication_state: str | None = None,
    submitted_modalities: list[str] | None = None,
    enrolled_modalities: list[str] | None = None,
    authenticated_modalities: list[str] | None = None,
    **rotation,
) -> None:
    """Write one server-side audit row for an authentication attempt.

    Shared by `/authenticate`, `/verify/{modality}`, and
    `/authenticate/fusion` so latency measurement and the "never log
    anything biometric-derived beyond a score" rule live in exactly one
    place (see `backend/database/models.py::AuditLog`'s docstring for what
    "biometric-derived" excludes). `started_at` is a `time.perf_counter()`
    value taken by the caller before decoding/inference began.
    """
    from backend.database import audit

    latency_ms = round((time.perf_counter() - started_at) * 1000)
    audit.record_attempt(
        db,
        user_id=user_id,
        building_id=building_id,
        modality_list=modality_list,
        similarity_scores=similarity_scores,
        thresholds_used=thresholds_used,
        authenticated=authenticated,
        latency_ms=latency_ms,
        template_versions=template_versions,
        key_versions=key_versions,
        fusion_score=fusion_score,
        fusion_policy=fusion_policy,
        fusion_similarity=fusion_similarity,
        template_set_version=template_set_version,
        template_set_status=template_set_status,
        authentication_state=authentication_state,
        submitted_modalities=submitted_modalities,
        enrolled_modalities=enrolled_modalities,
        authenticated_modalities=authenticated_modalities,
        **rotation,
    )


def decode_audio(contents: bytes) -> tuple[np.ndarray, int]:
    """Decode uploaded WAV bytes into a (float32 waveform, sample_rate) pair.

    Voice's counterpart to `decode_image` - delegates to
    `preprocessing.voice.load_wav_bytes` rather than re-implementing WAV
    parsing here, same "backend/api/*.py never duplicates preprocessing"
    rule every other modality already follows.
    """
    from preprocessing.voice import load_wav_bytes

    try:
        return load_wav_bytes(contents)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode the uploaded file as a WAV audio file"
        ) from error
