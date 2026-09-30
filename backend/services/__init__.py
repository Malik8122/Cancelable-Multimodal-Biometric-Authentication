"""Per-modality services wiring embeddings/pipelines.py + template_protection/ + the database together.

See base_service.py::ModalityService for the shared enroll/authenticate/revoke
logic, face_service.py/voice_service.py for the
thin, modality-specific instantiations, and hand_service.py for the dynamic hand
gesture (DTW over MediaPipe landmark sequences, same template-set interface) `backend/api/*.py` depends on.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from backend.services.base_service import ModalityService

_SUPPORTED_MODALITIES = ("face", "voice", "hand")


def get_service_for_modality(modality: str) -> ModalityService:  # HandGestureService for "hand" (same interface)
    """Resolve a modality name (as it arrives in an API request) to its service.

    Each per-modality module (face_service.py etc.) only constructs its
    pipeline on first actual use (see their `lru_cache`d `get_*_service`
    functions), so requesting "voice" here never triggers loading the face
    checkpoint.
    """
    if modality == "face":
        from backend.services.face_service import get_face_service

        return get_face_service()
    if modality == "voice":
        from backend.services.voice_service import get_voice_service

        return get_voice_service()
    if modality == "hand":
        from backend.services.hand_service import get_hand_service

        return get_hand_service()

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Unsupported modality {modality!r}; expected one of {_SUPPORTED_MODALITIES}",
    )
