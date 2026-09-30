"""Face biometric service: wraps `embeddings.pipelines.FacePipeline`."""

from __future__ import annotations

from functools import lru_cache

from backend.config import get_settings
from backend.services.base_service import ModalityService
from embeddings.pipelines import FacePipeline


@lru_cache
def get_face_service() -> ModalityService:
    """Builds the `FacePipeline` (and thus loads the checkpoint) on first use, not at import time.

    `lru_cache` means this runs once per process - the checkpoint is loaded
    once and reused across every `/enroll`, `/authenticate`, and
    `/verify/face` request. Replacing `models/face/saved/face_embedder.pt`
    with a different checkpoint requires no code change here: only
    restarting the process (which clears the cache) to pick it up.
    """
    settings = get_settings()
    pipeline = FacePipeline(checkpoint_path=settings.face_model_path, alignment=settings.face_alignment)
    return ModalityService(modality="face", pipeline=pipeline, settings=settings)
