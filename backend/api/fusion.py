"""POST /authenticate/fusion: one backend-computed decision over the modalities the USER submits.

Buildings define no biometric policy: `building_id` is an optional label recorded in the audit log (an unknown id is a
404). The user chooses which enrolled modalities to present, any non-empty subset of face / voice / hand:

1. A submitted modality that is not enrolled -> HTTP 409 `ENROLLMENT_REQUIRED` (with `missing_modalities`). That modality
   is not authenticated, nothing is decoded or evaluated, and the attempt is audited as its own state.
2. Otherwise exactly the submitted modalities are authenticated against the ACTIVE template set and fused - nothing else -
   giving `ACCESS_GRANTED` (all verified) or `ACCESS_DENIED`. One fusion engine serves every combination.

`ALL_REQUIRED` (default): every submitted modality must individually pass, so one strong match can never compensate for
another modality's failure. `WEIGHTED` vetoes on any per-modality score below `fusion.config.DEFAULT_WEIGHTED_FLOOR`.
`AT_LEAST_TWO` needs at least two submitted modalities (422 otherwise), of which at least two must pass.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.buildings import get_building
from backend.config import Settings, get_settings
from backend.database.schema import EnrollmentRequiredResponse, FusionAuthenticateResponse
from backend.database.session import get_db
from backend.services.authentication import authenticate_samples, release_modality_cache, respond
from fusion.config import ALL_POLICIES, DEFAULT_FUSION_POLICY, FusionPolicy

router = APIRouter()

# Kept under its previous name for any external caller.
_release_modality_cache = release_modality_cache


@router.post(
    "/authenticate/fusion",
    response_model=FusionAuthenticateResponse,
    response_model_exclude_none=True,
    responses={409: {"model": EnrollmentRequiredResponse, "description": "A submitted modality is not enrolled."}},
)
def authenticate_fusion(
    user_id: str = Form(...),
    application_id: str | None = Form(None),
    building_id: str | None = Form(None),
    fusion_policy: str | None = Form(None),
    face_image: UploadFile | None = File(None),
    voice_audio: UploadFile | None = File(None),
    hand_gesture: UploadFile | None = File(
        None, description="Hand gesture: ONE execution of the enrolled gesture, a JSON MediaPipe landmark sequence."),
    voice_audio_2: UploadFile | None = File(
        None, description="Optional second voice sentence (requires voice_audio). Both sentences are quality-gated and "
        "embedded separately, and their unit embeddings are combined (L2-normalized mean) into ONE embedding that is "
        "compared once with the ACTIVE template - the rule voice enrollment uses. The frontend always sends two."),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    started_at = time.perf_counter()

    policy_name = fusion_policy or DEFAULT_FUSION_POLICY.value
    if policy_name not in ALL_POLICIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported fusion_policy {policy_name!r}; expected one of {ALL_POLICIES}",
        )
    policy = FusionPolicy(policy_name)

    if voice_audio_2 is not None and voice_audio is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="voice_audio_2 is the second voice sentence: send it together with voice_audio.")
    uploads: dict[str, UploadFile | list[UploadFile]] = {
        modality: upload
        for modality, upload in (("face", face_image), ("voice", voice_audio), ("hand", hand_gesture))
        if upload is not None
    }
    if voice_audio_2 is not None:
        # Two voice sentences: combined into one embedding by the voice service (2/2 protocol, docs/VOICE_MODEL.md).
        uploads["voice"] = [voice_audio, voice_audio_2]
    if not uploads:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Select at least one biometric factor: face_image, voice_audio or hand_gesture.",
        )
    if policy == FusionPolicy.AT_LEAST_TWO and len(uploads) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="fusion_policy=AT_LEAST_TWO requires at least two submitted modalities.",
        )
    if building_id and get_building(building_id, settings.buildings_config_path) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown building {building_id!r}.")

    outcome = authenticate_samples(
        db,
        settings,
        user_id=user_id,
        application_id=application_id or settings.application_id,
        building_id=building_id,
        uploads=uploads,
        policy=policy,
        started_at=started_at,
    )
    return respond(outcome, settings)
