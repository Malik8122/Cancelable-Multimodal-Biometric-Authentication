"""Template-set management: revoke, activate, generate - each behind biometric authorization.

There is no login layer in this project. Instead, every management action
first requires a *successful authentication against the ACTIVE template set*
(a fresh capture of every modality the ACTIVE set contains, each of which must
match its own template). That stops a stranger who merely knows a user_id from
exhausting the user's set pool, activating a set, or seeding new standby sets
made from their own biometric. Failed authorization -> HTTP 403 (and an audit
row). Revoking or activating needs no model beyond the authorization capture
itself; generating a new set reuses the embeddings computed for authorization,
so nothing is captured twice.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.config import Settings
from backend.database import crud
from backend.database.models import STATUS_ACTIVE
from backend.services import get_service_for_modality, voice_quality
from preprocessing.hand_gesture import HandCaptureRejected
from backend.services.authentication import release_modality_cache
from backend.services.base_service import build_entry
from backend.states import ACCESS_DENIED, ACCESS_GRANTED
from backend.utils import call_modality_service, decode_biometric_sample, record_authentication_audit
from template_protection.biohash import TEMPLATE_FORMAT_VERSION

logger = logging.getLogger("backend.services.template_sets")

#: Audit `fusion_policy` value marking these rows as management-authorization attempts.
AUDIT_POLICY = "TEMPLATE_MANAGEMENT"


def authorize_with_active_set(
    db: Session,
    settings: Settings,
    *,
    user_id: str,
    application_id: str,
    uploads: dict[str, UploadFile],
    action: str,
) -> dict[str, np.ndarray]:
    """Authenticate `uploads` against the ACTIVE set; return the embeddings on success (see `_authorize`)."""
    return _authorize(db, settings, user_id=user_id, application_id=application_id, uploads=uploads, action=action)[0]


def _authorize(
    db: Session,
    settings: Settings,
    *,
    user_id: str,
    application_id: str,
    uploads: dict[str, UploadFile],
    action: str,
) -> tuple[dict[str, np.ndarray], int]:
    """Authenticate `uploads` against the ACTIVE set; return the embeddings on success.

    Raises `TemplateNotFoundError` (no ACTIVE set) or HTTP 403 (a capture is
    missing, or any modality fails to match). The caller owns the returned
    embeddings and must drop them when done.
    """
    started_at = time.perf_counter()
    # ONE snapshot of the pool for every modality (same reasoning as authentication.authenticate_samples).
    pool = crud.get_set_rows(db, user_id, application_id)
    active_rows = [row for row in pool if row.is_active]
    if not active_rows:
        raise crud.TemplateNotFoundError(f"No active template set for user_id={user_id!r}.")

    required = sorted({row.modality for row in active_rows})
    if sorted(uploads) != required:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Authorization requires a capture of every modality in the active template set: {', '.join(required)}.",
        )

    embeddings: dict[str, np.ndarray] = {}
    results = {}
    try:
        for modality in required:
            raw = decode_biometric_sample(modality, uploads[modality], settings)
            service = get_service_for_modality(modality)
            try:
                try:
                    service.gate(raw)  # voice capture-quality gate, as at enrollment and authentication
                except (voice_quality.VoiceCaptureRejected, HandCaptureRejected) as rejected:
                    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(rejected)) from rejected
                embeddings[modality] = call_modality_service(service.embed, modality, raw)
                results[modality] = service.authenticate_embedding(db, embeddings[modality], user_id, application_id, pool=pool)
            finally:
                del service
                if len(required) > 1:
                    release_modality_cache(modality)
    except BaseException:
        embeddings.clear()
        raise

    authorized = all(result.authenticated for result in results.values())
    set_version = next(iter(results.values())).template_set_version
    record_authentication_audit(
        db,
        user_id=user_id,
        building_id=None,
        modality_list=required,
        similarity_scores={m: r.score for m, r in results.items()},
        thresholds_used={m: r.threshold for m, r in results.items()},
        authenticated=authorized,
        started_at=started_at,
        template_versions={m: r.template_set_version for m, r in results.items()},
        key_versions={m: r.key_version for m, r in results.items()},
        fusion_policy=AUDIT_POLICY,
        template_set_version=set_version,
        template_set_status=STATUS_ACTIVE,
        authentication_state=ACCESS_GRANTED if authorized else ACCESS_DENIED,
        authenticated_modalities=[m for m, r in results.items() if r.authenticated],
    )
    logger.info("Template-management authorization user_id=%s action=%s authorized=%s", user_id, action, authorized)
    if not authorized:
        embeddings.clear()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Biometric authorization failed.")
    return embeddings, set_version


def revoke_active_set(
    db: Session, settings: Settings, *, user_id: str, application_id: str, uploads: dict[str, UploadFile], reason: str | None
) -> tuple[int, int, int]:
    """Authorize, then ACTIVE set -> REVOKED and the oldest STANDBY set -> ACTIVE (all modalities together).

    Returns (revoked_version, promoted_version, remaining_standby_sets).
    """
    embeddings, authorized_set = _authorize(
        db, settings, user_id=user_id, application_id=application_id, uploads=uploads, action="revoke"
    )
    embeddings.clear()
    # Rotate exactly the set that was just authorized against (compare-and-swap): a concurrent authorized revoke that
    # already retired it makes this one a 409 instead of a second rotation.
    revoked, promoted = crud.rotate_expected_or_conflict(
        db, user_id, application_id, expected_active=authorized_set, reason=reason or "revoked by user"
    )
    return revoked, promoted, len(crud.standby_set_versions(db, user_id, application_id))


def activate_set(
    db: Session, settings: Settings, *, user_id: str, application_id: str, uploads: dict[str, UploadFile], version: int
) -> tuple[int | None, int, int]:
    """Authorize, then make STANDBY set `version` ACTIVE (previous ACTIVE set -> REVOKED)."""
    embeddings, authorized_set = _authorize(
        db, settings, user_id=user_id, application_id=application_id, uploads=uploads, action="activate"
    )
    embeddings.clear()
    previous, promoted = crud.activate_standby_set_if_current(
        db, user_id, application_id, version, expected_active=authorized_set
    )
    return previous, promoted, len(crud.standby_set_versions(db, user_id, application_id))


def generate_template_set(
    db: Session, settings: Settings, *, user_id: str, application_id: str, uploads: dict[str, UploadFile]
) -> int:
    """Authorize, then add one complete new STANDBY set (every modality of the ACTIVE set) from the same captures.

    Refuses (409) when the pool already holds `TEMPLATE_POOL_SIZE` live sets.
    Key versions continue after the highest ever used, per modality.
    """
    if len(crud.live_set_versions(db, user_id, application_id)) >= settings.template_pool_size:
        raise crud.TemplatePoolExhaustedError("Template set pool is already full; nothing to generate.")
    embeddings = authorize_with_active_set(
        db, settings, user_id=user_id, application_id=application_id, uploads=uploads, action="generate"
    )
    try:
        pool = crud.get_set_rows(db, user_id, application_id)
        entries = {}
        for modality, embedding in embeddings.items():
            key_version = crud.next_key_version(db, user_id, modality, application_id)
            if modality == "hand":
                # One authorization gesture cannot replace five enrollment executions: the new set carries the ACTIVE
                # set's enrolled executions, re-keyed under the new key version (backend/services/hand_service.py).
                entries[modality] = get_service_for_modality("hand").rekey_entry(pool, user_id, application_id, key_version)
            else:
                entries[modality] = build_entry(
                    settings, embedding, user_id=user_id, application_id=application_id, modality=modality,
                    key_version=key_version,
                )
    finally:
        embeddings.clear()
    return crud.append_template_set(
        db,
        user_id=user_id,
        application_id=application_id,
        template_version=TEMPLATE_FORMAT_VERSION,
        output_bits=settings.template_bits,
        entries=entries,
    )
