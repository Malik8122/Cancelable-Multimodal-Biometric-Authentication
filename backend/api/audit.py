"""GET /audit/{user_id}, GET /audit/system, DELETE /audit/{user_id}.

Server-side authentication history, backed by
`backend.database.models.AuditLog` (metadata only - see that model's
docstring for exactly what "metadata only" excludes: no raw image/audio, no
embedding, no protected template, ever).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database import audit as audit_crud
from backend.database.schema import AuditHistoryResponse, AuditLogEntry, DeleteAuditResponse, SystemAuditResponse
from backend.database.session import get_db
from backend.states import ACCESS_DENIED, ACCESS_GRANTED

logger = logging.getLogger("backend.api.audit")

router = APIRouter()


def _to_entry(row, settings: Settings) -> AuditLogEntry:
    """Per-modality similarities are stored in every row but only returned with DEBUG_SCORES=true."""
    debug = {}
    if settings.debug_scores:
        debug = dict(
            similarity_scores=row.similarity_scores,
            thresholds_used=row.thresholds_used,
            face_similarity=row.face_similarity,
            voice_similarity=row.voice_similarity,
            hand_similarity=row.hand_similarity,
            modality_diagnostics=row.modality_diagnostics,
            fusion_score=row.fusion_score,
            failed_modalities=row.failed_modalities,
        )
    return AuditLogEntry(
        audit_id=row.audit_id,
        timestamp=row.timestamp,
        user_id=row.user_id,
        building_id=row.building_id,
        modality_list=row.modality_list,
        fusion_similarity=row.fusion_similarity if row.fusion_similarity is not None else row.fusion_score,
        fusion_policy=row.fusion_policy,
        template_set_version=row.template_set_version,
        template_set_status=row.template_set_status,
        authentication_state=row.authentication_state or (ACCESS_GRANTED if row.authenticated else ACCESS_DENIED),
        submitted_modalities=row.submitted_modalities,
        enrolled_modalities=row.enrolled_modalities,
        authenticated_modalities=row.authenticated_modalities,
        authenticated=row.authenticated,
        latency_ms=row.latency_ms,
        template_versions=row.template_versions,
        key_versions=row.key_versions,
        active_set_before=row.active_set_before,
        active_set_after=row.active_set_after,
        template_rotation_triggered=row.template_rotation_triggered,
        template_rotation_status=row.template_rotation_status,
        modality_statuses=row.modality_statuses,
        attempt_status=row.attempt_status,
        **debug,
    )


@router.get("/audit/system", response_model=SystemAuditResponse)
def get_system_audit(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SystemAuditResponse:
    """Latest authentication attempts across every user, newest first."""
    rows = audit_crud.get_system_history(db, limit=limit)
    return SystemAuditResponse(entries=[_to_entry(row, settings) for row in rows])


@router.get("/audit/{user_id}", response_model=AuditHistoryResponse)
def get_user_audit(
    user_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuditHistoryResponse:
    """Paginated authentication history for one user, newest first."""
    rows = audit_crud.get_history_for_user(db, user_id, limit=limit, offset=offset)
    total = audit_crud.count_for_user(db, user_id)
    return AuditHistoryResponse(
        user_id=user_id,
        total=total,
        limit=limit,
        offset=offset,
        entries=[_to_entry(row, settings) for row in rows],
    )


@router.delete("/audit/{user_id}", response_model=DeleteAuditResponse)
def delete_user_audit(user_id: str, db: Session = Depends(get_db)) -> DeleteAuditResponse:
    entries_deleted = audit_crud.delete_history_for_user(db, user_id)
    logger.info("Deleted audit history user_id=%s entries_deleted=%d", user_id, entries_deleted)
    return DeleteAuditResponse(success=True, user_id=user_id, entries_deleted=entries_deleted)
