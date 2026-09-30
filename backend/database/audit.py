"""CRUD for `backend.database.models.AuditLog`.

Kept separate from `crud.py` (which is about `User`/`ProtectedTemplate`) so
the "what gets written to the audit trail" surface is easy to review on its
own - exactly the kind of thing a privacy audit wants to check in one place.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.database.models import AuditLog


def record_attempt(
    db: Session,
    *,
    user_id: str,
    modality_list: list[str],
    similarity_scores: dict[str, float],
    thresholds_used: dict[str, float],
    authenticated: bool,
    latency_ms: int,
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
    failed_modalities: list[str] | None = None,
    active_set_before: int | None = None,
    active_set_after: int | None = None,
    template_rotation_triggered: bool | None = None,
    template_rotation_status: str | None = None,
    modality_statuses: dict[str, str] | None = None,
    attempt_status: str | None = None,
    modality_diagnostics: dict | None = None,
) -> AuditLog:
    """Insert one audit row. Never pass anything biometric-derived beyond a
    similarity score - see `AuditLog`'s docstring for what that means."""
    entry = AuditLog(
        user_id=user_id,
        building_id=building_id,
        modality_list=modality_list,
        similarity_scores=similarity_scores,
        thresholds_used=thresholds_used,
        fusion_score=fusion_score,
        face_similarity=similarity_scores.get("face"),
        voice_similarity=similarity_scores.get("voice"),
        hand_similarity=similarity_scores.get("hand"),
        fusion_similarity=fusion_similarity if fusion_similarity is not None else fusion_score,
        fusion_policy=fusion_policy,
        template_set_version=template_set_version,
        template_set_status=template_set_status,
        authentication_state=authentication_state,
        submitted_modalities=submitted_modalities,
        enrolled_modalities=enrolled_modalities,
        authenticated_modalities=authenticated_modalities,
        failed_modalities=failed_modalities,
        active_set_before=active_set_before,
        active_set_after=active_set_after,
        template_rotation_triggered=template_rotation_triggered,
        template_rotation_status=template_rotation_status,
        modality_statuses=modality_statuses,
        attempt_status=attempt_status,
        modality_diagnostics=modality_diagnostics,
        authenticated=authenticated,
        latency_ms=latency_ms,
        template_versions=template_versions,
        key_versions=key_versions,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_history_for_user(db: Session, user_id: str, limit: int = 50, offset: int = 0) -> list[AuditLog]:
    """Paginated, newest-first - what `GET /audit/{user_id}` returns."""
    statement = (
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.execute(statement).scalars().all())


def count_for_user(db: Session, user_id: str) -> int:
    return len(list(db.execute(select(AuditLog.audit_id).where(AuditLog.user_id == user_id)).scalars().all()))


def get_system_history(db: Session, limit: int = 50) -> list[AuditLog]:
    """Newest-first across every user - what `GET /audit/system` returns."""
    statement = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
    return list(db.execute(statement).scalars().all())


def delete_history_for_user(db: Session, user_id: str) -> int:
    """Delete every audit row for `user_id`; returns how many were removed."""
    existing = list(db.execute(select(AuditLog.audit_id).where(AuditLog.user_id == user_id)).scalars().all())
    if not existing:
        return 0
    db.execute(delete(AuditLog).where(AuditLog.user_id == user_id))
    db.commit()
    return len(existing)
