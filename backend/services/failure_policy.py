"""Failure handling for a DENIED attempt: retry, or escalate a suspicious attempt to the security response.

A biometric mismatch is not evidence of an attack. So a denied attempt is answered in one of two ways:

- RETRY_REQUESTED: the failed modalities should be presented again. Nothing about the templates changes. This covers a
  CAPTURE_ERROR (the sample could not be processed), QUALITY_INSUFFICIENT (processed, but failing the modality's
  existing quality gate) and an isolated VERIFICATION_MISMATCH (a usable capture that did not match).
- SUSPICIOUS_ATTEMPT: one modality produced more than `Settings.max_modality_retries` CONSECUTIVE high-quality
  mismatches against the same ACTIVE set within `Settings.suspicious_mismatch_window_seconds`. Only then is the existing
  security response (`authentication.rotate_after_failed_authentication`) invoked.

The count is read from the audit log (one row per attempt, `AuditLog.modality_statuses`), newest first, per modality.
It stops at the first row that ended the streak: a successful authentication, a template rotation, or an attempt that
was evaluated against a different template set (so a rotation or manual revocation starts a new count). A row in which
that modality VERIFIED also stops it; capture/quality failures and attempts that did not present the modality are
skipped - they neither count nor reset. Rows without statuses (older rows, template-management authorizations) are
skipped too. Status names only are stored; nothing biometric.

Attempt numbering (default MAX_MODALITY_RETRIES=2): the 1st and 2nd consecutive high-quality mismatch of a modality
are RETRY_REQUESTED ("attempt 1 of 3", "attempt 2 of 3"); the 3rd is a SUSPICIOUS_ATTEMPT and invokes the existing
template rotation. `Assessment.mismatch_attempt` / `mismatch_attempt_limit` carry these numbers to the client.

Repeated unusable captures (CAPTURE_ERROR / QUALITY_INSUFFICIENT) never count as mismatches and never rotate templates.
They are bounded separately: after `Settings.max_consecutive_capture_failures` consecutive attempts with an unusable
capture (within the same window), further attempts are refused for `capture_failure_cooldown_seconds`
(`capture_cooldown_remaining`, CAPTURE_COOLDOWN, HTTP 429) - a rate limit, not a security response.

The fusion policy is untouched: it alone decides ACCESS_GRANTED / ACCESS_DENIED, afresh for every attempt (a modality
verified in one attempt is never carried into the next). This module only decides what the system does after a denial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import Settings
from backend.database.models import AuditLog
from backend.states import (
    CAPTURE_ERROR,
    QUALITY_INSUFFICIENT,
    RETRY_REQUESTED,
    SUSPICIOUS_ATTEMPT,
    VERIFICATION_MISMATCH,
    VERIFIED,
)

#: How many recent audit rows are scanned for a streak - far more than any retry limit a deployment would set.
_SCAN_LIMIT = 200

_LABEL = {"face": "Face", "voice": "Voice", "hand": "Hand gesture"}

#: Face enrollment-gate verdicts (embeddings/pipelines.py::FacePipeline.check_capture) -> a user-facing reason.
_FACE_QUALITY_REASON = {
    "NO_FACE": "no face was detected",
    "MULTIPLE_FACES": "more than one face was in the frame",
    "LOW_CONFIDENCE": "the face detection confidence was low",
    "BLURRY": "the image was blurry",
    "TOO_SMALL": "the face was too far from the camera",
    "OFF_CENTER": "the face was not centred",
    "TOO_ANGLED": "the head was tilted or turned",
    "LANDMARK_FAILURE": "facial landmarks could not be located",
    "ALIGNMENT_FAILED": "the face could not be aligned",
}


@dataclass
class Assessment:
    status: str
    retry_modalities: list[str] = field(default_factory=list)
    retries_remaining: int = 0
    #: Modalities whose consecutive high-quality mismatches reached the escalation condition.
    escalated_modalities: list[str] = field(default_factory=list)
    #: {modality: consecutive high-quality mismatches, including this attempt} for the mismatched modalities.
    mismatch_streaks: dict[str, int] = field(default_factory=dict)
    #: Which consecutive high-quality mismatch this attempt was (the highest over its modalities; 0 = none), and the
    #: attempt number that escalates (MAX_MODALITY_RETRIES + 1).
    mismatch_attempt: int = 0
    mismatch_attempt_limit: int = 0
    #: Consecutive attempts with an unusable capture, including this one (0 = none), and the cooldown limit.
    capture_failures: int = 0
    capture_failure_limit: int = 0


def face_quality_reason(verdict: str) -> str:
    return _FACE_QUALITY_REASON.get(verdict, "the capture did not pass the quality check")


def _aware(timestamp: datetime) -> datetime:
    """SQLite returns naive datetimes for timezone-aware columns; every stored timestamp is UTC."""
    return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)


def prior_mismatch_streaks(
    db: Session, settings: Settings, *, user_id: str, active_set: int, modalities: list[str], now: datetime | None = None
) -> dict[str, int]:
    """{modality: consecutive high-quality mismatches before this attempt} on `active_set` (see the module docstring)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=settings.suspicious_mismatch_window_seconds)
    rows = db.execute(
        select(AuditLog).where(AuditLog.user_id == user_id).order_by(AuditLog.timestamp.desc()).limit(_SCAN_LIMIT)
    ).scalars().all()
    streaks: dict[str, int] = {}
    for modality in modalities:
        count = 0
        for row in rows:
            if _aware(row.timestamp) < cutoff:
                break
            if row.modality_statuses is None:
                continue
            if row.authenticated or row.template_rotation_triggered or row.template_set_version != active_set:
                break
            status = row.modality_statuses.get(modality)
            if status == VERIFIED:
                break
            if status == VERIFICATION_MISMATCH:
                count += 1
        streaks[modality] = count
    return streaks


def _capture_failed(statuses: dict | None) -> bool:
    return bool(statuses) and any(v in (CAPTURE_ERROR, QUALITY_INSUFFICIENT) for v in statuses.values())


def prior_capture_failures(db: Session, settings: Settings, *, user_id: str, now: datetime | None = None) -> tuple[int, datetime | None]:
    """(consecutive recent attempts with an unusable capture, time of the latest one) before this attempt.

    Newest first within the window: an attempt with a CAPTURE_ERROR / QUALITY_INSUFFICIENT modality counts; a
    successful authentication or an attempt whose captures were all usable ends the run; rows without statuses
    (cooldown refusals, older rows, management authorizations) are skipped.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=settings.suspicious_mismatch_window_seconds)
    rows = db.execute(
        select(AuditLog).where(AuditLog.user_id == user_id).order_by(AuditLog.timestamp.desc()).limit(_SCAN_LIMIT)
    ).scalars().all()
    count, latest = 0, None
    for row in rows:
        if _aware(row.timestamp) < cutoff:
            break
        if row.modality_statuses is None:
            continue
        if row.authenticated or not _capture_failed(row.modality_statuses):
            break
        count += 1
        latest = latest or _aware(row.timestamp)
    return count, latest


def capture_cooldown_remaining(db: Session, settings: Settings, *, user_id: str, now: datetime | None = None) -> int:
    """Seconds this user must still wait after too many consecutive unusable captures (0 = may attempt now)."""
    limit = settings.max_consecutive_capture_failures
    if limit <= 0:
        return 0
    now = now or datetime.now(timezone.utc)
    count, latest = prior_capture_failures(db, settings, user_id=user_id, now=now)
    if count < limit or latest is None:
        return 0
    remaining = settings.capture_failure_cooldown_seconds - (now - latest).total_seconds()
    return max(0, int(-(-remaining // 1)))  # ceil


def assess_denied_attempt(
    db: Session, settings: Settings, *, user_id: str, active_set: int, statuses: dict[str, str]
) -> Assessment:
    """RETRY_REQUESTED or SUSPICIOUS_ATTEMPT for a DENIED attempt with these per-modality statuses."""
    failed = [m for m, s in statuses.items() if s != VERIFIED]
    mismatched = [m for m, s in statuses.items() if s == VERIFICATION_MISMATCH]
    prior = prior_mismatch_streaks(db, settings, user_id=user_id, active_set=active_set, modalities=mismatched)
    streaks = {m: prior[m] + 1 for m in mismatched}
    limit = settings.max_modality_retries
    escalated = sorted(m for m, n in streaks.items() if n > limit)
    remaining = min((limit - n for n in streaks.values()), default=limit)
    captures = prior_capture_failures(db, settings, user_id=user_id)[0] + 1 if _capture_failed(statuses) else 0
    numbers = {"mismatch_streaks": streaks, "mismatch_attempt": max(streaks.values(), default=0),
               "mismatch_attempt_limit": limit + 1, "capture_failures": captures,
               "capture_failure_limit": settings.max_consecutive_capture_failures}
    if escalated:
        return Assessment(status=SUSPICIOUS_ATTEMPT, escalated_modalities=escalated, retries_remaining=0, **numbers)
    return Assessment(status=RETRY_REQUESTED, retry_modalities=sorted(failed), retries_remaining=max(0, remaining), **numbers)


def _names(modalities: list[str]) -> str:
    names = [_LABEL.get(m, m.capitalize()) for m in modalities]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def describe(assessment: Assessment, statuses: dict[str, str], details: dict[str, str], rotation_status: str | None,
             settings: Settings | None = None) -> dict:
    """User-facing wording for the assessment (headline, reason, per-modality messages and hints, attempt notice,
    security response).

    Deliberately factual: a capture problem is "capture was unsuccessful", a mismatch is "verification did not match",
    an escalation is a "suspicious verification pattern" - never "attacker" or "attack detected" (there is no
    presentation-attack detection here).
    """
    messages: dict[str, str] = {}
    hints: dict[str, str] = {}
    for modality, status in statuses.items():
        label = _LABEL.get(modality, modality.capitalize())
        if status == VERIFIED:
            messages[modality] = f"{label} verified."
        elif status in (CAPTURE_ERROR, QUALITY_INSUFFICIENT):
            messages[modality] = f"{label} capture was unsuccessful. Please try again."
            if details.get(modality):
                hints[modality] = details[modality][0].upper() + details[modality][1:].rstrip(".") + "."
        else:
            messages[modality] = f"{label} verification did not match. Please try again."

    n, limit = assessment.mismatch_attempt, assessment.mismatch_attempt_limit
    if assessment.status == SUSPICIOUS_ATTEMPT:
        escalated = assessment.escalated_modalities
        response = {
            "ROTATED": "cancellable template rotation triggered.",
            "ALREADY_ROTATED": "The template set was already rotated by a concurrent attempt; no second rotation.",
            "POOL_EXHAUSTED": "No standby template set left; the current set stays active. Re-enrollment creates new sets.",
            "ROTATION_FAILED": "Template rotation could not be completed; the current set stays active.",
            "DISABLED": "Template rotation after suspicious attempts is disabled in this deployment; no template changed.",
        }.get(rotation_status or "", "Security response invoked.")
        return {
            "headline": "Authentication failed — suspicious verification pattern detected.",
            "reason": f"Repeated high-quality {_names(escalated).lower()} verification mismatch.",
            "modality_messages": messages,
            "modality_hints": hints,
            "attempt_notice": f"Consecutive high-quality mismatch {n} of {limit}: the security response was invoked.",
            "security_response": response,
        }

    failed = assessment.retry_modalities
    if all(statuses[m] in (CAPTURE_ERROR, QUALITY_INSUFFICIENT) for m in failed):
        headline = f"{_names(failed)} capture was unsuccessful."
        reason = "Please try again."
    else:
        headline = "Authentication incomplete."
        mismatched = [m for m in failed if statuses[m] == VERIFICATION_MISMATCH]
        reason = f"{_names(mismatched)} verification did not match. Please try again."
    if n:
        left = limit - n
        notice = (f"Consecutive high-quality mismatch {n} of {limit}. {left} more consecutive mismatch"
                  f"{'es' if left != 1 else ''} will trigger the security response; a successful verification resets the count.")
    else:
        notice = "Unusable captures are not counted as verification mismatches and never change a template."
    if assessment.capture_failures and assessment.capture_failure_limit:
        pause = f"{settings.capture_failure_cooldown_seconds} s" if settings else "a short"
        notice += (f" Unusable capture {assessment.capture_failures} of {assessment.capture_failure_limit} in a row; after"
                   f" {assessment.capture_failure_limit} a {pause} pause is required.")
    return {"headline": headline, "reason": reason, "modality_messages": messages, "modality_hints": hints,
            "attempt_notice": notice, "security_response": "No template rotation."}
