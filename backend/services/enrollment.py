"""User enrollment profile.

The user - not the building - decides which modalities to enroll and which to present. A modality is enrolled exactly
when the user has a template for it in the ACTIVE template set: the profile is derived from the templates, so it can
never drift from them, and it lasts as long as they do (`DELETE /user/{id}` removes both).

Per-modality status (backend/states.py): NOT_REGISTERED, REGISTERED, UPDATED (re-enrolled), and RETRY_REQUIRED (voice:
the last enrollment failed its two-recording consistency check and stored nothing - remembered in `enrollment_events`).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.database import crud
from backend.states import (
    BIOMETRIC_MODALITIES,
    ENROLLMENT_INCONSISTENT,
    NOT_REGISTERED,
    REGISTERED,
    RETRY_REQUIRED,
    UPDATED,
)


@dataclass(frozen=True)
class EnrollmentStatus:
    user_id: str
    application_id: str
    #: {"face": bool, "voice": bool}.
    modalities: dict[str, bool]
    #: The four-value status per modality (same keys as `modalities`).
    statuses: dict[str, str]

    @property
    def enrolled(self) -> list[str]:
        return [modality for modality, is_enrolled in self.modalities.items() if is_enrolled]


def get_user_enrollment_status(db: Session, user_id: str, application_id: str) -> EnrollmentStatus:
    """Which modalities `user_id` has enrolled for `application_id`. Unknown users: everything NOT_REGISTERED."""
    rows = crud.get_set_rows(db, user_id, application_id)
    active = {row.modality for row in rows if row.is_active and _current_format(row)}
    updated = {row.modality for row in rows if row.revoked_reason == "re-enrolled"}

    modalities: dict[str, bool] = {}
    statuses: dict[str, str] = {}
    for modality in BIOMETRIC_MODALITIES:
        enrolled = modality in active
        modalities[modality] = enrolled
        if enrolled:
            statuses[modality] = UPDATED if modality in updated else REGISTERED
        elif crud.last_enrollment_outcome(db, user_id, application_id, modality) == ENROLLMENT_INCONSISTENT:
            statuses[modality] = RETRY_REQUIRED
        else:
            statuses[modality] = NOT_REGISTERED
    return EnrollmentStatus(user_id=user_id, application_id=application_id, modalities=modalities, statuses=statuses)


def _current_format(row) -> bool:
    """A hand template in a retired representation (e.g. the removed infinity gesture, hand_gesture_v1) can never be
    compared with a live Z gesture: the user is reported as not enrolled for the hand, i.e. must re-enroll it."""
    if row.modality != "hand":
        return True
    from preprocessing.hand_gesture import HAND_TEMPLATE_FORMAT

    return row.template_format == HAND_TEMPLATE_FORMAT


def missing_modalities(profile: EnrollmentStatus, submitted: list[str]) -> list[str]:
    """The submitted modalities the user has not enrolled (order preserved)."""
    return [modality for modality in submitted if not profile.modalities.get(modality, False)]
