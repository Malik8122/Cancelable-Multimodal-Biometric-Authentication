"""Plain CRUD functions over `backend/database/models.py`.

Every function takes a `Session` explicitly (from `backend/database/session.py::get_db`)
rather than reading a global - usable both from `backend/services/*.py` (one
session per HTTP request) and directly from tests.

Template sets (docs/MULTI_TEMPLATE_ARCHITECTURE.md): a user's credential for
one application is a pool of TEMPLATE SETS. A set (identified by
`template_set_version`) holds one protected template per enrolled modality.
Exactly one set is ACTIVE; the others are STANDBY or REVOKED. Activation and
revocation always move a whole set, in one transaction, so face and voice
templates change status together and a set is never mixed with
another.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database.models import (
    MASTER_SECRET_FINGERPRINT_ID,
    STATUS_ACTIVE,
    STATUS_REVOKED,
    STATUS_STANDBY,
    EnrollmentEvent,
    MasterSecretFingerprint,
    ProtectedTemplate,
    User,
)
from backend.display_names import fallback_display_name


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConcurrentEnrollmentError(RuntimeError):
    """Two racing requests both tried to change the active template set (mapped to HTTP 409).

    The partial unique indexes on `models.py::ProtectedTemplate` turn the
    second, non-atomic read-then-write into an `IntegrityError`, which this
    wraps into a retryable backend-layer error.
    """


class TemplatePoolExhaustedError(RuntimeError):
    """No STANDBY template set is left to promote / no room for another set (mapped to HTTP 409)."""


class ActiveSetChangedError(RuntimeError):
    """The set a caller expected to be ACTIVE no longer is: a concurrent request already rotated it."""

    def __init__(self, current_active: int | None):
        super().__init__(f"The active template set changed concurrently (now {current_active}).")
        self.current_active = current_active


class TemplateNotFoundError(LookupError):
    """Nothing enrolled / no such standby set (mapped to HTTP 404)."""


@dataclass(frozen=True)
class PoolEntry:
    """One template to persist: the HKDF key version it was generated under, and its packed bits.

    The optional fields override the call's defaults for this one row - used by modalities whose template is not a
    BioHash (the hand gesture: its own format, version, gesture type and metadata).
    """

    key_version: int
    protected_template: bytes
    template_version: int | None = None
    output_bits: int | None = None
    template_format: str | None = None
    gesture_type: str | None = None
    template_metadata: dict | None = None


def count_protected_templates(db: Session) -> int:
    """Total protected-template rows in this database, across every user/modality/application.

    Used only by the MASTER_SECRET key-continuity check (`backend/key_continuity.py`) to decide
    whether "no fingerprint recorded" is safe to auto-initialize (nothing to protect yet) or a
    hard-fail condition (existing templates whose protecting secret can no longer be verified).
    """
    return db.execute(select(func.count()).select_from(ProtectedTemplate)).scalar_one()


def get_master_secret_fingerprint(db: Session) -> MasterSecretFingerprint | None:
    """The single global MASTER_SECRET fingerprint row, if one has been recorded yet."""
    return db.get(MasterSecretFingerprint, MASTER_SECRET_FINGERPRINT_ID)


def initialize_master_secret_fingerprint(db: Session, fingerprint: bytes) -> MasterSecretFingerprint:
    """Record the MASTER_SECRET fingerprint for the very first time.

    Startup-only (`backend/key_continuity.py`), and only ever invoked when no protected templates
    exist yet - there is nothing at stake for an auto-initialize to get wrong. Raises if a row
    already exists rather than overwriting it: replacing an existing fingerprint is
    `rotate_master_secret_fingerprint`'s job alone, gated behind the operator's explicit,
    deliberate confirmation (`scripts/rotate_master_secret.py`).
    """
    if get_master_secret_fingerprint(db) is not None:
        raise ValueError("A MASTER_SECRET fingerprint is already recorded; refusing to overwrite it implicitly.")
    row = MasterSecretFingerprint(id=MASTER_SECRET_FINGERPRINT_ID, fingerprint=fingerprint)
    db.add(row)
    db.commit()
    return row


def rotate_master_secret_fingerprint(db: Session, fingerprint: bytes) -> MasterSecretFingerprint:
    """Overwrite (or create) the MASTER_SECRET fingerprint. `scripts/rotate_master_secret.py` ONLY.

    This is the one deliberate exception to `initialize_master_secret_fingerprint`'s
    never-overwrite rule: it exists so an operator can explicitly acknowledge a genuine key
    rotation (e.g. an unrecoverable old secret) after making that decision themselves - never as
    a side effect of normal server startup.
    """
    row = get_master_secret_fingerprint(db)
    if row is None:
        row = MasterSecretFingerprint(id=MASTER_SECRET_FINGERPRINT_ID, fingerprint=fingerprint)
        db.add(row)
    else:
        row.fingerprint = fingerprint
        row.created_at = _now()
    db.commit()
    return row


@dataclass(frozen=True)
class TemplateSetSummary:
    """Metadata of one template set (never template bytes)."""

    version: int
    status: str
    modalities: list[str]
    key_versions: dict[str, int]
    group_id: str | None
    created_at: datetime | None
    activated_at: datetime | None
    revoked_at: datetime | None
    revoked_reason: str | None


# ----------------------------------------------------------------------------- users


def get_or_create_user(db: Session, user_id: str, username: str | None = None) -> User:
    user = db.get(User, user_id)
    if user is not None:
        return user
    user = User(id=user_id, username=username)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: str) -> User | None:
    return db.get(User, user_id)


def create_named_user(db: Session, display_name: str) -> User:
    """A new user with a server-generated internal id and an already-validated display name.

    The id (not the name) identifies the user everywhere, so the same name may be registered any number of times.
    """
    while True:
        user_id = f"USER-{uuid4().hex[:12].upper()}"
        if db.get(User, user_id) is None:
            break
    user = User(id=user_id, username=display_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def set_display_name(db: Session, user_id: str, display_name: str) -> User | None:
    """Name (or rename) an existing user - e.g. one enrolled before names existed. None if there is no such user."""
    user = db.get(User, user_id)
    if user is None:
        return None
    user.username = display_name
    db.commit()
    db.refresh(user)
    return user


def display_name_for(db: Session, user_id: str) -> str:
    """The user's display name, or the "User <short id>" fallback when none was stored."""
    user = db.get(User, user_id)
    return user.username if user is not None and user.username else fallback_display_name(user_id)


# ----------------------------------------------------------------------------- reads


def get_active_template(db: Session, user_id: str, modality: str, application_id: str) -> ProtectedTemplate | None:
    """The modality's row inside the ACTIVE template set (the only one authentication reads)."""
    statement = select(ProtectedTemplate).where(
        ProtectedTemplate.user_id == user_id,
        ProtectedTemplate.modality == modality,
        ProtectedTemplate.application_id == application_id,
        ProtectedTemplate.is_active.is_(True),
    )
    return db.execute(statement).scalar_one_or_none()


def get_set_rows(db: Session, user_id: str, application_id: str) -> list[ProtectedTemplate]:
    """Every template row (any set, any status, any modality) for one (user, application)."""
    statement = (
        select(ProtectedTemplate)
        .where(ProtectedTemplate.user_id == user_id, ProtectedTemplate.application_id == application_id)
        .order_by(ProtectedTemplate.template_set_version, ProtectedTemplate.modality, ProtectedTemplate.key_version)
    )
    return list(db.execute(statement).scalars().all())


def get_active_set_rows(db: Session, user_id: str, application_id: str) -> list[ProtectedTemplate]:
    return [row for row in get_set_rows(db, user_id, application_id) if row.is_active]


def max_key_version(db: Session, user_id: str, modality: str, application_id: str) -> int:
    """Highest key_version ever used for this (user, modality, application), revoked rows included (0 if none).

    Revoked key versions are never reused, so a compromised key can't come
    back through re-enrollment or a newly generated set.
    """
    statement = select(func.max(ProtectedTemplate.key_version)).where(
        ProtectedTemplate.user_id == user_id,
        ProtectedTemplate.modality == modality,
        ProtectedTemplate.application_id == application_id,
    )
    return db.execute(statement).scalar_one() or 0


def next_key_version(db: Session, user_id: str, modality: str, application_id: str) -> int:
    return max_key_version(db, user_id, modality, application_id) + 1


def max_set_version(db: Session, user_id: str, application_id: str) -> int:
    statement = select(func.max(ProtectedTemplate.template_set_version)).where(
        ProtectedTemplate.user_id == user_id, ProtectedTemplate.application_id == application_id
    )
    return db.execute(statement).scalar_one() or 0


def _group_by_set(rows: list[ProtectedTemplate]) -> dict[int, list[ProtectedTemplate]]:
    grouped: dict[int, list[ProtectedTemplate]] = {}
    for row in rows:
        grouped.setdefault(row.template_set_version, []).append(row)
    return grouped


def _set_status(rows: list[ProtectedTemplate]) -> str:
    return rows[0].template_set_status


def live_set_versions(db: Session, user_id: str, application_id: str) -> list[int]:
    """Versions of the ACTIVE and STANDBY sets, ascending."""
    grouped = _group_by_set(get_set_rows(db, user_id, application_id))
    return sorted(version for version, rows in grouped.items() if _set_status(rows) != STATUS_REVOKED)


def standby_set_versions(db: Session, user_id: str, application_id: str) -> list[int]:
    """STANDBY set versions in promotion order: oldest (lowest version) first."""
    grouped = _group_by_set(get_set_rows(db, user_id, application_id))
    return sorted(version for version, rows in grouped.items() if _set_status(rows) == STATUS_STANDBY)


def get_template_sets(db: Session, user_id: str, application_id: str) -> list[TemplateSetSummary]:
    summaries = []
    for version, rows in sorted(_group_by_set(get_set_rows(db, user_id, application_id)).items()):
        key_versions: dict[str, int] = {}
        for row in rows:
            key_versions[row.modality] = max(key_versions.get(row.modality, 0), row.key_version)
        first = rows[0]
        summaries.append(
            TemplateSetSummary(
                version=version,
                status=first.template_set_status,
                modalities=sorted({row.modality for row in rows}),
                key_versions=key_versions,
                group_id=first.template_group_id,
                created_at=first.template_set_created_at or first.created_at,
                activated_at=first.template_set_activated_at,
                revoked_at=first.template_set_revoked_at,
                revoked_reason=next((row.revoked_reason for row in rows if row.template_set_revoked_at), None),
            )
        )
    return summaries


def record_enrollment_event(db: Session, *, user_id: str, application_id: str, modality: str, outcome: str) -> None:
    db.add(EnrollmentEvent(user_id=user_id, application_id=application_id, modality=modality, outcome=outcome))
    db.commit()


def last_enrollment_outcome(db: Session, user_id: str, application_id: str, modality: str) -> str | None:
    statement = (
        select(EnrollmentEvent.outcome)
        .where(
            EnrollmentEvent.user_id == user_id,
            EnrollmentEvent.application_id == application_id,
            EnrollmentEvent.modality == modality,
        )
        .order_by(EnrollmentEvent.created_at.desc())
        .limit(1)
    )
    return db.execute(statement).scalar_one_or_none()


def get_application_ids(db: Session, user_id: str) -> list[str]:
    statement = (
        select(ProtectedTemplate.application_id)
        .where(ProtectedTemplate.user_id == user_id)
        .distinct()
        .order_by(ProtectedTemplate.application_id)
    )
    return list(db.execute(statement).scalars().all())


# ----------------------------------------------------------------------------- writes


def _commit_or_conflict(db: Session, user_id: str, application_id: str) -> None:
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ConcurrentEnrollmentError(
            f"Another request already changed the template sets of user_id={user_id!r}, "
            f"application_id={application_id!r} concurrently; retry."
        ) from error


def plan_enrollment_sets(db: Session, user_id: str, application_id: str, pool_size: int) -> list[int]:
    """Which template set versions the next modality enrollment must write into.

    The existing live (ACTIVE + STANDBY) sets, so a newly enrolled or
    re-enrolled modality joins every set that can still authenticate; for a
    user with no live sets, `pool_size` brand-new sets (versions continue
    after the highest ever used).
    """
    live = live_set_versions(db, user_id, application_id)
    if live:
        return live
    start = max_set_version(db, user_id, application_id) + 1
    return list(range(start, start + pool_size))


def _row_status_for_set(set_status: str) -> str:
    return STATUS_ACTIVE if set_status == STATUS_ACTIVE else STATUS_STANDBY


def save_modality_templates(
    db: Session,
    *,
    user_id: str,
    application_id: str,
    modality: str,
    template_version: int,
    output_bits: int,
    entries: dict[int, PoolEntry],
) -> list[ProtectedTemplate]:
    """Write one modality's template into each set in `entries` ({set_version: entry}).

    Sets that already exist keep their status and metadata (the new row joins
    them, replacing - as REVOKED, reason "re-enrolled" - any live template
    this modality already had there). Sets that don't exist yet are created:
    the lowest new version ACTIVE if the user has no ACTIVE set, the rest
    STANDBY. One transaction.
    """
    if not entries:
        raise ValueError("At least one template set entry is required.")

    all_rows = get_set_rows(db, user_id, application_id)
    grouped = _group_by_set(all_rows)
    has_active_set = any(_set_status(rows) == STATUS_ACTIVE for rows in grouped.values())

    now = _now()
    for row in all_rows:
        if row.modality == modality and row.template_set_version in entries and row.template_status != STATUS_REVOKED:
            row.template_status = STATUS_REVOKED
            row.is_active = False
            row.revoked_time = now
            row.revoked_reason = "re-enrolled"
    db.flush()

    created: list[ProtectedTemplate] = []
    new_versions = sorted(version for version in entries if version not in grouped)
    first_new_is_active = not has_active_set
    for version in sorted(entries):
        entry = entries[version]
        if version in grouped:
            reference = grouped[version][0]
            set_status = reference.template_set_status
            set_created, set_activated, set_revoked = (
                reference.template_set_created_at,
                reference.template_set_activated_at,
                reference.template_set_revoked_at,
            )
            group_id = reference.template_group_id
        else:
            is_active_set = first_new_is_active and version == new_versions[0]
            set_status = STATUS_ACTIVE if is_active_set else STATUS_STANDBY
            set_created, set_activated, set_revoked = now, (now if is_active_set else None), None
            group_id = str(uuid4())
        row_status = STATUS_REVOKED if set_status == STATUS_REVOKED else _row_status_for_set(set_status)
        row = ProtectedTemplate(
            user_id=user_id,
            application_id=application_id,
            modality=modality,
            template_version=entry.template_version if entry.template_version is not None else template_version,
            key_version=entry.key_version,
            output_bits=entry.output_bits if entry.output_bits is not None else output_bits,
            protected_template=entry.protected_template,
            template_format=entry.template_format,
            gesture_type=entry.gesture_type,
            template_metadata=entry.template_metadata,
            is_active=row_status == STATUS_ACTIVE,
            template_status=row_status,
            activation_time=set_activated if row_status == STATUS_ACTIVE else None,
            template_group_id=group_id,
            template_index=version,
            template_set_version=version,
            template_set_status=set_status,
            template_set_created_at=set_created,
            template_set_activated_at=set_activated,
            template_set_revoked_at=set_revoked,
        )
        db.add(row)
        created.append(row)
    _commit_or_conflict(db, user_id, application_id)
    for row in created:
        db.refresh(row)
    return created


def append_template_set(
    db: Session,
    *,
    user_id: str,
    application_id: str,
    template_version: int,
    output_bits: int,
    entries: dict[str, PoolEntry],
) -> int:
    """Add one complete new STANDBY template set ({modality: entry}); returns its version."""
    if get_active_set_rows(db, user_id, application_id) == []:
        raise TemplateNotFoundError(f"No active template set for user_id={user_id!r}.")
    version = max_set_version(db, user_id, application_id) + 1
    now = _now()
    group_id = str(uuid4())
    for modality, entry in sorted(entries.items()):
        db.add(
            ProtectedTemplate(
                user_id=user_id,
                application_id=application_id,
                modality=modality,
                template_version=entry.template_version if entry.template_version is not None else template_version,
                key_version=entry.key_version,
                output_bits=entry.output_bits if entry.output_bits is not None else output_bits,
                protected_template=entry.protected_template,
                template_format=entry.template_format,
                gesture_type=entry.gesture_type,
                template_metadata=entry.template_metadata,
                is_active=False,
                template_status=STATUS_STANDBY,
                template_group_id=group_id,
                template_index=version,
                template_set_version=version,
                template_set_status=STATUS_STANDBY,
                template_set_created_at=now,
            )
        )
    _commit_or_conflict(db, user_id, application_id)
    return version


def _activate_set(rows: list[ProtectedTemplate]) -> None:
    now = _now()
    for row in rows:
        row.template_set_status = STATUS_ACTIVE
        row.template_set_activated_at = now
        if row.template_status != STATUS_REVOKED:
            row.template_status = STATUS_ACTIVE
            row.is_active = True
            row.activation_time = now


def revoke_active_set_and_promote(
    db: Session, user_id: str, application_id: str, reason: str = "revoked by user"
) -> tuple[int, int]:
    """ACTIVE set -> REVOKED and the oldest STANDBY set -> ACTIVE, for every modality at once.

    Returns (revoked_version, promoted_version). Raises `TemplateNotFoundError`
    with nothing enrolled and `TemplatePoolExhaustedError` with no STANDBY set;
    in the latter case nothing changes, so the ACTIVE set keeps working.
    """
    return rotate_expected_or_conflict(db, user_id, application_id, expected_active=None, reason=reason)


def rotate_expected_or_conflict(
    db: Session, user_id: str, application_id: str, *, expected_active: int | None, reason: str,
    promote_version: int | None = None,
) -> tuple[int, int]:
    """`rotate_active_set_if_current` for callers that must treat a concurrent change as a conflict.

    `expected_active` = the set the caller authorized against (None: whatever is ACTIVE now). If a concurrent request
    already changed the ACTIVE set, nothing is rotated and `ConcurrentEnrollmentError` (HTTP 409) is raised.
    """
    if expected_active is None:
        expected_active = active_set_version(db, user_id, application_id)
        if expected_active is None:
            raise TemplateNotFoundError(f"No active template set for user_id={user_id!r}.")
    try:
        return rotate_active_set_if_current(db, user_id, application_id, expected_active, reason, promote_version)
    except ActiveSetChangedError as changed:
        raise ConcurrentEnrollmentError(
            f"The active template set of user_id={user_id!r} changed concurrently (now {changed.current_active}); "
            f"nothing was changed - retry."
        ) from changed


def activate_standby_set(db: Session, user_id: str, application_id: str, version: int) -> tuple[int | None, int]:
    """Make STANDBY set `version` the ACTIVE one; the previous ACTIVE set becomes REVOKED.

    Never demotes to STANDBY, so a retired key can't be resurrected. Returns
    (previous_active_version_or_None, new_active_version).
    """
    return activate_standby_set_if_current(db, user_id, application_id, version, expected_active=None)


def activate_standby_set_if_current(
    db: Session, user_id: str, application_id: str, version: int, *, expected_active: int | None
) -> tuple[int | None, int]:
    """`activate_standby_set` with the ACTIVE set the caller authorized against (None: whatever is ACTIVE now)."""
    grouped = _group_by_set(get_set_rows(db, user_id, application_id))
    if version not in grouped or _set_status(grouped[version]) != STATUS_STANDBY:
        raise TemplateNotFoundError(f"No STANDBY template set with version {version} for user_id={user_id!r}.")
    previous = next((v for v, rows in grouped.items() if _set_status(rows) == STATUS_ACTIVE), None)
    if previous is None:  # nothing ACTIVE to retire (not reachable through the authorized endpoint)
        _activate_set(grouped[version])
        _commit_or_conflict(db, user_id, application_id)
        return None, version
    return rotate_expected_or_conflict(
        db, user_id, application_id, expected_active=expected_active if expected_active is not None else previous,
        reason="superseded by manual activation", promote_version=version,
    )


def save_template(
    db: Session,
    *,
    user_id: str,
    modality: str,
    application_id: str,
    template_version: int,
    key_version: int,
    output_bits: int,
    protected_template: bytes,
) -> ProtectedTemplate:
    """Legacy single-template path: one new ACTIVE set (version 1 if none) holding just this modality.

    Kept for callers/tests that predate template sets; enrollment goes through
    `save_modality_templates`.
    """
    previous = get_active_template(db, user_id, modality, application_id)
    version = previous.template_set_version if previous else max(max_set_version(db, user_id, application_id), 1)
    rows = save_modality_templates(
        db,
        user_id=user_id,
        application_id=application_id,
        modality=modality,
        template_version=template_version,
        output_bits=output_bits,
        entries={version: PoolEntry(key_version=key_version, protected_template=protected_template)},
    )
    return rows[0]


def get_templates_for_user(db: Session, user_id: str, active_only: bool = True) -> list[ProtectedTemplate]:
    statement = select(ProtectedTemplate).where(ProtectedTemplate.user_id == user_id)
    if active_only:
        statement = statement.where(ProtectedTemplate.is_active.is_(True))
    return list(db.execute(statement).scalars().all())


def delete_user_templates(db: Session, user_id: str) -> int:
    """Delete a user and every protected template they own; returns how many templates were removed."""
    user = get_user(db, user_id)
    if user is None:
        return 0
    templates_deleted = len(user.templates)
    db.execute(delete(EnrollmentEvent).where(EnrollmentEvent.user_id == user_id))
    db.delete(user)
    db.commit()
    return templates_deleted


def active_set_version(db: Session, user_id: str, application_id: str) -> int | None:
    return db.execute(
        select(func.max(ProtectedTemplate.template_set_version)).where(
            ProtectedTemplate.user_id == user_id, ProtectedTemplate.application_id == application_id,
            ProtectedTemplate.template_set_status == STATUS_ACTIVE)
    ).scalar_one()


def rotate_active_set_if_current(
    db: Session, user_id: str, application_id: str, expected_active: int, reason: str, promote_version: int | None = None
) -> tuple[int, int]:
    """Compare-and-swap rotation: set `expected_active` ACTIVE -> REVOKED and the oldest STANDBY set -> ACTIVE, for every
    modality, in ONE transaction - but only if `expected_active` is still the ACTIVE set.

    Concurrency: the first statement is a conditional UPDATE on the expected set's rows. It takes the database write
    lock (SQLite RESERVED lock / PostgreSQL row locks) before anything else is read, so concurrent rotators serialize;
    a later one re-evaluates `WHERE template_set_status = 'ACTIVE'` against the committed state, matches 0 rows and
    raises `ActiveSetChangedError` without changing anything. So N simultaneous failures evaluated against the same set
    cause exactly ONE rotation, no set is skipped, and all modalities move in the same commit. With no STANDBY set the
    claim is rolled back and `TemplatePoolExhaustedError` is raised (the ACTIVE set keeps working).

    This is the ONE rotation primitive: automatic rotation after a failed authentication, `/revoke-template`
    (`revoke_active_set_and_promote`) and manual activation (`activate_standby_set`, `promote_version` = the chosen
    STANDBY set; `TemplateNotFoundError` if it is not STANDBY) all go through it.
    """
    now = _now()
    of_user = (ProtectedTemplate.user_id == user_id, ProtectedTemplate.application_id == application_id)
    try:
        claimed = db.execute(
            update(ProtectedTemplate)
            .where(*of_user, ProtectedTemplate.template_set_version == expected_active,
                   ProtectedTemplate.template_set_status == STATUS_ACTIVE)
            .values(template_set_status=STATUS_REVOKED, template_set_revoked_at=now)
            .execution_options(synchronize_session=False)
        ).rowcount
        if not claimed:
            db.rollback()
            raise ActiveSetChangedError(active_set_version(db, user_id, application_id))
        db.execute(
            update(ProtectedTemplate)
            .where(*of_user, ProtectedTemplate.template_set_version == expected_active,
                   ProtectedTemplate.template_status != STATUS_REVOKED)
            .values(template_status=STATUS_REVOKED, is_active=False, revoked_time=now, revoked_reason=reason)
            .execution_options(synchronize_session=False)
        )
        if promote_version is None:
            promoted = db.execute(
                select(func.min(ProtectedTemplate.template_set_version)).where(
                    *of_user, ProtectedTemplate.template_set_status == STATUS_STANDBY)
            ).scalar_one()
            if promoted is None:
                db.rollback()
                raise TemplatePoolExhaustedError("Template set pool exhausted. Re-enrollment required.")
        else:
            promoted = promote_version
            is_standby = db.execute(
                select(func.count()).select_from(ProtectedTemplate).where(
                    *of_user, ProtectedTemplate.template_set_version == promoted,
                    ProtectedTemplate.template_set_status == STATUS_STANDBY)
            ).scalar_one()
            if not is_standby:
                db.rollback()
                raise TemplateNotFoundError(f"No STANDBY template set with version {promoted} for user_id={user_id!r}.")
        expected_rows = db.execute(
            select(func.count()).select_from(ProtectedTemplate).where(
                *of_user, ProtectedTemplate.template_set_version == promoted)
        ).scalar_one()
        moved = db.execute(
            update(ProtectedTemplate)
            .where(*of_user, ProtectedTemplate.template_set_version == promoted,
                   ProtectedTemplate.template_set_status == STATUS_STANDBY)
            .values(template_set_status=STATUS_ACTIVE, template_set_activated_at=now)
            .execution_options(synchronize_session=False)
        ).rowcount
        if moved != expected_rows:
            db.rollback()
            raise ConcurrentEnrollmentError(
                f"Template set {promoted} of user_id={user_id!r} changed during rotation; nothing rotated.")
        db.execute(
            update(ProtectedTemplate)
            .where(*of_user, ProtectedTemplate.template_set_version == promoted,
                   ProtectedTemplate.template_status != STATUS_REVOKED)
            .values(template_status=STATUS_ACTIVE, is_active=True, activation_time=now)
            .execution_options(synchronize_session=False)
        )
        _commit_or_conflict(db, user_id, application_id)
    finally:
        db.expire_all()  # ORM objects loaded earlier in this session must not keep the pre-rotation state
    return expected_active, promoted
