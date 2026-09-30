"""SQLAlchemy ORM models: users and their protected biometric templates.

`ProtectedTemplate.protected_template` is the *only* biometric-derived data
ever persisted, and it holds exactly what
`template_protection.biohash.generate_template` returns, packed to bytes
(see `template_protection.utils.pack_bits`) - never a raw image, never a raw
`BaseEmbedder.extract_embedding()` output. `output_bits` is stored purely as
metadata needed to unpack those bytes back into a bit array (see
`template_protection.utils.unpack_bits`), not as anything that reveals
embedding content.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


#: Lifecycle of one protected template (see docs/MULTI_TEMPLATE_ARCHITECTURE.md).
#: `is_active` is kept as a mirror of `template_status == ACTIVE` so the
#: partial unique index below (exactly one ACTIVE per context) keeps working
#: and pre-existing readers of `is_active` stay correct.
STATUS_ACTIVE = "ACTIVE"
STATUS_STANDBY = "STANDBY"
STATUS_REVOKED = "REVOKED"
TEMPLATE_STATUSES = (STATUS_ACTIVE, STATUS_STANDBY, STATUS_REVOKED)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    #: Caller-supplied external identifier (e.g. "U001" from the spec's
    #: examples), not an auto-incrementing surrogate key - this is what
    #: `user_id` means everywhere in the API and in template_protection's
    #: key derivation.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    #: The human-readable DISPLAY NAME (backend/display_names.py) - a label only, never an identifier. NULL for users
    #: enrolled before names existed (they are shown as "User <short id>").
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    templates: Mapped[list["ProtectedTemplate"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ProtectedTemplate(Base):
    __tablename__ = "protected_templates"

    template_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    modality: Mapped[str] = mapped_column(String, index=True)
    application_id: Mapped[str] = mapped_column(String, index=True)

    #: template_protection.biohash.TEMPLATE_FORMAT_VERSION at generation time -
    #: bumped only if the transform algorithm itself changes, independent of
    #: key_version below.
    template_version: Mapped[int] = mapped_column(Integer)

    #: Which HKDF key_version produced this template (see
    #: template_protection/hkdf_keys.py) - bumped on every
    #: `/revoke-template` call for this (user, modality, application).
    key_version: Mapped[int] = mapped_column(Integer)

    #: Bit-length of `protected_template` before packing, needed to unpack it
    #: back into a bit array (template_protection.utils.unpack_bits).
    output_bits: Mapped[int] = mapped_column(Integer)

    #: The protected template itself, packed to bytes
    #: (template_protection.utils.pack_bits). Never a raw image or embedding.
    #: For the hand-gesture modality: the keyed, orthonormally transformed
    #: enrollment sequences (template_protection/sequence_transform.py).
    protected_template: Mapped[bytes] = mapped_column(LargeBinary)

    #: Representation of `protected_template`: NULL for the 256-bit BioHash (face, voice - the original format), or
    #: e.g. "hand_gesture_v1" (preprocessing/hand_gesture.py::HAND_TEMPLATE_FORMAT). A template is only ever compared
    #: with a candidate built by the same representation.
    template_format: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Hand gesture only: which gesture was enrolled (e.g. "z").
    gesture_type: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Non-biometric enrollment metadata (hand gesture: attempts, per-attempt frame counts / durations, feature
    #: dimension, DTW settings, intra-enrollment distance statistics). Never landmarks, features or templates.
    template_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    #: Revoking a template deactivates the old row rather than deleting it,
    #: so there's an audit trail of past key rotations; only one row per
    #: (user_id, modality, application_id) may be active at a time. That is
    #: primarily enforced here at the schema level (the partial unique index
    #: below) - backend/database/crud.py::save_template's read-then-insert
    #: sequence isn't itself atomic under concurrent requests (FastAPI runs
    #: sync routes in a threadpool), so without this index two racing
    #: enroll/revoke calls for the same (user, modality, application) could
    #: each insert their own "active" row. With it, the second insert raises
    #: `IntegrityError` instead of silently leaving two active rows behind -
    #: turning a silent correctness bug into a loud, retryable one.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    #: ACTIVE / STANDBY / REVOKED. Only the ACTIVE row is ever compared
    #: against during authentication.
    template_status: Mapped[str] = mapped_column(String, default=STATUS_ACTIVE, index=True)
    activation_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Templates created by one enrollment/replenish call share a group id.
    template_group_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    #: Legacy mirror of `template_set_version` (kept so older readers work).
    template_index: Mapped[int] = mapped_column(Integer, default=1)

    # --- Template Set (docs/MULTI_TEMPLATE_ARCHITECTURE.md) -----------------
    # A template SET is one complete multimodal credential: every row with the
    # same (user_id, application_id, template_set_version) is one modality's
    # template inside it. Activation / revocation happen per SET, so the
    # set-level columns below are identical on every row of a set, and the
    # row-level `template_status` / `is_active` above follow them (a row is
    # ACTIVE exactly when its set is ACTIVE - except a row that was
    # individually replaced by re-enrolling one modality, which is REVOKED
    # while its set stays ACTIVE).
    #: 1-based, never reused, per (user_id, application_id). This is the
    #: "template set version" the API and UI show.
    template_set_version: Mapped[int] = mapped_column(Integer, default=1, index=True)
    #: ACTIVE / STANDBY / REVOKED - exactly one ACTIVE set per (user, application).
    template_set_status: Mapped[str] = mapped_column(String, default=STATUS_ACTIVE, index=True)
    template_set_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    template_set_activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    template_set_revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="templates")

    __table_args__ = (
        # `text(...)`, not the `is_active` column expression, because at
        # class-body evaluation time `is_active` above is still a bare
        # `MappedColumn` construct, not yet a compilable SQL expression -
        # raw SQL text sidesteps that ordering issue entirely.
        Index(
            "ux_one_active_template_per_context",
            "user_id",
            "modality",
            "application_id",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active = true"),
        ),
        # A template set holds at most one live template per modality.
        Index(
            "ux_one_live_template_per_set_modality",
            "user_id",
            "application_id",
            "template_set_version",
            "modality",
            unique=True,
            sqlite_where=text("template_status <> 'REVOKED'"),
            postgresql_where=text("template_status <> 'REVOKED'"),
        ),
    )


#: Singleton row id for `MasterSecretFingerprint` below - the model's primary key is always this
#: fixed value, which is what makes "at most one row" true by construction: a second row would
#: have to reuse this same primary key and either collide or simply be the same row.
MASTER_SECRET_FINGERPRINT_ID = 1


class MasterSecretFingerprint(Base):
    """At most one row (see `MASTER_SECRET_FINGERPRINT_ID`): a one-way fingerprint of the
    MASTER_SECRET that protected this database's existing biometric templates.

    Never the secret itself - `fingerprint` is `backend/secret_fingerprint.py::fingerprint_master_secret`'s
    output, a one-way HKDF-SHA256 derivation over a fixed public context
    ("master_secret_fingerprint/v1"), deliberately unrelated to and never reachable from
    `template_protection/hkdf_keys.py::derive_key`'s per-(user, modality, application, key_version)
    template key material - this row exists to answer "is the currently configured MASTER_SECRET
    the same one used before?", never "what is the secret?".

    `backend/key_continuity.py` compares a fresh fingerprint of the currently configured
    MASTER_SECRET against this row at startup, refusing to start rather than silently authenticate
    every enrolled user against a wrong key - see docs/AUTHENTICATION_RELIABILITY_REPORT.md for the
    real incident this exists to prevent from recurring.
    """

    __tablename__ = "master_secret_fingerprint"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: 32 bytes, opaque, one-way. Never printed/logged (backend/key_continuity.py only ever
    #: compares it with `hmac.compare_digest`) and never exposed through any API response.
    fingerprint: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuditLog(Base):
    """Server-side authentication attempt history - metadata only.

    Written once per `/authenticate`, `/verify/{modality}`, or
    `/authenticate/fusion` call (see `backend/database/audit.py::record_attempt`,
    called from those three route handlers). Never stores anything
    biometric-derived: no raw image/audio, no embedding, no protected
    template - only identifiers, scores (already-computed floats, not
    reconstructable back into a template or embedding), thresholds, and
    outcomes. This is what makes "server-side audit log" compatible with the
    same "cancelable, non-reconstructable" privacy posture as
    `ProtectedTemplate` above.
    """

    __tablename__ = "audit_logs"

    audit_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    user_id: Mapped[str] = mapped_column(String, index=True)

    #: Optional caller-supplied label (e.g. the frontend's building id) -
    #: purely descriptive, the backend has no concept of "buildings" itself.
    building_id: Mapped[str | None] = mapped_column(String, nullable=True)

    #: Modalities involved in this attempt, e.g. ["face", "voice"].
    modality_list: Mapped[list[str]] = mapped_column(JSON)

    #: {modality: score} - the same per-modality Hamming similarity scores
    #: already returned in the HTTP response, not re-derivable into a
    #: template or embedding.
    similarity_scores: Mapped[dict[str, float]] = mapped_column(JSON)

    #: {modality: threshold} - which calibrated (or fallback) threshold each
    #: modality was actually compared against, per `backend/threshold_loader.py`.
    thresholds_used: Mapped[dict[str, float]] = mapped_column(JSON)

    fusion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Internal per-modality similarities (also inside `similarity_scores`) as
    #: first-class columns for metrics/testing; never sent to the production
    #: frontend. `fusion_similarity` mirrors `fusion_score`.
    face_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    voice_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Hand gesture fusion-scale score (1 - DTW distance / (2 * threshold); see backend/services/hand_service.py).
    hand_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    fusion_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    fusion_policy: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Which template set matched, and its lifecycle status at the time.
    template_set_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    template_set_status: Mapped[str | None] = mapped_column(String, nullable=True)

    #: ACCESS_GRANTED / ACCESS_DENIED / ENROLLMENT_REQUIRED (backend/states.py). Enrollment-required
    #: attempts are recorded here - and NOT as failed authentications - with nothing biometric evaluated.
    authentication_state: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    #: The modalities the user submitted for this session (evaluated or not), the modalities the user had
    #: enrolled at the time, and the submitted ones that actually verified.
    submitted_modalities: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    enrolled_modalities: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    authenticated_modalities: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    authenticated: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[int] = mapped_column(Integer)

    #: Submitted modalities that did NOT verify, and the security response to a DENIED attempt
    #: (Settings.template_rotation_on_failed_auth): active set version before/after, whether a rotation happened,
    #: and why/why not (ROTATED / POOL_EXHAUSTED / DISABLED / NOT_APPLICABLE). Metadata only.
    failed_modalities: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    active_set_before: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active_set_after: Mapped[int | None] = mapped_column(Integer, nullable=True)
    template_rotation_triggered: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    template_rotation_status: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Failure handling (backend/services/failure_policy.py): {modality: VERIFIED / VERIFICATION_MISMATCH /
    #: QUALITY_INSUFFICIENT / CAPTURE_ERROR} and the attempt's security response (ACCESS_GRANTED / RETRY_REQUESTED /
    #: SUSPICIOUS_ATTEMPT). Consecutive VERIFICATION_MISMATCH rows are what an escalation counts. Status names only.
    modality_statuses: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    #: Per-modality research metadata, e.g. {"hand": {"dtw_distance", "threshold", "frames_detected", "duration_s",
    #: "capture_failure", "timing_ms": {...}}}. Numbers and codes only - never landmarks, features or templates.
    modality_diagnostics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempt_status: Mapped[str | None] = mapped_column(String, nullable=True)

    #: {modality: template_version} / {modality: key_version} - lets a
    #: reviewer see key rotation happening across a user's history without
    #: cross-referencing `protected_templates` (which only keeps the
    #: *current* active row per context, not history).
    template_versions: Mapped[dict[str, int]] = mapped_column(JSON)
    key_versions: Mapped[dict[str, int]] = mapped_column(JSON)


class EnrollmentEvent(Base):
    """One enrollment attempt for one modality: ENROLLED, or INCONSISTENT (voice: the two recordings did not match).

    Templates alone cannot say "the last attempt failed" - a rejected enrollment stores nothing - so the outcome is
    kept here. It is what turns a not-enrolled voice into RETRY_REQUIRED instead of NOT_REGISTERED. Metadata only.
    """

    __tablename__ = "enrollment_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String, index=True)
    application_id: Mapped[str] = mapped_column(String, index=True)
    modality: Mapped[str] = mapped_column(String, index=True)
    outcome: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
