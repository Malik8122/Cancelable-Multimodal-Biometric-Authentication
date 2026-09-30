"""The single authentication path shared by /authenticate, /verify/* and /authenticate/fusion.

The USER chooses which enrolled modalities to present; buildings only label the session. For every request:

1. Compare the submitted modalities with the user's enrollment profile. A submitted modality that is not enrolled ->
   `ENROLLMENT_REQUIRED` (HTTP 409): it is not authenticated, nothing is decoded or evaluated, and the attempt is audited
   as its own state (it is not a failed authentication).
2. Otherwise authenticate exactly the submitted modalities, each against the ACTIVE template set.
3. Fuse across exactly those modalities with the requested `fusion.config.FusionPolicy` (ALL_REQUIRED by default), write
   the audit row (which keeps the per-modality values) and return ONE decision.

Fusion formula (equal weights, m = submitted + enrolled modalities; s_m and t_m are on the higher-is-better
estimated-cosine scale of `backend/services/modality_metrics.py` - voice's Euclidean distance d is converted with
s = 1 - d**2 / 2 first, never averaged as a distance):

    fusion_similarity = sum(s_m) / |M|
    fusion_distance   = 1 - fusion_similarity           (computed after fusion only)
    fusion_threshold  = mean(t_m)                       (t_m = per-modality threshold used)

`authenticated` is decided by the policy (`ALL_REQUIRED`: every submitted modality must pass its own threshold), not by
comparing fusion_similarity with fusion_threshold - except under WEIGHTED.

4. After a denial, `failure_policy` decides the response: RETRY_REQUESTED (capture error, insufficient capture quality,
   or an isolated mismatch - the templates never change) or SUSPICIOUS_ATTEMPT (repeated high-quality mismatches), and
   only the latter invokes the security response (`rotate_after_failed_authentication`). A modality whose sample cannot
   be processed is a CAPTURE_ERROR: the attempt is denied (it never reaches fusion) and audited, not a bare 422.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from backend.config import Settings
from backend.database import crud
from backend.database.models import STATUS_ACTIVE
from backend.database.schema import (
    AttemptAssessment,
    AuthenticationDecision,
    EnrollmentRequiredResponse,
    HandGestureOutcome,
    ModalityAuthenticationResult,
    TemplateRotation,
)
from backend.security_validation import SecurityValidationError
from backend.services import enrollment, failure_policy, get_service_for_modality, voice_quality
from backend.services.base_service import AuthenticationResult
from backend.states import (
    ACCESS_DENIED,
    ACCESS_GRANTED,
    CAPTURE_COOLDOWN,
    CAPTURE_ERROR,
    ENROLLMENT_REQUIRED,
    QUALITY_INSUFFICIENT,
    SUSPICIOUS_ATTEMPT,
    VERIFICATION_MISMATCH,
    VERIFIED,
)
from backend.utils import decode_biometric_sample, record_authentication_audit
from fusion.config import DEFAULT_FUSION_POLICY, FusionPolicy
from fusion.diagnostics import build_fusion_diagnostics
from fusion.policy import evaluate_fusion_policy
from preprocessing.hand_gesture import MALFORMED_CAPTURE, HandCaptureRejected

logger = logging.getLogger("backend.services.authentication")


@dataclass
class AuthenticationOutcome:
    """Internal result. Holds the per-modality values that must not reach production clients."""

    user_id: str
    policy: FusionPolicy
    #: ACCESS_GRANTED / ACCESS_DENIED after an evaluation; ENROLLMENT_REQUIRED when a submitted modality is not enrolled.
    state: str = ACCESS_DENIED
    building_id: str | None = None
    submitted_modalities: list[str] = field(default_factory=list)
    enrolled_modalities: list[str] = field(default_factory=list)
    #: Submitted modalities that are not enrolled (ENROLLMENT_REQUIRED only).
    missing_modalities: list[str] = field(default_factory=list)
    per_modality: dict[str, AuthenticationResult] = field(default_factory=dict)
    fusion_similarity: float = 0.0
    fusion_threshold: float = 0.0
    authenticated: bool = False
    matched_modalities: list[str] = field(default_factory=list)
    failed_modalities: list[str] = field(default_factory=list)
    latency_ms: int = 0
    template_set_version: int = 0
    #: The user's human-readable name (crud.display_name_for) - only ever sent to the client on ACCESS_GRANTED.
    display_name: str | None = None
    #: Security response to a SUSPICIOUS attempt (see rotate_after_failed_authentication); None when not invoked.
    rotation: TemplateRotation | None = None
    #: {modality: VERIFIED / VERIFICATION_MISMATCH / QUALITY_INSUFFICIENT / CAPTURE_ERROR} for every submitted modality.
    modality_statuses: dict[str, str] = field(default_factory=dict)
    #: A short, non-biometric reason for a CAPTURE_ERROR / QUALITY_INSUFFICIENT modality (e.g. "the image was blurry").
    modality_details: dict[str, str] = field(default_factory=dict)
    #: ACCESS_GRANTED / RETRY_REQUESTED / SUSPICIOUS_ATTEMPT (after an evaluation).
    attempt_status: str | None = None
    assessment: failure_policy.Assessment | None = None
    #: CAPTURE_COOLDOWN: seconds until the next attempt is accepted.
    cooldown_seconds: int = 0
    #: Per-modality research metadata (hand gesture: capture stats, DTW distance, timings, failure code). Audited.
    modality_diagnostics: dict[str, dict] = field(default_factory=dict)


def release_modality_cache(modality: str) -> None:
    """Drop the process-wide cached service/model for one modality.

    The per-modality services are `@lru_cache`d singletons; on a 512MB Render instance a multi-modality request that
    loaded both of them would exceed memory. Only call this after that modality's `authenticate(...)` has
    returned (a concurrent request still holding the old service keeps it alive via normal reference counting).
    """
    if modality == "face":
        from backend.services.face_service import get_face_service

        get_face_service.cache_clear()
    elif modality == "voice":
        from backend.services.voice_service import get_voice_service

        get_voice_service.cache_clear()


def rotate_after_failed_authentication(
    db: Session, settings: Settings, user_id: str, application_id: str, active_before: int,
    reason: str = "automatic rotation after failed authentication",
) -> TemplateRotation:
    """Retire the ACTIVE template set and promote the next STANDBY set, for EVERY enrolled modality at once.

    Invoked only for a SUSPICIOUS_ATTEMPT (backend/services/failure_policy.py), never for a single failed login.

    Uses the one rotation primitive (`crud.rotate_active_set_if_current`, shared with /revoke-template and activation). The promoted set was generated at enrollment from
    the LEGITIMATE enrollment centroid under its own HKDF key versions, so the user keeps authenticating with their own
    biometrics; nothing from the failed attempt is templated or stored. With no STANDBY set left nothing changes (the
    ACTIVE set keeps working) and POOL_EXHAUSTED is reported - the user must re-enroll to get new sets.
    """
    modalities = sorted({r.modality for r in crud.get_active_set_rows(db, user_id, application_id)})
    if not settings.template_rotation_on_failed_auth:
        return TemplateRotation(status="DISABLED", triggered=False, active_set_before=active_before, active_set_after=active_before,
                                modalities=modalities, remaining_standby_sets=len(crud.standby_set_versions(db, user_id, application_id)))
    try:
        # Compare-and-swap: rotate only the set THIS attempt was evaluated against (crud.rotate_active_set_if_current).
        revoked, promoted = crud.rotate_active_set_if_current(
            db, user_id, application_id, active_before, reason=reason
        )
    except crud.ActiveSetChangedError as changed:
        # A concurrent failed attempt already retired this set: one rotation per set, never a second one.
        current = changed.current_active or active_before
        return TemplateRotation(status="ALREADY_ROTATED", triggered=False, active_set_before=active_before, active_set_after=current,
                                modalities=modalities, remaining_standby_sets=len(crud.standby_set_versions(db, user_id, application_id)),
                                reason="authentication failed; a concurrent failed attempt already rotated this template set")
    except crud.TemplatePoolExhaustedError:
        return TemplateRotation(status="POOL_EXHAUSTED", triggered=False, active_set_before=active_before, active_set_after=active_before,
                                modalities=modalities, remaining_standby_sets=0, reason="authentication failed; no standby template set left")
    except (crud.ConcurrentEnrollmentError, OperationalError) as error:
        # e.g. SQLite "database is locked" after the busy timeout: nothing was committed; the DENIED decision stands.
        db.rollback()
        logger.error("Template rotation for user_id=%s did not complete (%s); active set unchanged.", user_id, type(error).__name__)
        current = crud.active_set_version(db, user_id, application_id) or active_before
        return TemplateRotation(status="ROTATION_FAILED", triggered=False, active_set_before=active_before, active_set_after=current,
                                modalities=modalities, remaining_standby_sets=len(crud.standby_set_versions(db, user_id, application_id)),
                                reason="authentication failed; template rotation could not be completed")
    return TemplateRotation(status="ROTATED", triggered=True, active_set_before=revoked, active_set_after=promoted, modalities=modalities,
                            remaining_standby_sets=len(crud.standby_set_versions(db, user_id, application_id)))


def authenticate_samples(
    db: Session,
    settings: Settings,
    *,
    user_id: str,
    application_id: str,
    building_id: str | None,
    uploads: dict[str, UploadFile | list[UploadFile]],
    policy: FusionPolicy = DEFAULT_FUSION_POLICY,
    started_at: float | None = None,
) -> AuthenticationOutcome:
    """Authenticate exactly the modalities in `uploads` (the user's choice) and fuse them."""
    started_at = started_at if started_at is not None else time.perf_counter()
    submitted = sorted(uploads)
    outcome = AuthenticationOutcome(user_id=user_id, policy=policy, building_id=building_id, submitted_modalities=submitted)

    # 1. Enrollment first, before any biometric is touched.
    profile = enrollment.get_user_enrollment_status(db, user_id, application_id)
    outcome.enrolled_modalities = profile.enrolled
    outcome.missing_modalities = enrollment.missing_modalities(profile, submitted)
    if outcome.missing_modalities:
        outcome.state = ENROLLMENT_REQUIRED
        outcome.latency_ms = round((time.perf_counter() - started_at) * 1000)
        record_authentication_audit(
            db,
            user_id=user_id,
            building_id=building_id,
            modality_list=[],  # nothing was evaluated
            similarity_scores={},
            thresholds_used={},
            authenticated=False,
            started_at=started_at,
            template_versions={},
            key_versions={},
            fusion_policy=policy.value,
            authentication_state=ENROLLMENT_REQUIRED,
            submitted_modalities=submitted,
            enrolled_modalities=outcome.enrolled_modalities,
            authenticated_modalities=[],
        )
        return outcome

    # 1b. Too many consecutive unusable captures: a short cooldown (rate limit). Nothing is decoded or evaluated, no
    # template changes - a capture problem is never a security response (failure_policy.capture_cooldown_remaining).
    outcome.cooldown_seconds = failure_policy.capture_cooldown_remaining(db, settings, user_id=user_id)
    if outcome.cooldown_seconds:
        outcome.state, outcome.attempt_status = ACCESS_DENIED, CAPTURE_COOLDOWN
        outcome.latency_ms = round((time.perf_counter() - started_at) * 1000)
        record_authentication_audit(
            db, user_id=user_id, building_id=building_id, modality_list=[], similarity_scores={}, thresholds_used={},
            authenticated=False, started_at=started_at, template_versions={}, key_versions={}, fusion_policy=policy.value,
            authentication_state=ACCESS_DENIED, submitted_modalities=submitted, enrolled_modalities=outcome.enrolled_modalities,
            authenticated_modalities=[], template_rotation_triggered=False, template_rotation_status="NOT_TRIGGERED",
            attempt_status=CAPTURE_COOLDOWN,
        )
        logger.info("CAPTURE_COOLDOWN user_id=%s retry_after=%ss (no evaluation, no template change)", user_id, outcome.cooldown_seconds)
        return outcome

    # 2. Authenticate exactly the submitted modalities.
    release_models = len(uploads) > 1
    # ONE consistent snapshot of the user's template sets for the whole attempt: every modality is evaluated against
    # the same ACTIVE set even if a concurrent request rotates the pool meanwhile (that rotation is then simply
    # ordered after this attempt; see rotate_after_failed_authentication / crud.rotate_active_set_if_current).
    pool = crud.get_set_rows(db, user_id, application_id)
    for modality in submitted:
        detail = _evaluate_modality(db, settings, outcome, modality, uploads[modality], pool, user_id, application_id,
                                    release_models)
        if detail:
            outcome.modality_details[modality] = detail

    # Never mix templates from different sets: every modality was compared with its row of the ACTIVE set.
    set_versions = {r.template_set_version for r in outcome.per_modality.values() if r.template_set_version}
    if len(set_versions) > 1:
        raise SecurityValidationError(
            f"Active template set mismatch for user_id={user_id!r}: modalities matched sets {sorted(set_versions)}."
        )
    active_in_pool = next((r.template_set_version for r in pool if r.is_active), 0)
    outcome.template_set_version = next(iter(set_versions), active_in_pool)

    # 3. Fusion over exactly these modalities. A modality that was never compared (CAPTURE_ERROR, or a voice recording
    # rejected by the quality gate) has no score: the attempt fails closed (denied without fusion) instead of fusing an
    # incomplete result - no policy can grant on a sample that was never evaluated.
    scores = {m: r.score for m, r in outcome.per_modality.items()}
    thresholds = {m: r.threshold for m, r in outcome.per_modality.items()}
    outcome.fusion_threshold = sum(thresholds.values()) / len(thresholds) if thresholds else 0.0
    if any(m not in outcome.per_modality for m in submitted):
        outcome.fusion_similarity = 0.0  # nothing was fused
        outcome.authenticated = False
        outcome.matched_modalities = sorted(m for m, r in outcome.per_modality.items() if r.authenticated)
        outcome.failed_modalities = sorted(set(submitted) - set(outcome.matched_modalities))
    else:
        decision = evaluate_fusion_policy(
            scores=scores,
            individually_authenticated={m: r.authenticated for m, r in outcome.per_modality.items()},
            policy=policy,
            fusion_threshold=outcome.fusion_threshold,
        )
        outcome.fusion_similarity = decision.fused_score
        outcome.authenticated = decision.authenticated
        outcome.matched_modalities = decision.matched_modalities
        outcome.failed_modalities = decision.failed_modalities
    outcome.state = ACCESS_GRANTED if outcome.authenticated else ACCESS_DENIED
    if outcome.authenticated:
        outcome.attempt_status = ACCESS_GRANTED
        outcome.display_name = crud.display_name_for(db, user_id)
    else:
        # 4. Retry or escalate - decided only by the backend (never by the client). Only an escalation reaches the
        # existing security response; a retry leaves every template set exactly as it was.
        outcome.assessment = failure_policy.assess_denied_attempt(
            db, settings, user_id=user_id, active_set=outcome.template_set_version, statuses=outcome.modality_statuses
        )
        outcome.attempt_status = outcome.assessment.status
        if outcome.assessment.status == SUSPICIOUS_ATTEMPT:
            reason = ("suspicious authentication attempt: repeated high-quality "
                      f"{', '.join(outcome.assessment.escalated_modalities)} verification mismatch")
            outcome.rotation = rotate_after_failed_authentication(
                db, settings, user_id, application_id, outcome.template_set_version, reason=reason
            )
            if outcome.rotation.status == "ROTATED":
                outcome.rotation.reason = reason
            logger.info(
                "SUSPICIOUS_ATTEMPT user_id=%s escalated=%s streaks=%s rotation=%s active_set %d -> %d modalities=%s",
                user_id, outcome.assessment.escalated_modalities, outcome.assessment.mismatch_streaks,
                outcome.rotation.status, outcome.rotation.active_set_before, outcome.rotation.active_set_after,
                outcome.rotation.modalities,
            )
        else:
            logger.info(
                "RETRY_REQUESTED user_id=%s statuses=%s streaks=%s retries_remaining=%d (no template change)",
                user_id, outcome.modality_statuses, outcome.assessment.mismatch_streaks,
                outcome.assessment.retries_remaining,
            )
    outcome.latency_ms = round((time.perf_counter() - started_at) * 1000)

    logger.info(
        "Auth user_id=%s modalities=%s policy=%s authenticated=%s",
        user_id, submitted, policy.value, outcome.authenticated,
    )
    if settings.debug_scores:
        # DEBUG ONLY: the fusion stage. Nothing here reaches a production response.
        logger.info(
            "FUSION-DEBUG building=%s submitted=%s enrolled=%s entering_fusion=%s rejected_before_fusion=%s "
            "matched=%s failed=%s fusion_similarity=%.4f fusion_distance=%.4f fusion_threshold=%.2f "
            "set_version=%s state=%s",
            outcome.building_id, submitted, outcome.enrolled_modalities, sorted(scores),
            sorted(set(submitted) - set(scores)), outcome.matched_modalities, outcome.failed_modalities,
            outcome.fusion_similarity, 1.0 - outcome.fusion_similarity, outcome.fusion_threshold,
            outcome.template_set_version, outcome.state,
        )
    record_authentication_audit(
        db,
        user_id=user_id,
        building_id=building_id,
        modality_list=submitted,
        similarity_scores=scores,
        thresholds_used=thresholds,
        authenticated=outcome.authenticated,
        started_at=started_at,
        template_versions={m: r.template_set_version for m, r in outcome.per_modality.items()},
        key_versions={m: r.key_version for m, r in outcome.per_modality.items()},
        # `fusion_score` was historically only recorded for multi-modality requests; `fusion_similarity` is always recorded.
        fusion_score=outcome.fusion_similarity if len(uploads) > 1 else None,
        fusion_policy=policy.value if len(uploads) > 1 else None,
        fusion_similarity=outcome.fusion_similarity,
        template_set_version=outcome.template_set_version or None,
        template_set_status=STATUS_ACTIVE if outcome.template_set_version else None,
        authentication_state=outcome.state,
        submitted_modalities=submitted,
        enrolled_modalities=outcome.enrolled_modalities,
        authenticated_modalities=outcome.matched_modalities,
        failed_modalities=outcome.failed_modalities,
        active_set_before=(outcome.rotation.active_set_before if outcome.rotation else outcome.template_set_version) or None,
        active_set_after=outcome.rotation.active_set_after if outcome.rotation else (outcome.template_set_version or None),
        template_rotation_triggered=outcome.rotation.triggered if outcome.rotation else False,
        template_rotation_status=(
            outcome.rotation.status if outcome.rotation else "NOT_APPLICABLE" if outcome.authenticated else "NOT_TRIGGERED"
        ),
        modality_statuses=outcome.modality_statuses,
        attempt_status=outcome.attempt_status,
        modality_diagnostics=outcome.modality_diagnostics or None,
    )
    return outcome


def _decode(modality: str, upload: UploadFile, settings: Settings):
    """Decode one upload; None for an undecodable sample (a CAPTURE_ERROR). Type/size errors stay request errors."""
    try:
        return decode_biometric_sample(modality, upload, settings)
    except HTTPException as error:
        if error.status_code != status.HTTP_400_BAD_REQUEST:
            raise  # unsupported type / oversized upload: a request error, not a biometric capture
        return None


def _evaluate_modality(db, settings, outcome, modality, upload, pool, user_id, application_id, release_models) -> str | None:
    """Evaluate one submitted modality and record its status in `outcome`. Returns a short reason for a capture/quality
    failure (never anything biometric), else None.

    CAPTURE_ERROR: the sample could not be decoded or processed (e.g. no face detected, undecodable audio, a silent
    recording). QUALITY_INSUFFICIENT - face: the capture did not match AND fails the existing enrollment quality gates
    (FacePipeline.check_capture); a capture that matched is VERIFIED whatever its quality. Voice: a recording failed
    the capture-quality gate (backend/services/voice_quality.py) and was never compared. Anything else that did not
    match is a VERIFICATION_MISMATCH.

    One sample per modality at authentication - one face capture, one hand gesture execution - except voice, which
    takes one or two sentences: each is gated and embedded, and two unit embeddings are combined (L2-normalized mean,
    the enrollment rule) into ONE embedding compared once with the ACTIVE template.
    """
    uploads = upload if isinstance(upload, list) else [upload]
    if len(uploads) > (2 if modality == "voice" else 1):
        raise ValueError(f"{modality} authentication takes {'at most two samples' if modality == 'voice' else 'exactly one sample'}, got {len(uploads)}")
    raws = [_decode(modality, u, settings) for u in uploads]
    if any(raw is None for raw in raws):
        outcome.modality_statuses[modality] = CAPTURE_ERROR
        if modality == "hand":
            outcome.modality_diagnostics[modality] = {"capture_failure": MALFORMED_CAPTURE}
        return "the uploaded sample could not be decoded"
    service = get_service_for_modality(modality)
    try:
        try:
            if modality == "hand":
                try:
                    result = service.authenticate(db, raws[0], user_id=user_id, application_id=application_id, pool=pool)
                except HandCaptureRejected as rejected:
                    # A capture problem, never a biometric mismatch: no comparison happened.
                    outcome.modality_statuses[modality] = CAPTURE_ERROR if rejected.is_capture_error else QUALITY_INSUFFICIENT
                    outcome.modality_diagnostics[modality] = {
                        "capture_failure": rejected.code, "gesture_type": raws[0].gesture_type,
                        "frames_total": raws[0].frame_count, "frames_detected": raws[0].detected_frames,
                    }
                    return rejected.reason
                if result.diagnostics:
                    outcome.modality_diagnostics[modality] = result.diagnostics
            elif modality == "voice":
                embedding = service.embed_recordings(raws)
                try:
                    result = service.authenticate_embedding(db, embedding, user_id, application_id, pool=pool)
                finally:
                    del embedding
            else:
                result = service.authenticate(db, raws[0], user_id=user_id, application_id=application_id, pool=pool)
        except voice_quality.VoiceCaptureRejected as rejected:
            capture = rejected.report.verdict in (voice_quality.INVALID_AUDIO, voice_quality.NO_SPEECH)
            outcome.modality_statuses[modality] = CAPTURE_ERROR if capture else QUALITY_INSUFFICIENT
            reason = voice_quality.REASONS[rejected.report.verdict]
            return f"{'sentence ' + str(rejected.sentence) + ': ' if rejected.sentence else ''}{reason}"
        except ValueError as error:  # e.g. MTCNN "No face detected in the provided image."
            outcome.modality_statuses[modality] = CAPTURE_ERROR
            return str(error).rstrip(".").lower() or "the sample could not be processed"
        outcome.per_modality[modality] = result
        if result.authenticated:
            outcome.modality_statuses[modality] = VERIFIED
            return None
        if modality == "face":
            verdict = service.check_capture(raws[0])
            if verdict != "VALID":
                outcome.modality_statuses[modality] = QUALITY_INSUFFICIENT
                return failure_policy.face_quality_reason(verdict)
        outcome.modality_statuses[modality] = VERIFICATION_MISMATCH
        return None
    finally:
        del service
        if release_models:
            release_modality_cache(modality)


def enrollment_required_response(outcome: AuthenticationOutcome) -> JSONResponse:
    """HTTP 409 body for ENROLLMENT_REQUIRED: which submitted modalities are not enrolled."""
    names = " and ".join(m.capitalize() for m in outcome.missing_modalities)
    detail = (
        f"{names} {'is' if len(outcome.missing_modalities) == 1 else 'are'} not registered. "
        f"Complete {names.lower()} enrollment before authenticating with {'it' if len(outcome.missing_modalities) == 1 else 'them'}."
    )
    body = EnrollmentRequiredResponse(
        status=ENROLLMENT_REQUIRED,
        authentication_state=ENROLLMENT_REQUIRED,
        detail=detail,
        user_id=outcome.user_id,
        building_id=outcome.building_id,
        submitted_modalities=outcome.submitted_modalities,
        enrolled_modalities=outcome.enrolled_modalities,
        missing_modalities=outcome.missing_modalities,
    )
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=body.model_dump(exclude_none=True))


def to_public_response(outcome: AuthenticationOutcome, settings: Settings) -> AuthenticationDecision:
    """Build the public response: one decision, one fusion similarity, one active template set.

    The user's display name is added only when access was granted. Per-modality similarities / thresholds /
    distances are added only under DEBUG_SCORES.
    """
    key_versions = {m: r.key_version for m, r in outcome.per_modality.items()}
    response = AuthenticationDecision(
        user_id=outcome.user_id,
        authentication_state=outcome.state,
        status=outcome.state,
        authenticated=outcome.authenticated,
        fusion_similarity=outcome.fusion_similarity,
        fusion_distance=1.0 - outcome.fusion_similarity,
        fusion_threshold=outcome.fusion_threshold,
        fusion_policy=outcome.policy.value,
        matched_modalities=outcome.matched_modalities,
        modalities_used=sorted(outcome.per_modality),
        active_template_set=outcome.template_set_version,
        template_set_version=outcome.template_set_version,
        key_version=max(key_versions.values(), default=0),
        authentication_time_ms=outcome.latency_ms,
        building_id=outcome.building_id,
        display_name=outcome.display_name if outcome.authenticated else None,
        template_rotation=outcome.rotation if outcome.rotation and outcome.rotation.status != "DISABLED" else None,
        attempt_assessment=_public_assessment(outcome, settings),
    )
    if settings.debug_scores:
        response.failed_modalities = outcome.failed_modalities
        response.fused_score = outcome.fusion_similarity
        response.template_version = outcome.template_set_version
        response.template_versions = {m: r.template_set_version for m, r in outcome.per_modality.items()}
        response.key_versions = key_versions
        response.results = {
            m: ModalityAuthenticationResult(
                score=r.score,
                threshold=r.threshold,
                authenticated=r.authenticated,
                distance=r.distance,
                metric=r.metric,
                metric_value=r.metric_value,
                metric_threshold=r.metric_threshold,
                metric_higher_is_better=r.metric_higher_is_better,
                metric_uncertainty=r.metric_uncertainty,
                hamming_similarity=r.hamming_similarity,
                hamming_distance_bits=r.hamming_distance_bits,
                template_bits=r.template_bits,
                template_version=r.template_set_version,
                key_version=r.key_version,
                mock_embedder=r.mock_embedder,
                template_status=r.template_status,
            )
            for m, r in outcome.per_modality.items()
        }
        if len(outcome.per_modality) == 1:
            (modality, only), = outcome.per_modality.items()
            response.modality = modality
            response.score = only.score
            response.threshold = only.threshold
            response.distance = only.distance
        response.fusion_diagnostics = build_fusion_diagnostics(
            submitted_modalities=outcome.submitted_modalities,
            scores={m: r.score for m, r in outcome.per_modality.items()},
            thresholds={m: r.threshold for m, r in outcome.per_modality.items()},
            individually_authenticated={m: r.authenticated for m, r in outcome.per_modality.items()},
            fused_score=outcome.fusion_similarity,
            fusion_threshold=outcome.fusion_threshold,
            policy=outcome.policy.value,
            access_granted=outcome.authenticated,
        )
    hand = outcome.modality_diagnostics.get("hand")
    if hand is not None:
        response.hand_gesture = _public_hand(outcome, hand, settings)
    return response


def _public_hand(outcome: AuthenticationOutcome, diagnostics: dict, settings: Settings) -> HandGestureOutcome:
    """Hand gesture status for the client: decision / capture failure code, capture stats and timings. The DTW distance
    and threshold are per-modality scores - added only under DEBUG_SCORES, like every other modality's score."""
    status_ = outcome.modality_statuses.get("hand", "")
    result = HandGestureOutcome(
        status=status_,
        decision=diagnostics.get("capture_failure") or diagnostics.get("decision") or status_,
        gesture_type=diagnostics.get("gesture_type"),
        frames_total=diagnostics.get("frames_total"),
        frames_detected=diagnostics.get("frames_detected"),
        duration_s=diagnostics.get("duration_s"),
        timing_ms=diagnostics.get("timing_ms"),
        reason=outcome.modality_details.get("hand"),
    )
    if settings.debug_scores:
        result.dtw_distance = diagnostics.get("dtw_distance")
        result.dtw_distances = diagnostics.get("dtw_distances")
        result.threshold = diagnostics.get("threshold")
        result.threshold_source = diagnostics.get("threshold_source")
        result.tracking_fps = diagnostics.get("tracking_fps")
        result.motion_duration_s = diagnostics.get("motion_duration_s")
        result.idle_trimmed_s = diagnostics.get("idle_trimmed_s")
        result.trajectory_points = diagnostics.get("resampled_length")
        result.z_validity = diagnostics.get("z_validity")
        result.aggregation = diagnostics.get("aggregation")
        result.dtw_best_rotation_deg = diagnostics.get("dtw_best_rotation_deg")
        result.feature_group_shares = diagnostics.get("feature_group_shares")
        result.normalized_duration = diagnostics.get("normalized_duration")
        result.finger_shares = diagnostics.get("finger_shares")
        result.finger_extension_live = diagnostics.get("finger_extension_live")
        result.finger_extension_enrolled = diagnostics.get("finger_extension_enrolled")
        result.shadow_distances = diagnostics.get("shadow_distances")
        result.path_length_palms = diagnostics.get("path_length_palms")
        result.mean_speed_palms_per_s = diagnostics.get("mean_speed_palms_per_s")
        if "segment_time_top" in diagnostics:
            result.segment_time_fractions = [diagnostics[f"segment_time_{k}"] for k in ("top", "diagonal", "bottom")]
    return result


def _public_assessment(outcome: AuthenticationOutcome, settings: Settings) -> AttemptAssessment | None:
    """Retry guidance / escalation for a DENIED attempt (status names and fixed wording only)."""
    if outcome.authenticated or outcome.assessment is None:
        return None
    wording = failure_policy.describe(
        outcome.assessment, outcome.modality_statuses, outcome.modality_details,
        outcome.rotation.status if outcome.rotation else None, settings,
    )
    return AttemptAssessment(
        status=outcome.assessment.status,
        modality_statuses=outcome.modality_statuses,
        retry_modalities=outcome.assessment.retry_modalities,
        retries_remaining=outcome.assessment.retries_remaining,
        mismatch_attempt=outcome.assessment.mismatch_attempt,
        mismatch_attempt_limit=outcome.assessment.mismatch_attempt_limit,
        capture_failures=outcome.assessment.capture_failures,
        capture_failure_limit=outcome.assessment.capture_failure_limit,
        **wording,
    )


def respond(outcome: AuthenticationOutcome, settings: Settings) -> AuthenticationDecision | JSONResponse:
    """The HTTP result of an outcome: 409 ENROLLMENT_REQUIRED, or the single fusion decision (200)."""
    if outcome.state == ENROLLMENT_REQUIRED:
        return enrollment_required_response(outcome)
    if outcome.attempt_status == CAPTURE_COOLDOWN:
        seconds = outcome.cooldown_seconds
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(seconds)},
            content={"status": CAPTURE_COOLDOWN, "authentication_state": ACCESS_DENIED, "retry_after_seconds": seconds,
                     "detail": f"Too many unsuccessful captures in a row. Please wait {seconds} seconds, check the camera "
                               f"and microphone, and try again. No template was changed."},
        )
    return to_public_response(outcome, settings)
