"""HandGestureService: three-sample Z enrollment, one-sample authentication, template sets, security validation.
Synthetic landmark sequences (tests/hand_signals.py) - software tests, not biometric trials."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.config import get_settings
from backend.database import crud
from backend.security_validation import SecurityValidationError
from backend.services.hand_service import GESTURE_MISMATCH, MATCH, HandEnrollmentIncomplete, HandGestureService, fusion_score
from preprocessing.hand_gesture import HAND_TEMPLATE_FORMAT, INVALID_TRAJECTORY, NO_HAND_DETECTED, HandCaptureRejected, parse_capture
from template_protection.sequence_transform import deserialize
from tests.hand_signals import Person, no_hand_capture, still_hand_capture

APP = "capstone-demo"
ALICE, MALLORY = Person(1), Person(2)


def _cap(payload):
    return parse_capture(json.dumps(payload))


def _enroll(db, service, person=ALICE, user="alice", seeds=range(3)):
    return service.enroll_attempts(db, [_cap(person.capture(s)) for s in seeds], user, APP)


@pytest.fixture
def service():
    return HandGestureService(get_settings())


def test_three_samples_enroll_into_every_template_set(db_session, service):
    rows, summary = _enroll(db_session, service)
    assert len(rows) == 4 and [r.template_status for r in rows] == ["ACTIVE", "STANDBY", "STANDBY", "STANDBY"]
    assert {r.modality for r in rows} == {"hand"} and {r.template_format for r in rows} == {HAND_TEMPLATE_FORMAT}
    assert {r.gesture_type for r in rows} == {"z"} and sorted(r.key_version for r in rows) == [1, 2, 3, 4]
    assert summary["attempts"] == 3 and len(summary["attempt_frames_detected"]) == 3 and summary["medoid_sample"] in (1, 2, 3)
    for row in rows:
        _, sequences = deserialize(row.protected_template)
        assert len(sequences) == 3 and sequences[0].shape == (64, 13)  # all three samples are kept, separately
    assert len({bytes(r.protected_template) for r in rows}) == 4  # every set under its own key
    # metadata is numbers/codes only - never features or landmarks
    assert not any(isinstance(v, list) and v and isinstance(v[0], list) for v in rows[0].template_metadata.values())


def test_a_failed_capture_rejects_the_whole_enrollment_and_stores_nothing(db_session, service):
    captures = [_cap(ALICE.capture(s)) for s in range(3)]
    captures[2] = _cap(no_hand_capture())
    with pytest.raises(HandCaptureRejected) as error:
        service.enroll_attempts(db_session, captures, "alice", APP)
    assert error.value.code == NO_HAND_DETECTED and error.value.attempt == 3
    assert crud.get_set_rows(db_session, "alice", APP) == []


@pytest.mark.parametrize("count", [0, 2, 4, 5])
def test_enrollment_needs_exactly_three_samples(db_session, service, count):
    with pytest.raises(HandEnrollmentIncomplete):
        service.enroll_attempts(db_session, [_cap(ALICE.capture(s)) for s in range(count)], "alice", APP)
    assert crud.get_set_rows(db_session, "alice", APP) == []


def test_genuine_gesture_matches(db_session, service):
    _enroll(db_session, service)
    result = service.authenticate(db_session, _cap(ALICE.capture(99)), "alice", APP)
    assert result.authenticated and result.metric == "dtw_distance" and not result.metric_higher_is_better
    assert result.metric_value <= result.metric_threshold and result.diagnostics["decision"] == MATCH
    assert len(result.diagnostics["dtw_distances"]) == 3 and result.diagnostics["z_validity"] == "PASS"
    assert result.score == pytest.approx(fusion_score(result.metric_value, result.metric_threshold))
    assert result.threshold == 0.5 and result.template_set_version == 1 and result.key_version == 1
    timing = result.diagnostics["timing_ms"]
    assert timing["feature_extraction"] > 0 and timing["dtw_matching"] > 0 and timing["mediapipe_ms_mean"] == 11.5


def test_impostor_gesture_is_a_mismatch(db_session, service):
    _enroll(db_session, service)
    result = service.authenticate(db_session, _cap(MALLORY.capture(99)), "alice", APP)
    assert not result.authenticated and result.diagnostics["decision"] == GESTURE_MISMATCH
    assert result.metric_value > result.metric_threshold and result.score < 0.5


def test_missing_template_is_not_enrolled_not_an_error(db_session, service):
    result = service.authenticate(db_session, _cap(ALICE.capture(1)), "nobody", APP)
    assert not result.authenticated and result.score == 0.0 and result.diagnostics is None


def test_invalid_capture_raises_a_capture_failure(db_session, service):
    _enroll(db_session, service)
    with pytest.raises(HandCaptureRejected) as error:
        service.authenticate(db_session, _cap(still_hand_capture()), "alice", APP)
    assert error.value.code == INVALID_TRAJECTORY


def test_a_revoked_set_is_replaced_and_the_genuine_user_still_matches(db_session, service):
    _enroll(db_session, service)
    crud.revoke_active_set_and_promote(db_session, "alice", APP)
    result = service.authenticate(db_session, _cap(ALICE.capture(99)), "alice", APP)
    assert result.authenticated and result.template_set_version == 2 and result.key_version == 2


def test_rekeyed_entry_gives_identical_distances(db_session, service):
    _enroll(db_session, service)
    pool = crud.get_set_rows(db_session, "alice", APP)
    entry = service.rekey_entry(pool, "alice", APP, key_version=9)
    active = next(r for r in pool if r.is_active)
    assert entry.key_version == 9 and entry.protected_template != active.protected_template
    assert entry.template_format == HAND_TEMPLATE_FORMAT and entry.gesture_type == "z"


def test_a_corrupted_template_fails_closed(db_session, service):
    _enroll(db_session, service)
    active = next(r for r in crud.get_set_rows(db_session, "alice", APP) if r.is_active)
    active.protected_template = b"garbage"
    db_session.commit()
    with pytest.raises(SecurityValidationError):
        service.authenticate(db_session, _cap(ALICE.capture(99)), "alice", APP)


def test_fusion_score_mapping():
    assert fusion_score(0.0, 0.3) == 1.0
    assert fusion_score(0.3, 0.3) == pytest.approx(0.5)
    assert fusion_score(0.6, 0.3) == pytest.approx(0.0)
    assert fusion_score(5.0, 0.3) == -1.0


def test_face_and_voice_templates_are_untouched_by_hand_enrollment(db_session, service):
    from tests.test_flexible_auth import _StubPipeline
    from backend.services.base_service import ModalityService

    face = ModalityService("face", _StubPipeline(512), get_settings())
    face_rows = face.enroll(db_session, np.random.default_rng(1).integers(0, 256, (300, 300, 3), dtype=np.uint8), "alice", APP)
    before = {r.template_id: bytes(r.protected_template) for r in face_rows}
    _enroll(db_session, service)
    rows = crud.get_set_rows(db_session, "alice", APP)
    assert {r.template_id: bytes(r.protected_template) for r in rows if r.modality == "face"} == before
    assert {r.template_set_version for r in rows if r.modality == "hand"} == {1, 2, 3, 4}  # joined the same sets
    assert all(r.template_format is None for r in rows if r.modality == "face")
