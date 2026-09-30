"""HTTP: POST /enroll/hand, /enroll/hand/check, /verify/hand; audit research metadata; template management with hand.
Synthetic landmark sequences (tests/hand_signals.py); face/voice use the deterministic stubs of tests/test_flexible_auth.py."""

from __future__ import annotations

import json

import pytest

from tests.hand_signals import Person, no_hand_capture, still_hand_capture
from tests.test_flexible_auth import APPLICATION_ID, FACE, OTHER_FACE, VOICE, _audit, _db, _enroll, client  # noqa: F401

ALICE, MALLORY = Person(1), Person(2)


@pytest.fixture(autouse=True)
def _fresh_hand_service():
    from backend.services.hand_service import get_hand_service

    get_hand_service.cache_clear()
    yield
    get_hand_service.cache_clear()


def _file(payload, name="g.json", content_type="application/json"):
    return (name, payload if isinstance(payload, (str, bytes)) else json.dumps(payload), content_type)


def enroll_hand(client, user="u", person=ALICE, seeds=range(3), captures=None):
    captures = captures if captures is not None else [person.capture(s) for s in seeds]
    return client.post("/enroll/hand", data={"user_id": user, "application_id": APPLICATION_ID},
                       files=[("attempts", _file(c)) for c in captures])


def verify_hand(client, payload, user="u", **kw):
    return client.post("/verify/hand", data={"user_id": user, "application_id": APPLICATION_ID},
                       files={"gesture": _file(payload, **kw)})


# ----------------------------------------------------------------------------- enrollment


def test_enroll_hand_with_three_z_samples(client):
    response = enroll_hand(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["modality"] == "hand" and body["templates_created"] == 4 and body["active_template_set_version"] == 1
    hand = body["hand_gesture"]
    assert hand["gesture_type"] == "z" and hand["attempts"] == 3 and hand["template_format"] == "hand_gesture_v2"
    assert "protected_template" not in response.text
    status = client.get(f"/user/u/enrollment-status", params={"application_id": APPLICATION_ID}).json()
    assert status["modalities"]["hand"] is True and status["statuses"]["hand"] == "REGISTERED"


def test_enroll_hand_rejects_a_bad_attempt_with_its_number_and_stores_nothing(client):
    captures = [ALICE.capture(s) for s in range(3)]
    captures[1] = still_hand_capture()
    response = enroll_hand(client, captures=captures)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "HAND_CAPTURE_REJECTED" and body["verdict"] == "INVALID_TRAJECTORY" and body["attempt"] == 2
    assert body["capture_error"] is False
    assert client.get("/user/u/enrollment-status", params={"application_id": APPLICATION_ID}).json()["modalities"]["hand"] is False


@pytest.mark.parametrize("count", [2, 5])
def test_enroll_hand_needs_exactly_three_samples(client, count):
    response = enroll_hand(client, seeds=range(count))
    assert response.status_code == 422 and "exactly 3" in response.json()["detail"]


def test_malformed_attempt_and_wrong_content_type(client):
    captures = [ALICE.capture(s) for s in range(3)]
    response = client.post("/enroll/hand", data={"user_id": "u"},
                           files=[("attempts", _file(c)) for c in captures[:2]] + [("attempts", _file("{not json"))])
    assert response.status_code == 422 and response.json()["verdict"] == "MALFORMED_CAPTURE" and response.json()["attempt"] == 3
    wrong = client.post("/enroll/hand", data={"user_id": "u"},
                        files=[("attempts", _file(c, name="v.mp4", content_type="video/mp4")) for c in captures])
    assert wrong.status_code == 415  # raw video is never accepted


def test_generic_enroll_route_points_to_enroll_hand(client):
    response = client.post("/enroll", data={"user_id": "u", "modality": "hand"}, files={"image": _file(ALICE.capture(1))})
    assert response.status_code == 422 and "/enroll/hand" in response.json()["detail"]


def test_check_endpoint_gives_a_per_attempt_verdict_and_stores_nothing(client):
    ok = client.post("/enroll/hand/check", data={"attempt": "2"}, files={"gesture": _file(ALICE.capture(1))}).json()
    assert ok["passed"] and ok["verdict"] == "OK" and ok["attempt"] == 2 and ok["metrics"]["frames_detected"] > 0
    bad = client.post("/enroll/hand/check", files={"gesture": _file(no_hand_capture())}).json()
    assert not bad["passed"] and bad["verdict"] == "NO_HAND_DETECTED"
    assert client.get("/user/u/enrollment-status", params={"application_id": APPLICATION_ID}).json()["modalities"]["hand"] is False


# ----------------------------------------------------------------------------- verification


def test_verify_hand_genuine(client):
    enroll_hand(client)
    body = verify_hand(client, ALICE.capture(99)).json()
    assert body["authentication_state"] == "ACCESS_GRANTED" and body["matched_modalities"] == ["hand"]
    hand = body["hand_gesture"]
    assert hand["status"] == "VERIFIED" and hand["decision"] == "MATCH" and hand["frames_detected"] > 0
    assert hand["dtw_distance"] <= hand["threshold"] and hand["threshold_source"] == "DEVELOPMENT_DEFAULT"  # DEBUG_SCORES on
    assert len(hand["dtw_distances"]) == 3 and hand["aggregation"] == "median"   # vs sample 1, 2, 3
    assert hand["z_validity"] == "PASS" and hand["trajectory_points"] == 64 and hand["tracking_fps"] >= 10


def test_verify_hand_impostor(client):
    enroll_hand(client)
    body = verify_hand(client, MALLORY.capture(99)).json()
    assert body["authentication_state"] == "ACCESS_DENIED"
    assert body["hand_gesture"]["status"] == "VERIFICATION_MISMATCH" and body["hand_gesture"]["decision"] == "GESTURE_MISMATCH"
    assert body["attempt_assessment"]["modality_statuses"] == {"hand": "VERIFICATION_MISMATCH"}


@pytest.mark.parametrize("payload, status, decision", [
    (no_hand_capture(), "CAPTURE_ERROR", "NO_HAND_DETECTED"),
    (still_hand_capture(), "QUALITY_INSUFFICIENT", "INVALID_TRAJECTORY"),
    ("{broken", "CAPTURE_ERROR", "MALFORMED_CAPTURE"),
])
def test_invalid_captures_are_capture_failures_not_mismatches(client, payload, status, decision):
    enroll_hand(client)
    body = verify_hand(client, payload).json()
    assert body["authentication_state"] == "ACCESS_DENIED"
    assert body["hand_gesture"]["status"] == status and body["hand_gesture"]["decision"] == decision
    assert body["attempt_assessment"]["status"] == "RETRY_REQUESTED"
    assert body["attempt_assessment"]["mismatch_attempt"] in (None, 0)  # never counted as a mismatch


def test_verify_hand_without_enrollment_is_enrollment_required(client):
    response = verify_hand(client, ALICE.capture(1), user="nobody")
    assert response.status_code == 409 and response.json()["missing_modalities"] == ["hand"]


def test_production_response_hides_the_distance(client, monkeypatch):
    enroll_hand(client)
    monkeypatch.setenv("DEBUG_SCORES", "false")
    from backend.config import get_settings

    get_settings.cache_clear()
    body = verify_hand(client, ALICE.capture(99)).json()
    assert body["hand_gesture"]["decision"] == "MATCH"
    assert not {"dtw_distance", "dtw_distances", "threshold"} & set(body["hand_gesture"])
    assert "results" not in body and "score" not in body


def test_audit_records_research_metadata_but_no_landmarks(client):
    enroll_hand(client)
    verify_hand(client, ALICE.capture(99))
    verify_hand(client, no_hand_capture())
    from backend.database.models import AuditLog

    rows = _db().query(AuditLog).filter(AuditLog.user_id == "u").order_by(AuditLog.timestamp).all()
    matched, failed = rows[-2].modality_diagnostics["hand"], rows[-1].modality_diagnostics["hand"]
    assert matched["decision"] == "MATCH" and matched["dtw_distance"] <= matched["threshold"]
    assert matched["frames_detected"] > 0 and matched["duration_s"] > 0 and "dtw_matching" in matched["timing_ms"]
    assert failed["capture_failure"] == "NO_HAND_DETECTED"
    assert rows[-2].hand_similarity is not None
    serialized = json.dumps([r.modality_diagnostics for r in rows])
    assert "landmarks" not in serialized and "features" not in serialized


# ----------------------------------------------------------------------------- template management


def test_revoke_and_generate_with_hand_authorization(client):
    _enroll(client, "u", "face", FACE)
    enroll_hand(client)
    auth = {"face_image": ("f.png", FACE, "image/png"), "hand_gesture": _file(ALICE.capture(77))}
    missing = client.post("/revoke-template", data={"user_id": "u", "application_id": APPLICATION_ID},
                          files={"face_image": ("f.png", FACE, "image/png")})
    assert missing.status_code == 403  # the hand capture is required: it is in the ACTIVE set
    impostor = client.post("/revoke-template", data={"user_id": "u", "application_id": APPLICATION_ID},
                           files={"face_image": ("f.png", FACE, "image/png"), "hand_gesture": _file(MALLORY.capture(1))})
    assert impostor.status_code == 403
    revoked = client.post("/revoke-template", data={"user_id": "u", "application_id": APPLICATION_ID}, files=auth)
    assert revoked.status_code == 200 and revoked.json()["new_active_template_set_version"] == 2
    for _ in range(2):
        client.post("/revoke-template", data={"user_id": "u", "application_id": APPLICATION_ID}, files=auth)
    generated = client.post(f"/templates/u/generate", data={"application_id": APPLICATION_ID}, files=auth)
    assert generated.status_code == 200, generated.text
    sets = {s["template_set_version"]: s for s in client.get("/templates/u", params={"application_id": APPLICATION_ID}).json()["sets"]}
    assert sets[5]["modalities"] == ["face", "hand"] and sets[5]["status"] == "STANDBY"
    client.post("/revoke-template", data={"user_id": "u", "application_id": APPLICATION_ID}, files=auth)
    body = verify_hand(client, ALICE.capture(99)).json()  # the re-keyed set 5 still recognizes the user
    assert body["authentication_state"] == "ACCESS_GRANTED" and body["active_template_set"] == 5
