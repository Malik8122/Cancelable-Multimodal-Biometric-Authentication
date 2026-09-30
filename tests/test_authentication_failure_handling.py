"""Failure handling: capture error / quality failure / isolated mismatch -> RETRY; repeated high-quality mismatch ->
SUSPICIOUS_ATTEMPT -> the existing template rotation (backend/services/failure_policy.py).

Real HTTP layer, services, BioHash, database, fusion policy and audit log; deterministic stub embedders
(tests/test_flexible_auth.py). Default retry policy (MAX_MODALITY_RETRIES=2) with template rotation ON, so any rotation
that happened would be visible. SOFTWARE TEST - not a biometric trial.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from scipy.io import wavfile

from tests.test_flexible_auth import (  # noqa: F401  (client is a fixture)
    APPLICATION_ID,
    FACE,
    OTHER_FACE,
    OTHER_VOICE,
    VOICE,
    _audit,
    _authenticate,
    _db,
    _enroll_user,
    _png,
    client,
)

A = "user-a"
BUILDING = "national_data_center"


def _silent_wav() -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, 16000, np.zeros(32000, dtype=np.int16))
    return buffer.getvalue()


SILENT_VOICE = _silent_wav()
BROKEN_WAV = b"RIFF-not-really-a-wav"
#: Low-contrast image: the stub face pipeline's quality gate calls it BLURRY (tests/test_flexible_auth.py::_StubPipeline).
BLURRY_FACE = _png(np.random.default_rng(7).integers(120, 136, size=(300, 300, 3), dtype=np.uint8))


@pytest.fixture
def rotation_on(monkeypatch):
    monkeypatch.setenv("TEMPLATE_ROTATION_ON_FAILED_AUTH", "true")


@pytest.fixture
def rotation_spy(monkeypatch):
    """Records every call of the one rotation primitive (the only code path that changes the ACTIVE set)."""
    from backend.database import crud

    calls = []
    original = crud.rotate_active_set_if_current

    def spy(*args, **kwargs):
        calls.append(args[3] if len(args) > 3 else kwargs.get("expected_active"))
        return original(*args, **kwargs)

    monkeypatch.setattr(crud, "rotate_active_set_if_current", spy)
    return calls


def _snapshot(user):
    """{set_version: (status, {modality: (key_version, template bytes)})} - the whole template lifecycle state."""
    from backend.database import crud

    out = {}
    for r in crud.get_set_rows(_db(), user, APPLICATION_ID):
        status, rows = out.setdefault(r.template_set_version, (r.template_set_status, {}))
        rows[r.modality] = (r.key_version, bytes(r.protected_template))
    return out


def _active(user):
    return next(v for v, (status, _) in _snapshot(user).items() if status == "ACTIVE")


def _body(response):
    assert response.status_code == 200, response.text
    return response.json()


# ----------------------------------------------------------------------------- the required cases


def test_1_poor_quality_voice_capture_is_retried_and_never_rotates(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    for _ in range(5):  # more attempts than MAX_MODALITY_RETRIES: capture errors never count towards an escalation
        body = _body(_authenticate(client, A, BUILDING, face=FACE, voice=SILENT_VOICE))
        assert body["status"] == "ACCESS_DENIED" and "template_rotation" not in body
        a = body["attempt_assessment"]
        assert a["status"] == "RETRY_REQUESTED" and a["retry_modalities"] == ["voice"]
        assert a["modality_statuses"] == {"face": "VERIFIED", "voice": "CAPTURE_ERROR"}
        assert a["headline"] == "Voice capture was unsuccessful." and a["security_response"] == "No template rotation."
        assert a["modality_messages"]["voice"] == "Voice capture was unsuccessful. Please try again."
        assert a["mismatch_attempt"] == 0  # a capture problem is not a mismatch attempt
    assert _snapshot(A) == before and rotation_spy == []
    entry = _audit(client, A)[0]
    assert (entry["attempt_status"], entry["template_rotation_status"]) == ("RETRY_REQUESTED", "NOT_TRIGGERED")
    assert entry["modality_statuses"] == {"face": "VERIFIED", "voice": "CAPTURE_ERROR"}


def test_1b_undecodable_audio_is_a_capture_error_not_a_bare_422(rotation_on, client):
    _enroll_user(client, A, voice=VOICE)
    body = _body(_authenticate(client, A, BUILDING, voice=BROKEN_WAV))
    assert body["attempt_assessment"]["modality_statuses"] == {"voice": "CAPTURE_ERROR"}
    assert _audit(client, A)[0]["attempt_status"] == "RETRY_REQUESTED"  # audited, unlike the old 422


def test_2_single_high_quality_voice_mismatch_is_retried_and_does_not_rotate(rotation_on, rotation_spy, client):
    """Focused regression: ONE voice mismatch does NOT rotate the template."""
    _enroll_user(client, A, voice=VOICE)
    before = _snapshot(A)
    body = _body(_authenticate(client, A, BUILDING, voice=OTHER_VOICE))
    assert body["status"] == "ACCESS_DENIED" and "template_rotation" not in body
    a = body["attempt_assessment"]
    assert a["status"] == "RETRY_REQUESTED" and a["modality_statuses"] == {"voice": "VERIFICATION_MISMATCH"}
    assert a["retries_remaining"] == 1  # default MAX_MODALITY_RETRIES=2: this was the first of three
    assert a["headline"] == "Authentication incomplete." and a["reason"] == "Voice verification did not match. Please try again."
    assert (a["mismatch_attempt"], a["mismatch_attempt_limit"]) == (1, 3) and "mismatch 1 of 3" in a["attempt_notice"]
    assert _snapshot(A) == before and _active(A) == 1 and rotation_spy == []


def test_3_face_match_voice_mismatch_retries_voice_without_rotation(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    a = _body(_authenticate(client, A, BUILDING, face=FACE, voice=OTHER_VOICE))["attempt_assessment"]
    assert a["retry_modalities"] == ["voice"]
    assert a["modality_messages"] == {"face": "Face verified.", "voice": "Voice verification did not match. Please try again."}
    assert _snapshot(A) == before and rotation_spy == []


def test_4_voice_match_face_mismatch_retries_face_without_rotation(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    a = _body(_authenticate(client, A, BUILDING, face=OTHER_FACE, voice=VOICE))["attempt_assessment"]
    assert a["retry_modalities"] == ["face"] and a["modality_statuses"]["voice"] == "VERIFIED"
    assert _snapshot(A) == before and rotation_spy == []


def test_5_repeated_high_quality_mismatch_escalates_to_the_existing_security_response(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    first, second, third = (_body(_authenticate(client, A, BUILDING, face=FACE, voice=OTHER_VOICE)) for _ in range(3))
    assert [b["attempt_assessment"]["status"] for b in (first, second)] == ["RETRY_REQUESTED"] * 2
    assert [b["attempt_assessment"]["retries_remaining"] for b in (first, second)] == [1, 0]
    a = third["attempt_assessment"]
    assert third["status"] == "ACCESS_DENIED" and a["status"] == "SUSPICIOUS_ATTEMPT" and a["retry_modalities"] == []
    assert a["headline"] == "Authentication failed — suspicious verification pattern detected."
    assert a["reason"] == "Repeated high-quality voice verification mismatch."
    assert a["security_response"] == "cancellable template rotation triggered."  # shown as "Security response: ..."
    assert (a["mismatch_attempt"], a["mismatch_attempt_limit"]) == (3, 3)
    assert "attacker" not in str(a).lower() and "attack detected" not in str(a).lower()
    rot = third["template_rotation"]
    assert (rot["status"], rot["active_set_before"], rot["active_set_after"]) == ("ROTATED", 1, 2)
    assert rotation_spy == [1]  # exactly one call of the existing compare-and-swap primitive
    assert [e["attempt_status"] for e in _audit(client, A)] == ["SUSPICIOUS_ATTEMPT", "RETRY_REQUESTED", "RETRY_REQUESTED"]


def test_6_successful_multimodal_authentication_is_accepted_without_rotation(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    body = _body(_authenticate(client, A, BUILDING, face=FACE, voice=VOICE))
    assert body["status"] == "ACCESS_GRANTED" and "attempt_assessment" not in body and "template_rotation" not in body
    assert _snapshot(A) == before and rotation_spy == []
    entry = _audit(client, A)[0]
    assert (entry["attempt_status"], entry["template_rotation_status"]) == ("ACCESS_GRANTED", "NOT_APPLICABLE")


def test_7_capture_failure_followed_by_a_successful_retry_is_accepted(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    assert _body(_authenticate(client, A, BUILDING, face=FACE, voice=SILENT_VOICE))["attempt_assessment"]["status"] == "RETRY_REQUESTED"
    body = _body(_authenticate(client, A, BUILDING, face=FACE, voice=VOICE))
    assert body["status"] == "ACCESS_GRANTED" and body["template_set_version"] == 1
    assert _snapshot(A) == before and rotation_spy == []


def test_8_escalation_changes_the_template_version_through_the_existing_rotation(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    for _ in range(3):
        body = _body(_authenticate(client, A, BUILDING, face=OTHER_FACE))
    assert body["attempt_assessment"]["status"] == "SUSPICIOUS_ATTEMPT"
    after = _snapshot(A)
    assert after[1][0] == "REVOKED" and after[2][0] == "ACTIVE" and after[3][0] == after[4][0] == "STANDBY"
    assert set(after[2][1]) == {"face", "voice"}  # every enrolled modality moved together
    assert after[2][1] == before[2][1]  # the promoted set is the one generated at enrollment (same keys, same bytes)
    # The legitimate user keeps authenticating on the new set, and the count restarts on it.
    assert _body(_authenticate(client, A, BUILDING, face=FACE))["template_set_version"] == 2
    assert _body(_authenticate(client, A, BUILDING, face=OTHER_FACE))["attempt_assessment"]["status"] == "RETRY_REQUESTED"


def test_9_a_failed_sample_is_never_stored_as_a_template(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    enrolled = {blob for _, rows in _snapshot(A).values() for _, blob in rows.values()}
    row_count = sum(len(rows) for _, rows in _snapshot(A).values())
    for voice in (SILENT_VOICE, OTHER_VOICE, OTHER_VOICE, OTHER_VOICE):  # retry, retry, retry, escalation
        _authenticate(client, A, BUILDING, face=FACE, voice=voice)
    after = _snapshot(A)
    assert sum(len(rows) for _, rows in after.values()) == row_count  # no row added
    assert {blob for _, rows in after.values() for _, blob in rows.values()} == enrolled  # only enrollment templates


def test_10_ordinary_retries_never_revoke_the_active_template(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    samples = [dict(face=FACE, voice=SILENT_VOICE), dict(face=BLURRY_FACE, voice=VOICE), dict(face=FACE, voice=OTHER_VOICE),
               dict(face=OTHER_FACE, voice=VOICE), dict(face=FACE, voice=BROKEN_WAV)]
    for kwargs in samples:
        assert _body(_authenticate(client, A, BUILDING, **kwargs))["attempt_assessment"]["status"] == "RETRY_REQUESTED"
    assert _snapshot(A) == before  # no status, key version or template byte changed
    assert rotation_spy == []  # the rotation primitive was never called
    assert all(e["template_rotation_triggered"] is False for e in _audit(client, A))


# ----------------------------------------------------------------------------- policy details


def test_face_mismatch_failing_the_quality_gate_is_quality_insufficient_and_never_escalates(rotation_on, rotation_spy, client):
    _enroll_user(client, A, face=FACE)
    for _ in range(5):
        a = _body(_authenticate(client, A, BUILDING, face=BLURRY_FACE))["attempt_assessment"]
        assert a["modality_statuses"] == {"face": "QUALITY_INSUFFICIENT"} and a["status"] == "RETRY_REQUESTED"
    assert "blurry" in a["modality_hints"]["face"] and a["modality_messages"]["face"] == "Face capture was unsuccessful. Please try again."
    assert rotation_spy == [] and _active(A) == 1


def test_a_successful_authentication_resets_the_mismatch_count(rotation_on, rotation_spy, client):
    _enroll_user(client, A, voice=VOICE)
    for _ in range(2):
        _authenticate(client, A, BUILDING, voice=OTHER_VOICE)
    assert _body(_authenticate(client, A, BUILDING, voice=VOICE))["status"] == "ACCESS_GRANTED"
    a = _body(_authenticate(client, A, BUILDING, voice=OTHER_VOICE))["attempt_assessment"]
    assert a["status"] == "RETRY_REQUESTED" and a["retries_remaining"] == 1
    assert rotation_spy == []


def test_capture_errors_between_mismatches_neither_count_nor_reset(rotation_on, client):
    _enroll_user(client, A, voice=VOICE)
    statuses = [_body(_authenticate(client, A, BUILDING, voice=v))["attempt_assessment"]["status"]
                for v in (OTHER_VOICE, SILENT_VOICE, OTHER_VOICE, SILENT_VOICE, OTHER_VOICE)]
    assert statuses == ["RETRY_REQUESTED"] * 4 + ["SUSPICIOUS_ATTEMPT"]


def test_mismatches_are_counted_per_modality(rotation_on, rotation_spy, client):
    """Alternating failures of different modalities: each is an isolated mismatch of that modality."""
    _enroll_user(client, A, face=FACE, voice=VOICE)
    for kwargs in (dict(face=OTHER_FACE, voice=VOICE), dict(face=FACE, voice=OTHER_VOICE)) * 2:
        assert _body(_authenticate(client, A, BUILDING, **kwargs))["attempt_assessment"]["status"] == "RETRY_REQUESTED"
    assert rotation_spy == []


def test_old_mismatches_outside_the_window_no_longer_count(monkeypatch, rotation_on, client):
    import time

    monkeypatch.setenv("SUSPICIOUS_MISMATCH_WINDOW_SECONDS", "1")
    from backend.config import get_settings

    get_settings.cache_clear()
    _enroll_user(client, A, voice=VOICE)
    for _ in range(2):
        _authenticate(client, A, BUILDING, voice=OTHER_VOICE)
    time.sleep(1.3)
    a = _body(_authenticate(client, A, BUILDING, voice=OTHER_VOICE))["attempt_assessment"]
    assert a["status"] == "RETRY_REQUESTED" and a["retries_remaining"] == 1


def test_escalation_with_rotation_disabled_changes_nothing_and_says_so(rotation_spy, client):
    _enroll_user(client, A, voice=VOICE)
    before = _snapshot(A)
    for _ in range(3):
        body = _body(_authenticate(client, A, BUILDING, voice=OTHER_VOICE))
    a = body["attempt_assessment"]
    assert a["status"] == "SUSPICIOUS_ATTEMPT" and "disabled" in a["security_response"].lower()
    assert "template_rotation" not in body and _snapshot(A) == before and rotation_spy == []
    assert _audit(client, A)[0]["template_rotation_status"] == "DISABLED"


def test_the_single_modality_endpoint_uses_the_same_failure_handling(rotation_on, rotation_spy, client):
    _enroll_user(client, A, voice=VOICE)
    response = client.post("/verify/voice", data={"user_id": A, "application_id": APPLICATION_ID},
                           files={"image": ("v.wav", OTHER_VOICE, "audio/wav")})
    assert _body(response)["attempt_assessment"]["status"] == "RETRY_REQUESTED"
    assert rotation_spy == []


def test_the_audit_log_holds_status_names_only(rotation_on, client):
    from backend.states import ATTEMPT_STATUSES, MODALITY_VERIFICATION_STATUSES

    _enroll_user(client, A, face=FACE, voice=VOICE)
    for kwargs in (dict(face=FACE, voice=SILENT_VOICE), dict(face=BLURRY_FACE, voice=OTHER_VOICE), dict(face=FACE, voice=VOICE)):
        _authenticate(client, A, BUILDING, **kwargs)
    for entry in _audit(client, A):
        assert entry["attempt_status"] in ATTEMPT_STATUSES
        assert set(entry["modality_statuses"].values()) <= set(MODALITY_VERIFICATION_STATUSES)
        assert all(isinstance(v, str) and len(v) < 40 for v in entry["modality_statuses"].values())


# ----------------------------------------------------------------------------- retry numbering and capture cooldown


def _reload_settings(monkeypatch, **env):
    from backend.config import get_settings

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def test_retry_numbering_is_attempt_1_and_2_retry_then_attempt_3_rotates(rotation_on, rotation_spy, client):
    _enroll_user(client, A, voice=VOICE)
    before = _snapshot(A)
    seen = []
    for expected in (1, 2):
        a = _body(_authenticate(client, A, BUILDING, voice=OTHER_VOICE))["attempt_assessment"]
        seen.append((a["status"], a["mismatch_attempt"], a["mismatch_attempt_limit"]))
        assert f"mismatch {expected} of 3" in a["attempt_notice"]
        assert _snapshot(A) == before and rotation_spy == []  # nothing changed yet
    body = _body(_authenticate(client, A, BUILDING, voice=OTHER_VOICE))
    a = body["attempt_assessment"]
    seen.append((a["status"], a["mismatch_attempt"], a["mismatch_attempt_limit"]))
    assert seen == [("RETRY_REQUESTED", 1, 3), ("RETRY_REQUESTED", 2, 3), ("SUSPICIOUS_ATTEMPT", 3, 3)]
    assert body["template_rotation"]["status"] == "ROTATED" and rotation_spy == [1]
    assert a["headline"] == "Authentication failed — suspicious verification pattern detected."


def test_each_attempt_is_fused_on_its_own_and_no_verified_factor_is_carried_over(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    first = _body(_authenticate(client, A, BUILDING, face=FACE, voice=OTHER_VOICE))   # face verified, voice not
    second = _body(_authenticate(client, A, BUILDING, face=OTHER_FACE, voice=VOICE))  # voice verified, face not
    assert first["status"] == second["status"] == "ACCESS_DENIED"  # ALL_REQUIRED per attempt, no carry-over


def test_repeated_unusable_captures_pause_attempts_but_never_rotate(rotation_on, rotation_spy, monkeypatch, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _snapshot(A)
    for n in range(1, 6):  # MAX_CONSECUTIVE_CAPTURE_FAILURES = 5
        a = _body(_authenticate(client, A, BUILDING, face=FACE, voice=SILENT_VOICE))["attempt_assessment"]
        assert (a["status"], a["capture_failures"], a["capture_failure_limit"]) == ("RETRY_REQUESTED", n, 5)
    blocked = _authenticate(client, A, BUILDING, face=FACE, voice=VOICE)  # even a genuine attempt waits
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) > 0
    body = blocked.json()
    assert body["status"] == "CAPTURE_COOLDOWN" and "No template was changed" in body["detail"]
    assert "attack" not in body["detail"].lower()
    assert _snapshot(A) == before and rotation_spy == []  # a capture problem is never a security response
    entry = _audit(client, A)[0]
    assert (entry["attempt_status"], entry["template_rotation_triggered"]) == ("CAPTURE_COOLDOWN", False)


def test_the_cooldown_expires_and_a_usable_capture_ends_it(rotation_on, monkeypatch, client):
    import time

    _reload_settings(monkeypatch, CAPTURE_FAILURE_COOLDOWN_SECONDS="1")
    _enroll_user(client, A, voice=VOICE)
    for _ in range(5):
        _authenticate(client, A, BUILDING, voice=SILENT_VOICE)
    assert _authenticate(client, A, BUILDING, voice=VOICE).status_code == 429
    time.sleep(1.3)
    assert _body(_authenticate(client, A, BUILDING, voice=VOICE))["status"] == "ACCESS_GRANTED"
    assert _body(_authenticate(client, A, BUILDING, voice=SILENT_VOICE))["attempt_assessment"]["capture_failures"] == 1


def test_a_usable_mismatching_capture_breaks_the_unusable_capture_run(rotation_on, client):
    _enroll_user(client, A, voice=VOICE)
    for voice in (SILENT_VOICE, SILENT_VOICE, SILENT_VOICE, SILENT_VOICE, OTHER_VOICE, SILENT_VOICE, SILENT_VOICE):
        response = _authenticate(client, A, BUILDING, voice=voice)
        assert response.status_code == 200  # never 5 unusable captures in a row


def test_the_capture_cooldown_can_be_disabled(rotation_on, monkeypatch, client):
    _reload_settings(monkeypatch, MAX_CONSECUTIVE_CAPTURE_FAILURES="0")
    _enroll_user(client, A, voice=VOICE)
    for _ in range(7):
        assert _authenticate(client, A, BUILDING, voice=SILENT_VOICE).status_code == 200
