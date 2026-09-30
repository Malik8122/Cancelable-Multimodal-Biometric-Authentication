"""Edit Profile, the voice capture-quality gate, and two-sentence voice enrollment / verification.

Real HTTP layer, services, BioHash, database and failure handling; deterministic stub embedders
(tests/test_flexible_auth.py). The gate itself runs on the real measurements (preprocessing/voice.py). SOFTWARE TEST -
not a biometric trial; the measured effect on real speech is evaluation/results/voice_capture_pipeline.csv.
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
    _audit,
    _db,
    _enroll,
    client,
)

A = "voice-user"
BUILDING = "national_data_center"
SR = 16000


def _wav(samples: np.ndarray, sr: int = SR) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
    return buffer.getvalue()


def speech_like(frequency: float, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """A tone with a syllable-rate (4 Hz) loudness envelope - level variation like speech, unlike a pure tone."""
    t = np.arange(int(seconds * SR)) / SR
    envelope = 0.1 + 0.9 * (0.5 + 0.5 * np.cos(2 * np.pi * 4 * t))
    return 0.3 * envelope * np.sin(2 * np.pi * frequency * t) + 1e-4 * np.random.default_rng(seed).standard_normal(len(t))


# The stub embedder keys identity on the tone frequency: one "speaker" = one frequency; sentences differ in length/noise.
S1, S2 = _wav(speech_like(220, seed=1)), _wav(speech_like(220, seconds=3.4, seed=2))
OTHER_S1, OTHER_S2 = _wav(speech_like(880, seed=3)), _wav(speech_like(880, seconds=3.4, seed=4))
SILENT = _wav(np.zeros(3 * SR))
SHORT = _wav(speech_like(220, seconds=1.0))
NOISE_ONLY = _wav(0.2 * np.random.default_rng(9).standard_normal(3 * SR))
CLIPPED = _wav(np.clip(speech_like(220) * 8, -1, 1))


@pytest.fixture
def rotation_on(monkeypatch):
    monkeypatch.setenv("TEMPLATE_ROTATION_ON_FAILED_AUTH", "true")


def _enroll_voice(client, user=A, first=S1, second=S2):
    return client.post("/enroll", data={"user_id": user, "modality": "voice", "application_id": APPLICATION_ID},
                       files={"image": ("s1.wav", first, "audio/wav"), "confirm_image": ("s2.wav", second, "audio/wav")})


def _verify(client, user=A, first=S1, face=None, second=None):
    """Voice authentication: one sentence (voice_audio) or two (voice_audio + voice_audio_2, combined into one embedding)."""
    files = {"voice_audio": ("v1.wav", first, "audio/wav")}
    if second is not None:
        files["voice_audio_2"] = ("v2.wav", second, "audio/wav")
    if face is not None:
        files["face_image"] = ("f.png", face, "image/png")
    return client.post("/authenticate/fusion", data={"user_id": user, "building_id": BUILDING, "application_id": APPLICATION_ID},
                       files=files)


def _snapshot(user):
    from backend.database import crud

    return sorted((r.template_set_version, r.modality, r.template_set_status, r.key_version, bytes(r.protected_template))
                  for r in crud.get_set_rows(_db(), user, APPLICATION_ID))


# ----------------------------------------------------------------------------- Edit Profile


def _named_user(client, name="Sanya Malik"):
    user = client.post("/users", json={"display_name": name}).json()["user_id"]
    _enroll(client, user, "face", FACE)
    assert _enroll_voice(client, user).status_code == 200
    return user


def test_editing_the_display_name_never_touches_a_biometric_template(client):
    user = _named_user(client)
    before = _snapshot(user)
    response = client.post(f"/user/{user}/display-name", json={"display_name": "Sanya M."})
    assert response.status_code == 200, response.text
    assert response.json()["display_name"] == "Sanya M."
    assert _snapshot(user) == before  # same templates, key versions and ACTIVE / STANDBY statuses
    status = client.get(f"/user/{user}/enrollment-status", params={"application_id": APPLICATION_ID}).json()
    assert status["display_name"] == "Sanya M." and status["profile_editing_enabled"] is True
    assert _verify(client, user).json()["status"] == "ACCESS_GRANTED"  # the key binding (user_id) is unchanged


def test_profile_edit_does_not_revoke_the_active_template_set(client):
    user = _named_user(client)
    active_before = [row for row in _snapshot(user) if row[2] == "ACTIVE"]
    client.post(f"/user/{user}/display-name", json={"display_name": "Someone Else"})
    assert [row for row in _snapshot(user) if row[2] == "ACTIVE"] == active_before
    assert not any(row[2] == "REVOKED" for row in _snapshot(user))


def test_profile_editing_is_refused_in_production_but_naming_an_unnamed_user_still_works(client, monkeypatch):
    from backend.config import get_settings

    user = _named_user(client)
    _enroll(client, "legacy-unnamed", "face", FACE)  # enrolled without a name (created implicitly)
    monkeypatch.setenv("ENV", "production")
    get_settings.cache_clear()
    before = _snapshot(user)
    assert client.post(f"/user/{user}/display-name", json={"display_name": "Changed"}).status_code == 403
    assert _snapshot(user) == before
    assert client.post("/user/legacy-unnamed/display-name", json={"display_name": "First Name"}).status_code == 200


# ----------------------------------------------------------------------------- the capture-quality gate


GATE_CASES = {"speech": (S1, "OK"), "silent": (SILENT, "NO_SPEECH"), "short": (SHORT, "TOO_LITTLE_SPEECH"),
              "noise": (NOISE_ONLY, "TOO_NOISY"), "clipped": (CLIPPED, "CLIPPED")}


@pytest.mark.parametrize("case", list(GATE_CASES))
def test_the_gate_separates_usable_recordings_from_capture_problems(case):
    sample, verdict = GATE_CASES[case]
    from backend.config import Settings
    from backend.services import voice_quality
    from preprocessing.voice import load_wav_bytes

    report = voice_quality.assess(load_wav_bytes(sample), Settings(master_secret="t"))
    assert report.verdict == verdict, report.metrics
    assert report.passed == (verdict == "OK") and report.message
    assert all(v is None or isinstance(v, float) for v in report.metrics.values())  # scalars only, never audio


def test_empty_audio_is_invalid():
    from backend.config import Settings
    from backend.services import voice_quality

    assert voice_quality.assess((np.zeros(0, np.float32), SR), Settings(master_secret="t")).verdict == "INVALID_AUDIO"


def test_check_quality_endpoint_reports_each_sentence_and_stores_nothing(client):
    good = client.post("/voice/check-quality", data={"sentence": "1"}, files={"audio": ("s.wav", S1, "audio/wav")}).json()
    bad = client.post("/voice/check-quality", data={"sentence": "2"}, files={"audio": ("s.wav", NOISE_ONLY, "audio/wav")}).json()
    assert (good["passed"], good["verdict"], good["sentence"]) == (True, "OK", 1)
    assert (bad["passed"], bad["verdict"], bad["sentence"]) == (False, "TOO_NOISY", 2)
    assert "closer to the microphone" in bad["message"]
    from backend.database.models import AuditLog, ProtectedTemplate

    db = _db()
    assert db.query(ProtectedTemplate).count() == 0 and db.query(AuditLog).count() == 0


# ----------------------------------------------------------------------------- two-sentence enrollment


def test_a_poor_sentence_at_enrollment_names_the_sentence_and_stores_nothing(client):
    response = _enroll_voice(client, first=S1, second=SHORT)
    assert response.status_code == 422
    body = response.json()
    assert (body["status"], body["sentence"], body["verdict"]) == ("VOICE_CAPTURE_QUALITY_FAILURE", 2, "TOO_LITTLE_SPEECH")
    assert _snapshot(A) == []


def test_voice_enrollment_templates_the_centroid_of_both_sentences(client):
    from backend.config import get_settings
    from backend.services import get_service_for_modality
    from backend.services.base_service import build_entry
    from embeddings.centroid import centroid_embedding
    from preprocessing.voice import load_wav_bytes

    assert _enroll_voice(client).status_code == 200
    service = get_service_for_modality("voice")
    centroid = centroid_embedding([service.embed(load_wav_bytes(S1)), service.embed(load_wav_bytes(S2))])
    assert centroid.shape == (192,) and abs(float(np.linalg.norm(centroid)) - 1) < 1e-5  # still the 192-D ECAPA space
    active = next(r for r in _snapshot(A) if r[2] == "ACTIVE")
    expected = build_entry(get_settings(), centroid, user_id=A, application_id=APPLICATION_ID, modality="voice", key_version=active[3])
    assert active[4] == bytes(expected.protected_template)


def test_both_paths_run_the_same_gate_and_the_same_embedding(client, monkeypatch):
    from backend.services import voice_quality

    calls = []
    original = voice_quality.require
    monkeypatch.setattr(voice_quality, "require", lambda raw, settings, sentence=None: calls.append(sentence) or original(raw, settings, sentence))
    _enroll_voice(client)
    enrolled, calls[:] = list(calls), []
    _verify(client)
    # Enrollment gates its two sentences; authentication gates exactly one recording (no sentence number).
    assert enrolled == [1, 2] and calls == [None]


# ----------------------------------------------------------------------------- one-sentence verification
# Protocol: voice ENROLLMENT records two sentences (image + confirm_image); voice AUTHENTICATION records ONE.


def test_one_sentence_verification_succeeds_for_the_enrolled_speaker(rotation_on, client):
    assert _enroll_voice(client).status_code == 200
    before = _snapshot(A)
    body = _verify(client).json()
    assert body["status"] == "ACCESS_GRANTED" and "attempt_assessment" not in body
    assert _snapshot(A) == before


def test_either_enrolled_sentence_alone_verifies_the_speaker(client):
    # The model is text-independent: one recording of either prompt is a complete authentication sample.
    assert _enroll_voice(client).status_code == 200
    assert _verify(client, first=S1).json()["status"] == "ACCESS_GRANTED"
    assert _verify(client, first=S2).json()["status"] == "ACCESS_GRANTED"


def test_two_voice_sentences_verify_the_speaker(client):
    # The frontend protocol: two sentences at authentication, combined like enrollment (2/2, docs/VOICE_MODEL.md).
    assert _enroll_voice(client).status_code == 200
    response = _verify(client, first=S1, second=S2)
    assert response.status_code == 200 and response.json()["status"] == "ACCESS_GRANTED"


def test_a_poor_second_sentence_is_a_quality_failure_naming_the_sentence(client):
    assert _enroll_voice(client).status_code == 200
    body = _verify(client, first=S1, second=CLIPPED).json()
    assert body["status"] != "ACCESS_GRANTED"
    assert body["attempt_assessment"]["modality_statuses"] == {"voice": "QUALITY_INSUFFICIENT"}
    assert "sentence 2" in body["attempt_assessment"]["modality_hints"]["voice"].lower()


def test_a_lone_second_sentence_is_refused(client):
    response = client.post("/authenticate/fusion", data={"user_id": A, "application_id": APPLICATION_ID},
                           files={"voice_audio_2": ("v2.wav", S2, "audio/wav")})
    assert response.status_code == 422


def test_the_authentication_service_refuses_more_than_one_sample():
    from backend.services.authentication import _evaluate_modality

    with pytest.raises(ValueError, match="at most two samples"):
        _evaluate_modality(None, None, None, "voice", ["s1", "s2", "s3"], None, A, APPLICATION_ID, False)
    with pytest.raises(ValueError, match="exactly one sample"):
        _evaluate_modality(None, None, None, "face", ["f1", "f2"], None, A, APPLICATION_ID, False)


def test_voice_enrollment_still_uses_two_sentences(client):
    one = client.post("/enroll", data={"user_id": A, "modality": "voice", "application_id": APPLICATION_ID},
                      files={"image": ("s1.wav", S1, "audio/wav")})
    both = _enroll_voice(client, user="VOICE-TWO")
    assert both.status_code == 200 and both.json()["recording_quality"] in ("EXCELLENT", "GOOD")
    assert one.status_code == 200 and "recording_quality" not in one.json()  # single-recording enrollment: no pair check


def test_a_poor_sentence_at_verification_asks_for_a_new_recording_and_never_rotates(rotation_on, client):
    assert _enroll_voice(client).status_code == 200
    before = _snapshot(A)
    for _ in range(5):  # more than MAX_MODALITY_RETRIES: capture-quality failures never count
        body = _verify(client, first=NOISE_ONLY).json()
        a = body["attempt_assessment"]
        assert body["status"] == "ACCESS_DENIED" and "template_rotation" not in body
        assert a["status"] == "RETRY_REQUESTED" and a["modality_statuses"] == {"voice": "QUALITY_INSUFFICIENT"}
        assert a["modality_messages"]["voice"] == "Voice capture was unsuccessful. Please try again."
        hint = a["modality_hints"]["voice"].lower()
        assert hint.startswith("too much background noise") and "sentence 2" not in hint
        assert a["security_response"] == "No template rotation."
    assert _snapshot(A) == before
    assert _audit(client, A)[0]["modality_statuses"] == {"voice": "QUALITY_INSUFFICIENT"}


def test_a_silent_sentence_is_a_capture_error_not_a_mismatch(rotation_on, client):
    assert _enroll_voice(client).status_code == 200
    a = _verify(client, first=SILENT).json()["attempt_assessment"]
    assert a["modality_statuses"] == {"voice": "CAPTURE_ERROR"} and a["status"] == "RETRY_REQUESTED"


def test_a_single_high_quality_voice_mismatch_is_retried_not_rotated(rotation_on, client):
    assert _enroll_voice(client).status_code == 200
    before = _snapshot(A)
    a = _verify(client, first=OTHER_S1).json()["attempt_assessment"]
    assert a["status"] == "RETRY_REQUESTED" and a["modality_statuses"] == {"voice": "VERIFICATION_MISMATCH"}
    assert _snapshot(A) == before


def test_repeated_high_quality_one_sentence_mismatches_still_escalate(rotation_on, client):
    assert _enroll_voice(client).status_code == 200
    statuses = [_verify(client, first=OTHER_S1).json()["attempt_assessment"]["status"] for _ in range(3)]
    assert statuses == ["RETRY_REQUESTED", "RETRY_REQUESTED", "SUSPICIOUS_ATTEMPT"]


def test_fusion_with_one_voice_sentence_keeps_all_required(client):
    _enroll(client, A, "face", FACE)
    assert _enroll_voice(client).status_code == 200
    assert _verify(client, face=FACE).json()["status"] == "ACCESS_GRANTED"
    body = _verify(client, face=OTHER_FACE).json()
    assert body["status"] == "ACCESS_DENIED" and body["attempt_assessment"]["modality_statuses"] == {
        "face": "VERIFICATION_MISMATCH", "voice": "VERIFIED"}
