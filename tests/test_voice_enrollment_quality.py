"""Voice enrollment consistency check: ECAPA embedding cosine similarity in four quality bands.

    >= 0.85 EXCELLENT (enrolled)   0.75-0.84 GOOD (enrolled)
    0.60-0.74 FAIR -> LOW_QUALITY_WARNING (nothing stored unless the user continues)   < 0.60 POOR -> ENROLLMENT_INCONSISTENT

Deliberately lenient: it only rejects recordings that clearly do not belong together. It changes no authentication
threshold, no fusion logic, no template generation and not the T1-T4 pool.

Boundary tests use embeddings with an exact cosine. API tests use a voice stub that maps a tone's frequency to an angle
(embedding = [cos(f/1000), sin(f/1000), 0...]), so the cosine of two recordings is exactly cos((f2-f1)/1000).
"""

from __future__ import annotations

import functools
import io
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from tests.voice_signals import speech_envelope
from backend.services.recording_quality import (
    EXCELLENT,
    FAIR,
    FAIR_MESSAGE,
    GOOD,
    POOR,
    POOR_MESSAGE,
    classify,
    cosine_similarity,
)

APPLICATION_ID = "capstone-demo"


# ------------------------------------------------------------------ the bands

@pytest.mark.parametrize(
    "similarity, band",
    [
        (1.0, EXCELLENT), (0.98, EXCELLENT), (0.85, EXCELLENT),
        (0.8499, GOOD), (0.80, GOOD), (0.75, GOOD),
        (0.7499, FAIR), (0.66, FAIR), (0.60, FAIR),
        (0.5999, POOR), (0.5, POOR), (0.0, POOR), (-0.3, POOR),
    ],
)
def test_band_boundaries(similarity, band):
    assert classify(similarity) == band


def test_cosine_similarity_is_norm_safe_and_handles_degenerate_input():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([3.0, 0.0])) == pytest.approx(1.0)  # not unit length
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_similarity(np.zeros(4), np.array([1.0, 0, 0, 0])) == 0.0


# ------------------------------------------------------------------ service level, exact cosines


def _pair(cosine):
    first = np.zeros(192)
    first[0] = 1.0
    second = np.zeros(192)
    second[0], second[1] = cosine, math.sqrt(1 - cosine**2)
    return first, second


def _service():
    from backend.config import get_settings
    from backend.services.base_service import ModalityService

    class Stub:
        is_mock = False

        def embed(self, raw):  # the "recording" IS the embedding here
            return raw

    service = ModalityService("voice", Stub(), get_settings())
    service.capture_gate = False  # the "recordings" here are embeddings, not audio (the gate is tested elsewhere)
    return service


@pytest.mark.parametrize("cosine, quality", [(0.86, EXCELLENT), (0.85, EXCELLENT), (0.80, GOOD), (0.75, GOOD)])
def test_excellent_and_good_are_enrolled(db_session, cosine, quality):
    from backend.database import crud

    first, second = _pair(cosine)
    rows, band = _service().enroll_confirmed(db_session, first, second, "u", APPLICATION_ID)
    assert band == quality and len(rows) == 4  # T1 ACTIVE + T2-T4 STANDBY, as always
    assert [r.template_status for r in rows] == ["ACTIVE", "STANDBY", "STANDBY", "STANDBY"]
    assert crud.get_active_template(db_session, "u", "voice", APPLICATION_ID) is not None


@pytest.mark.parametrize("cosine", [0.74, 0.66, 0.60])
def test_fair_stores_nothing_until_the_user_continues(db_session, cosine):
    from backend.database import crud
    from backend.services.base_service import LowQualityWarning

    first, second = _pair(cosine)
    service = _service()
    with pytest.raises(LowQualityWarning) as warning:
        service.enroll_confirmed(db_session, first, second, "u", APPLICATION_ID)
    assert warning.value.quality == FAIR and str(warning.value) == FAIR_MESSAGE
    assert crud.get_set_rows(db_session, "u", APPLICATION_ID) == []  # nothing stored

    rows, band = service.enroll_confirmed(db_session, first, second, "u", APPLICATION_ID, accept_low_quality=True)
    assert band == FAIR and len(rows) == 4  # the user chose to continue: stored, same T1-T4 pool


@pytest.mark.parametrize("cosine", [0.59, 0.3, 0.0, -0.5])
def test_poor_is_rejected_and_cannot_be_overridden(db_session, cosine):
    from backend.database import crud
    from backend.services.base_service import EnrollmentInconsistent

    first, second = _pair(cosine)
    for accept in (False, True):
        with pytest.raises(EnrollmentInconsistent) as error:
            _service().enroll_confirmed(db_session, first, second, "u", APPLICATION_ID, accept_low_quality=accept)
        assert error.value.quality == POOR and POOR_MESSAGE in str(error.value)
    assert crud.get_set_rows(db_session, "u", APPLICATION_ID) == []


def test_the_templates_come_from_the_centroid_of_both_recordings(db_session):
    """Two-sentence enrollment protects the L2-normalized mean of the two unit embeddings (still 192-D)."""
    from backend.database import crud
    from backend.services.base_service import build_entry
    from embeddings.centroid import centroid_embedding

    first, second = _pair(0.9)
    service = _service()
    service.enroll_confirmed(db_session, first, second, "a", APPLICATION_ID)
    centroid = centroid_embedding([first, second])
    assert centroid.shape == (192,)
    assert build_entry(service.settings, centroid, user_id="a", application_id=APPLICATION_ID, modality="voice", key_version=1).protected_template == bytes(
        crud.get_active_template(db_session, "a", "voice", APPLICATION_ID).protected_template
    )
    assert service.authenticate(db_session, centroid, "a", APPLICATION_ID).hamming_similarity == 1.0
    assert service.authenticate(db_session, first, "a", APPLICATION_ID).hamming_similarity < 1.0  # not the first alone


# ------------------------------------------------------------------ API level


def _tone(frequency):
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    buffer = io.BytesIO()
    wavfile.write(buffer, 16000, (0.3 * speech_envelope(t) * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16))
    return buffer.getvalue()


class _AngleVoice:
    """Embedding = unit vector at angle f/1000 rad, f = the tone's dominant frequency."""

    is_mock = False

    def embed(self, raw):
        waveform, sample_rate = raw
        spectrum = np.abs(np.fft.rfft(np.asarray(waveform, dtype=np.float64)))
        frequency = np.fft.rfftfreq(len(waveform), 1.0 / sample_rate)[int(np.argmax(spectrum))]
        angle = frequency / 1000.0
        vector = np.zeros(192)
        vector[0], vector[1] = math.cos(angle), math.sin(angle)
        return vector


@pytest.fixture
def client(tmp_path, monkeypatch):
    import backend.services.voice_service as voice_service
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory
    from backend.services.base_service import ModalityService

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'vq.db'}")
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()

    @functools.lru_cache
    def getter():
        return ModalityService("voice", _AngleVoice(), get_settings())

    monkeypatch.setattr(voice_service, "get_voice_service", getter)
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()


def _enroll(client, first, second=None, user="u", **extra):
    files = {"image": ("v.wav", _tone(first), "audio/wav")}
    if second is not None:
        files["confirm_image"] = ("v2.wav", _tone(second), "audio/wav")
    return client.post("/enroll", data={"user_id": user, "modality": "voice", "application_id": APPLICATION_ID, **extra}, files=files)


def _state(client, user="u"):
    from backend.database import crud
    from backend.database.session import get_session_factory

    rows = [r for r in crud.get_set_rows(get_session_factory()(), user, APPLICATION_ID) if r.modality == "voice"]
    profile = client.get(f"/user/{user}/enrollment-status", params={"application_id": APPLICATION_ID}).json()
    return profile["statuses"]["voice"], len(rows)


# 200 Hz first recording (angle 0.2 rad); second recording chosen for an exact cosine = cos(delta/1000)
EXCELLENT_PAIR = (200, 500)  # delta 0.3 rad -> cos 0.955
GOOD_PAIR = (200, 800)       # delta 0.6 rad -> cos 0.825
FAIR_PAIR = (200, 980)       # delta 0.78 rad -> cos 0.711
POOR_PAIR = (200, 1200)      # delta 1.0 rad -> cos 0.540


def test_excellent_and_good_recordings_enroll_and_report_their_quality(client):
    ok = _enroll(client, *EXCELLENT_PAIR, user="a")
    assert ok.status_code == 200 and ok.json()["recording_quality"] == "EXCELLENT" and ok.json()["templates_created"] == 4
    ok = _enroll(client, *GOOD_PAIR, user="b")
    assert ok.status_code == 200 and ok.json()["recording_quality"] == "GOOD"
    assert _state(client, "a") == ("REGISTERED", 4) and _state(client, "b") == ("REGISTERED", 4)


def test_fair_returns_a_warning_and_stores_nothing(client):
    response = _enroll(client, *FAIR_PAIR)
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "LOW_QUALITY_WARNING" and body["recording_quality"] == "FAIR"
    assert body["detail"] == FAIR_MESSAGE == "Your recordings are usable, but quality is lower than recommended."
    assert body["similarity"] == pytest.approx(0.711, abs=0.005)  # (debug mode only - see the production test below)
    # not enrolled, and NOT counted as a failed attempt: a warning is not RETRY_REQUIRED
    assert _state(client) == ("NOT_REGISTERED", 0)


def test_fair_continue_enrolls_and_the_user_can_authenticate(client):
    assert _enroll(client, *FAIR_PAIR).status_code == 409
    cont = _enroll(client, *FAIR_PAIR, accept_low_quality="true")  # "Continue Enrollment"
    assert cont.status_code == 200 and cont.json()["recording_quality"] == "FAIR" and cont.json()["templates_created"] == 4
    assert _state(client) == ("REGISTERED", 4)
    auth = client.post(
        "/authenticate",
        data={"user_id": "u", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", _tone(200), "audio/wav")},
    ).json()
    assert auth["authentication_state"] == "ACCESS_GRANTED"


def test_fair_rerecord_then_a_good_pair_enrolls(client):
    assert _enroll(client, *FAIR_PAIR).status_code == 409  # "Re-record": nothing was kept, so just record again
    again = _enroll(client, *GOOD_PAIR)
    assert again.status_code == 200 and again.json()["recording_quality"] == "GOOD"
    assert _state(client) == ("REGISTERED", 4)


def test_poor_is_rejected_stores_nothing_and_needs_a_retry(client):
    response = _enroll(client, *POOR_PAIR)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "ENROLLMENT_INCONSISTENT" and body["recording_quality"] == "POOR"
    assert body["enrollment_status"] == "RETRY_REQUIRED" and POOR_MESSAGE in body["detail"]
    assert _state(client) == ("RETRY_REQUIRED", 0)
    # "continue" cannot force a poor pair through
    assert _enroll(client, *POOR_PAIR, accept_low_quality="true").status_code == 422
    assert _state(client) == ("RETRY_REQUIRED", 0)
    # recording again with a consistent pair succeeds and clears RETRY_REQUIRED
    assert _enroll(client, *EXCELLENT_PAIR).status_code == 200
    assert _state(client) == ("REGISTERED", 4)


def test_a_rejected_or_warned_reenrollment_keeps_the_previous_enrollment(client):
    assert _enroll(client, *EXCELLENT_PAIR).status_code == 200
    assert _enroll(client, 700, 1800).status_code == 422  # poor replacement (delta 1.1 rad, cos 0.45)
    assert _enroll(client, 300, 1100).status_code == 409  # fair replacement (delta 0.8 rad, cos 0.70), not confirmed
    assert _state(client) == ("REGISTERED", 4)  # still the original 4 rows, no revoked replacement
    auth = client.post(
        "/authenticate",
        data={"user_id": "u", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", _tone(200), "audio/wav")},
    ).json()
    assert auth["authentication_state"] == "ACCESS_GRANTED"


def test_a_single_recording_still_enrolls_without_any_quality_check(client):
    response = _enroll(client, 200)
    assert response.status_code == 200 and response.json().get("recording_quality") is None


def test_confirm_image_is_voice_only(client):
    response = client.post(
        "/enroll",
        data={"user_id": "u", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("a.png", b"x", "image/png"), "confirm_image": ("b.png", b"x", "image/png")},
    )
    assert response.status_code == 422 and "only used for voice" in response.json()["detail"]


def test_the_raw_similarity_is_debug_only(client, monkeypatch):
    monkeypatch.setenv("DEBUG_SCORES", "false")
    from backend.config import get_settings

    get_settings.cache_clear()
    for pair, status in ((FAIR_PAIR, 409), (POOR_PAIR, 422)):
        body = _enroll(client, *pair, user=f"p{status}").json()
        assert "similarity" not in body and body["recording_quality"] in ("FAIR", "POOR")  # only the band is public


def test_authentication_thresholds_and_pool_are_untouched(client):
    from backend.config import get_settings

    assert get_settings().match_threshold == 0.9 and get_settings().template_pool_size == 4
    assert _enroll(client, *GOOD_PAIR).json()["standby_template_set_versions"] == [2, 3, 4]
