"""Offline, full-stack tests for POST /authenticate/fusion via `TestClient`.

Voice uses its real checkpoint (models/voice/saved/, speechbrain/torchaudio),
so voice scores come from real backend inference, not mock mode. Face uses the
deterministic stub embedder from tests/test_flexible_auth.py (the `stub_face`
fixture): real MTCNN face detection cannot run on the synthetic images these
offline tests have, so face-specific real-checkpoint behavior is asserted
elsewhere (tests/test_genuine_auth_regression.py's real-models test).
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from tests.test_flexible_auth import FACE, _stub_getter
from tests.voice_signals import speech_envelope

APPLICATION_ID = "capstone-demo"
SAMPLE_RATE = 16_000


def _encode_png(image: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    assert ok
    return buffer.tobytes()


def _encode_wav(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    int16_waveform = (waveform * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, int16_waveform)
    return buffer.getvalue()


def _tone(duration_seconds: float = 2.0, frequency: float = 220.0) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(SAMPLE_RATE * duration_seconds), endpoint=False)
    return (0.3 * speech_envelope(t) * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _reset_caches():
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_backend.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    _reset_caches()

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client

    _reset_caches()


@pytest.fixture
def stub_face(monkeypatch):
    import backend.services.face_service as face_service

    monkeypatch.setattr(face_service, "get_face_service", _stub_getter("face", 512))


def test_fusion_requires_at_least_one_modality(client):
    response = client.post("/authenticate/fusion", data={"user_id": "U1", "application_id": APPLICATION_ID})
    assert response.status_code == 422


def test_fusion_single_modality_equals_that_modalitys_own_score(client):
    wav_bytes = _encode_wav(_tone())
    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-1", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", wav_bytes, "audio/wav")},
    )

    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-1", "application_id": APPLICATION_ID},
        files={"voice_audio": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["modalities_used"] == ["voice"]
    assert body["fused_score"] == pytest.approx(body["results"]["voice"]["score"])
    assert body["authenticated"] == (body["fused_score"] >= body["fusion_threshold"])


def test_fusion_two_modalities_face_and_voice(client, stub_face):
    face_bytes = FACE
    voice_bytes = _encode_wav(_tone())

    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-2", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-2", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", voice_bytes, "audio/wav")},
    )

    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-2", "application_id": APPLICATION_ID},
        files={
            "face_image": ("face.png", face_bytes, "image/png"),
            "voice_audio": ("sample.wav", voice_bytes, "audio/wav"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert sorted(body["modalities_used"]) == ["face", "voice"]

    expected_fused = (body["results"]["face"]["score"] + body["results"]["voice"]["score"]) / 2
    assert body["fused_score"] == pytest.approx(expected_fused)


def test_fusion_submitting_a_never_enrolled_face_is_enrollment_required(client, stub_face):
    """Proves the endpoint accepts and routes both `UploadFile` fields together, with real voice inference. Face was
    submitted but never enrolled, so the attempt is ENROLLMENT_REQUIRED and nothing is evaluated (never a fabricated
    success or a zero-score failure); presenting only what IS enrolled then fuses exactly that."""
    face_bytes = FACE
    voice_bytes = _encode_wav(_tone())

    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-3", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", voice_bytes, "audio/wav")},
    )

    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-3", "application_id": APPLICATION_ID},
        files={
            "face_image": ("face.png", face_bytes, "image/png"),
            "voice_audio": ("sample.wav", voice_bytes, "audio/wav"),
        },
    )
    # face was submitted but never enrolled: ENROLLMENT_REQUIRED, and nothing at all is evaluated
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "ENROLLMENT_REQUIRED" and body["missing_modalities"] == ["face"]
    assert body["enrolled_modalities"] == ["voice"]

    # the user simply presents what IS enrolled: fusion runs over exactly that
    ok = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-3", "application_id": APPLICATION_ID},
        files={"voice_audio": ("sample.wav", voice_bytes, "audio/wav")},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["modalities_used"] == ["voice"]
    assert body["results"]["voice"]["authenticated"] is True
    assert body["fused_score"] == pytest.approx(body["results"]["voice"]["score"])


def test_fusion_diagnostics_present_and_matches_the_real_response_values(client, stub_face):
    """`fusion_diagnostics` (DEBUG_SCORES=true, on by default in this suite - see conftest.py)
    must report the exact same numbers already present elsewhere in the response - it's a
    reshaping of existing values, never a second, independently-computed fusion result."""
    face_bytes = FACE
    voice_bytes = _encode_wav(_tone())

    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-DIAG-1", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-DIAG-1", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", voice_bytes, "audio/wav")},
    )

    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-DIAG-1", "application_id": APPLICATION_ID},
        files={
            "face_image": ("face.png", face_bytes, "image/png"),
            "voice_audio": ("sample.wav", voice_bytes, "audio/wav"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    diagnostics = body["fusion_diagnostics"]

    for modality in ("face", "voice"):
        assert diagnostics[modality]["score"] == pytest.approx(body["results"][modality]["score"])
        assert diagnostics[modality]["threshold"] == pytest.approx(body["results"][modality]["threshold"])
        assert diagnostics[modality]["verified"] == body["results"][modality]["authenticated"]

    assert diagnostics["weights"] == {"face": 0.5, "voice": 0.5}
    assert diagnostics["fused_score"] == pytest.approx(body["fused_score"])
    assert diagnostics["threshold"] == pytest.approx(body["fusion_threshold"])
    assert diagnostics["policy"] == body["fusion_policy"]
    assert diagnostics["access_granted"] == body["authenticated"]

    # face not submitted in this request - represented, never fabricated.
    voice_only = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-DIAG-1", "application_id": APPLICATION_ID},
        files={"voice_audio": ("sample.wav", voice_bytes, "audio/wav")},
    ).json()["fusion_diagnostics"]
    assert voice_only["face"] == {"score": None, "threshold": None, "verified": None, "status": "not_presented"}
    assert voice_only["weights"] == {"voice": 1.0}


def test_fusion_diagnostics_decision_matches_backend_even_when_fused_score_alone_would_mislead(client, stub_face):
    """The end-to-end version of the ALL_REQUIRED override: enroll face and voice with
    matching samples (both pass individually, fused score should be high), then confirm the
    diagnostics' access_granted always equals the backend's actual `authenticated` - never an
    independently-derived "fused_score >= threshold" shortcut, which is exactly the bug
    fusion/config.py's docstring documents ALL_REQUIRED was introduced to fix."""
    face_bytes = FACE
    voice_bytes = _encode_wav(_tone())

    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-DIAG-2", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-DIAG-2", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", voice_bytes, "audio/wav")},
    )

    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-DIAG-2", "application_id": APPLICATION_ID},
        files={
            "face_image": ("face.png", face_bytes, "image/png"),
            "voice_audio": ("sample.wav", voice_bytes, "audio/wav"),
        },
    )
    body = response.json()
    diagnostics = body["fusion_diagnostics"]

    # The displayed/reported decision must match the backend's real decision under ALL_REQUIRED:
    # every submitted modality individually verified, never inferred from fused_score alone.
    all_verified = all(diagnostics[m]["verified"] for m in ("face", "voice"))
    assert diagnostics["access_granted"] == (all_verified and body["authenticated"])
    assert diagnostics["access_granted"] == body["authenticated"]


def test_fusion_diagnostics_omitted_from_production_responses(client, monkeypatch):
    """DEBUG_SCORES=false (production) must not expose fusion_diagnostics, same as the other
    per-modality debug-only fields (see test_flexible_auth.py::test_production_response_exposes_only_the_fusion_values)."""
    monkeypatch.setenv("DEBUG_SCORES", "false")
    from backend.config import get_settings

    get_settings.cache_clear()

    wav_bytes = _encode_wav(_tone())
    client.post(
        "/enroll",
        data={"user_id": "U-FUSION-DIAG-3", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", wav_bytes, "audio/wav")},
    )
    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "U-FUSION-DIAG-3", "application_id": APPLICATION_ID},
        files={"voice_audio": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    assert "fusion_diagnostics" not in response.json()

    get_settings.cache_clear()  # restore DEBUG_SCORES=true for any test after this one in the same session


def test_fusion_unenrolled_modality_is_enrollment_required_not_a_zero_score(client):
    """A submitted modality that was never enrolled is ENROLLMENT_REQUIRED (409): it is not authenticated and not
    scored - never silently dropped, never fabricated as a zero-score failure."""
    wav_bytes = _encode_wav(_tone())

    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "never-enrolled", "application_id": APPLICATION_ID},
        files={"voice_audio": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "ENROLLMENT_REQUIRED" and body["missing_modalities"] == ["voice"]
    assert "results" not in body and "authenticated" not in body
