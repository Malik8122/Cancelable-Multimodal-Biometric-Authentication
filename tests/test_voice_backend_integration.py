"""Offline tests for Voice's backend integration (Phase 3A, Step 2).

Voice was added as a modality after Phase 2's backend was built, so it never
had a `backend/services/voice_service.py` or a `/verify/voice` route until
this integration - these tests exist specifically to prove that gap is
closed: the real trained checkpoint loads (not mock mode), the
`_VoicePipelineAdapter` correctly bridges `VoicePipeline`'s two-argument
`embed(waveform, sample_rate)` to `ModalityService`'s single-argument
`embed(raw_image)` contract, and the full enroll -> verify flow works
end-to-end through the actual HTTP API, mirroring `tests/test_backend_api.py`'s
fixture/pattern exactly (face there, voice here).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from tests.voice_signals import speech_envelope

APPLICATION_ID = "capstone-demo"
SAMPLE_RATE = 16_000


def _encode_wav(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    int16_waveform = (waveform * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, int16_waveform)
    return buffer.getvalue()


def _tone(duration_seconds: float = 2.0, frequency: float = 220.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
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


def test_voice_pipeline_adapter_bridges_the_two_argument_embed_call():
    """No speechbrain/torchaudio needed - proves the adapter's wiring itself
    is correct, independent of whether the real backbone is installed."""
    from backend.services.voice_service import _VoicePipelineAdapter
    from embeddings.pipelines import VoicePipeline

    adapter = _VoicePipelineAdapter(VoicePipeline(checkpoint_path=None))  # mock mode
    assert adapter.is_mock is True
    assert adapter.embedding_dim == 192

    embedding = adapter.embed((_tone(), SAMPLE_RATE))
    assert embedding.shape == (192,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_get_voice_service_loads_the_real_checkpoint_not_mock_mode():
    """Per the integration requirement: /verify/voice must not silently fall
    back to mock mode. This fails loudly if the committed checkpoint is ever
    missing or fails to load, rather than passing accidentally."""
    pytest.importorskip("speechbrain", reason="speechbrain not installed in this environment")
    pytest.importorskip("torchaudio", reason="torchaudio not installed in this environment")
    from backend.services.voice_service import get_voice_service

    service = get_voice_service()
    assert service.modality == "voice"
    assert service.pipeline.is_mock is False, "voice checkpoint failed to load - falling back to mock mode"
    assert service.pipeline.embedding_dim == 192


def test_enroll_then_verify_voice_via_the_real_api(client):
    """Full HTTP round-trip: POST /enroll (modality=voice) -> POST /verify/voice,
    using the real trained checkpoint - confirms decode_biometric_sample's
    voice branch, the adapter, and the route wiring all work together."""
    pytest.importorskip("speechbrain", reason="speechbrain not installed in this environment")
    pytest.importorskip("torchaudio", reason="torchaudio not installed in this environment")

    wav_bytes = _encode_wav(_tone())

    enroll_response = client.post(
        "/enroll",
        data={"user_id": "U-VOICE-API", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert enroll_response.status_code == 200
    assert enroll_response.json()["modality"] == "voice"

    verify_response = client.post(
        "/verify/voice",
        data={"user_id": "U-VOICE-API", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert verify_response.status_code == 200
    body = verify_response.json()
    assert body["modality"] == "voice"
    assert body["authenticated"] is True
    assert body["score"] >= body["threshold"]


def test_recapture_with_small_noise_is_no_longer_catastrophically_unstable(client):
    """Regression test for the authentication-debug-sprint bug: before fixing
    `preprocessing/voice.py::VoicePreprocessor._trim_silence`'s overlapping-
    frame duplication, a genuine same-speaker "recapture" (the same tone plus
    a tiny 1% amount of noise, standing in for real sensor/mic variation)
    produced a similarity score of ~0.13-0.59 depending on noise level - once
    even *negative* (-0.18) - because the unstable, duplicated intermediate
    waveform meant the final center-cropped segment could end up almost
    unrelated to the original. This doesn't assert a pass against the
    (separately tracked, uncalibrated) match threshold - see
    tests/test_authentication_debug_sprint.py for why that's a distinct,
    not-yet-resolved gap - only that the fix restores basic stability: a
    small perturbation must produce a *clearly correlated* result, not one
    indistinguishable from (or worse than) chance.
    """
    pytest.importorskip("speechbrain", reason="speechbrain not installed in this environment")
    pytest.importorskip("torchaudio", reason="torchaudio not installed in this environment")

    base = _tone(4.0)
    rng = np.random.default_rng(5)
    recaptured = base + rng.normal(scale=0.01 * np.std(base), size=base.shape).astype(np.float32)

    enroll_response = client.post(
        "/enroll",
        data={"user_id": "U-VOICE-STABILITY", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", _encode_wav(base), "audio/wav")},
    )
    assert enroll_response.status_code == 200

    verify_response = client.post(
        "/verify/voice",
        data={"user_id": "U-VOICE-STABILITY", "application_id": APPLICATION_ID},
        files={"image": ("recapture.wav", _encode_wav(recaptured), "audio/wav")},
    )
    assert verify_response.status_code == 200
    body = verify_response.json()
    # 0.5 is the Hamming-similarity midpoint (statistically-unrelated bit
    # strings average ~0.5, see template_protection/biohash.py) - a genuine
    # recapture must land clearly above that, not at-or-below chance.
    assert body["score"] > 0.5, f"genuine recapture score collapsed to {body['score']} - the trim_silence bug may have regressed"


def _speech_like(seed: int, seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A broadband, envelope-shaped 'utterance' - continuous spectral energy
    like real speech (unlike a pure tone, which this real ECAPA-TDNN model
    handles unpredictably - see this module's other tests)."""
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    b, a = butter(4, [300 / (sample_rate / 2), 3400 / (sample_rate / 2)], btype="band")
    signal = lfilter(b, a, rng.standard_normal(len(t))).astype(np.float32)
    envelope = (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)).astype(np.float32)
    signal = signal * envelope
    return (signal / np.std(signal) * 0.1).astype(np.float32)


def test_genuine_recapture_at_a_different_realistic_duration_authenticates(client):
    """Regression test for the reliability-sprint bug fixed in
    preprocessing/voice.py::VoicePreprocessor.preprocess: a second, genuine
    recording will rarely last exactly as long as the enrollment recording -
    a real speaker naturally varies how long they take. Before the fix, a
    second take a little *shorter* than the enrolled one (so its VAD-trimmed
    content fell short of the 4-second target and got padded with raw
    silence before mel extraction) collapsed to cosine similarity 0.26 with
    the enrolled embedding - worse than two genuinely unrelated recordings
    typically score. After the fix (padding happens in the mel domain, after
    per-clip mean-normalization, so the padded frames never contaminate the
    real speech frames' statistics), the same scenario measures ~0.99 cosine
    / ~0.97-0.98 Hamming similarity and authenticates at the ORIGINAL,
    unmodified 0.9 threshold - no threshold change was needed or made.
    """
    pytest.importorskip("speechbrain", reason="speechbrain not installed in this environment")
    pytest.importorskip("torchaudio", reason="torchaudio not installed in this environment")

    enrolled = _speech_like(seed=1, seconds=4.0)
    second_take = _speech_like(seed=1, seconds=3.7)  # same "speaker" pattern, naturally shorter take

    enroll_response = client.post(
        "/enroll",
        data={"user_id": "U-VOICE-GENUINE", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", _encode_wav(enrolled), "audio/wav")},
    )
    assert enroll_response.status_code == 200

    verify_response = client.post(
        "/verify/voice",
        data={"user_id": "U-VOICE-GENUINE", "application_id": APPLICATION_ID},
        files={"image": ("recapture.wav", _encode_wav(second_take), "audio/wav")},
    )
    assert verify_response.status_code == 200
    body = verify_response.json()
    # The regression bar is the ORIGINAL 0.9: the genuine recapture must clear it, whatever threshold voice is configured with
    # (evaluation/results/voice_threshold.json currently sets 0.8), so this stays meaningful and cannot pass by threshold relaxation alone.
    assert body["score"] >= 0.9, f"genuine recapture scored {body['score']}, below the original 0.9 bar"
    assert body["authenticated"] is True, f"genuine recapture (score={body['score']}, threshold={body['threshold']}) was denied"


def test_verify_voice_without_enrollment_returns_enrollment_required(client):
    pytest.importorskip("speechbrain", reason="speechbrain not installed in this environment")
    pytest.importorskip("torchaudio", reason="torchaudio not installed in this environment")

    response = client.post(
        "/verify/voice",
        data={"user_id": "never-enrolled", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", _encode_wav(_tone()), "audio/wav")},
    )
    assert response.status_code == 409
    assert response.json()["status"] == "ENROLLMENT_REQUIRED"


def test_enroll_voice_rejects_a_non_audio_content_type(client):
    response = client.post(
        "/enroll",
        data={"user_id": "U-VOICE-API", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415


def test_enroll_voice_rejects_a_corrupt_wav_file(client):
    response = client.post(
        "/enroll",
        data={"user_id": "U-VOICE-API", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("broken.wav", b"not actually a wav file", "audio/wav")},
    )
    assert response.status_code == 400
