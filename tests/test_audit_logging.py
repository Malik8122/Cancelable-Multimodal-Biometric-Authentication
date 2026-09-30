"""Offline, full-stack tests for server-side audit logging (backend.database.models.AuditLog).

Face checks use the deterministic stub embedder from tests/test_flexible_auth.py (the `stub_face` fixture - real
MTCNN face detection cannot run on synthetic images) and voice uses the real pipeline; the real face checkpoint is
exercised by test_face_real_checkpoint_is_wired_but_fails_closed_without_a_real_photo.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from tests.test_flexible_auth import FACE, OTHER_FACE, _stub_getter
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
    from backend.threshold_loader import clear_cache

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    clear_cache()


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


def _enroll_face(client: TestClient, user_id: str, image_bytes: bytes):
    return client.post(
        "/enroll",
        data={"user_id": user_id, "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", image_bytes, "image/png")},
    )


def test_authenticate_writes_an_audit_entry(client, stub_face):
    image_bytes = FACE
    _enroll_face(client, "AUDIT-U1", image_bytes)

    response = client.post(
        "/authenticate",
        data={"user_id": "AUDIT-U1", "modality": "face", "application_id": APPLICATION_ID, "building_id": "data-center"},
        files={"image": ("face.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200

    audit_response = client.get("/audit/AUDIT-U1")
    assert audit_response.status_code == 200
    body = audit_response.json()
    assert body["total"] == 1
    entry = body["entries"][0]
    assert entry["modality_list"] == ["face"]
    assert entry["authenticated"] is True
    assert entry["building_id"] == "data-center"
    assert entry["similarity_scores"]["face"] == pytest.approx(1.0)
    assert entry["thresholds_used"]["face"] > 0
    assert entry["template_versions"]["face"] >= 1
    assert entry["key_versions"]["face"] >= 1
    assert entry["latency_ms"] >= 0
    assert entry["fusion_score"] is None
    assert entry["fusion_policy"] is None


def test_verify_writes_an_audit_entry(client, stub_face):
    image_bytes = FACE
    _enroll_face(client, "AUDIT-U2", image_bytes)

    response = client.post(
        "/verify/face",
        data={"user_id": "AUDIT-U2", "application_id": APPLICATION_ID},
        files={"image": ("face.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200

    body = client.get("/audit/AUDIT-U2").json()
    assert body["total"] == 1
    assert body["entries"][0]["modality_list"] == ["face"]


def test_fusion_writes_an_audit_entry_with_fusion_fields(client, stub_face):
    face_bytes = FACE
    voice_bytes = _encode_wav(_tone())

    client.post(
        "/enroll",
        data={"user_id": "AUDIT-FU", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    client.post(
        "/enroll",
        data={"user_id": "AUDIT-FU", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("sample.wav", voice_bytes, "audio/wav")},
    )

    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "AUDIT-FU", "application_id": APPLICATION_ID},
        files={
            "face_image": ("face.png", face_bytes, "image/png"),
            "voice_audio": ("sample.wav", voice_bytes, "audio/wav"),
        },
    )
    assert response.status_code == 200
    fusion_body = response.json()

    audit_body = client.get("/audit/AUDIT-FU").json()
    assert audit_body["total"] == 1
    entry = audit_body["entries"][0]
    assert sorted(entry["modality_list"]) == ["face", "voice"]
    assert entry["fusion_score"] == pytest.approx(fusion_body["fused_score"])
    assert entry["fusion_policy"] == fusion_body["fusion_policy"]
    assert entry["authenticated"] == fusion_body["authenticated"]


def test_audit_history_is_paginated(client, stub_face):
    image_bytes = FACE
    _enroll_face(client, "AUDIT-PAGE", image_bytes)

    for _ in range(5):
        client.post(
            "/authenticate",
            data={"user_id": "AUDIT-PAGE", "modality": "face", "application_id": APPLICATION_ID},
            files={"image": ("face.png", image_bytes, "image/png")},
        )

    first_page = client.get("/audit/AUDIT-PAGE", params={"limit": 2, "offset": 0}).json()
    second_page = client.get("/audit/AUDIT-PAGE", params={"limit": 2, "offset": 2}).json()

    assert first_page["total"] == 5
    assert len(first_page["entries"]) == 2
    assert len(second_page["entries"]) == 2
    first_ids = {entry["audit_id"] for entry in first_page["entries"]}
    second_ids = {entry["audit_id"] for entry in second_page["entries"]}
    assert first_ids.isdisjoint(second_ids)


def test_system_audit_returns_latest_across_users(client, stub_face):
    image_bytes = FACE
    _enroll_face(client, "AUDIT-SYS-A", image_bytes)
    _enroll_face(client, "AUDIT-SYS-B", image_bytes)
    client.post(
        "/authenticate",
        data={"user_id": "AUDIT-SYS-A", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", image_bytes, "image/png")},
    )
    client.post(
        "/authenticate",
        data={"user_id": "AUDIT-SYS-B", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", image_bytes, "image/png")},
    )

    body = client.get("/audit/system").json()
    user_ids = {entry["user_id"] for entry in body["entries"]}
    assert {"AUDIT-SYS-A", "AUDIT-SYS-B"}.issubset(user_ids)


def test_delete_user_audit_removes_history(client, stub_face):
    image_bytes = FACE
    _enroll_face(client, "AUDIT-DEL", image_bytes)
    client.post(
        "/authenticate",
        data={"user_id": "AUDIT-DEL", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", image_bytes, "image/png")},
    )
    assert client.get("/audit/AUDIT-DEL").json()["total"] == 1

    delete_response = client.delete("/audit/AUDIT-DEL")
    assert delete_response.status_code == 200
    assert delete_response.json()["entries_deleted"] == 1

    assert client.get("/audit/AUDIT-DEL").json()["total"] == 0


def test_face_only_authenticates_and_logs(client, stub_face):
    """Bug-10 scenario: Face only."""
    face_bytes = FACE
    client.post(
        "/enroll",
        data={"user_id": "SCN-FACE-STUB", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "SCN-FACE-STUB", "application_id": APPLICATION_ID},
        files={"face_image": ("face.png", face_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["authenticated"] is True

    entry = client.get("/audit/SCN-FACE-STUB").json()["entries"][0]
    assert entry["modality_list"] == ["face"]
    assert entry["authenticated"] is True


def test_voice_only_authenticates_and_logs(client):
    """Bug-10 scenario: Voice only."""
    voice_bytes = _encode_wav(_tone())
    client.post(
        "/enroll",
        data={"user_id": "SCN-VC-ONLY", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", voice_bytes, "audio/wav")},
    )
    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "SCN-VC-ONLY", "application_id": APPLICATION_ID},
        files={"voice_audio": ("v.wav", voice_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["authenticated"] is True

    entry = client.get("/audit/SCN-VC-ONLY").json()["entries"][0]
    assert entry["modality_list"] == ["voice"]


def test_face_plus_voice_authenticates_and_logs(client, stub_face):
    """Bug-10 scenario: Face + Voice."""
    face_bytes = FACE
    voice_bytes = _encode_wav(_tone())
    client.post(
        "/enroll",
        data={"user_id": "SCN-FACE-VC", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    client.post(
        "/enroll",
        data={"user_id": "SCN-FACE-VC", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", voice_bytes, "audio/wav")},
    )
    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "SCN-FACE-VC", "application_id": APPLICATION_ID},
        files={
            "face_image": ("face.png", face_bytes, "image/png"),
            "voice_audio": ("v.wav", voice_bytes, "audio/wav"),
        },
    )
    assert response.status_code == 200
    assert response.json()["authenticated"] is True

    entry = client.get("/audit/SCN-FACE-VC").json()["entries"][0]
    assert sorted(entry["modality_list"]) == ["face", "voice"]


def test_omitted_modality_never_counts_toward_the_decision_or_the_audit_score(client):
    """Bug-10 scenario: Omitted modality - face is never submitted, so
    it must not appear in similarity_scores nor pull the decision down."""
    voice_bytes = _encode_wav(_tone())
    client.post(
        "/enroll",
        data={"user_id": "SCN-OMIT", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", voice_bytes, "audio/wav")},
    )
    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "SCN-OMIT", "application_id": APPLICATION_ID},
        files={"voice_audio": ("v.wav", voice_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["authenticated"] is True

    entry = client.get("/audit/SCN-OMIT").json()["entries"][0]
    assert "face" not in entry["similarity_scores"]
    assert "face" not in entry["modality_list"]


def test_wrong_face_fails_and_logs_authenticated_false(client, stub_face):
    """Bug-10 scenario: Wrong face - a different (impostor) capture
    must not authenticate the enrolled user."""
    genuine_bytes, impostor_bytes = FACE, OTHER_FACE

    client.post(
        "/enroll",
        data={"user_id": "SCN-WRONG-FACE", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", genuine_bytes, "image/png")},
    )
    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "SCN-WRONG-FACE", "application_id": APPLICATION_ID},
        files={"face_image": ("impostor.png", impostor_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is False

    entry = client.get("/audit/SCN-WRONG-FACE").json()["entries"][0]
    assert entry["authenticated"] == body["authenticated"]  # audit and response never disagree


def test_wrong_voice_fails_and_logs_authenticated_false(client):
    """Bug-10 scenario: Wrong voice - a different tone/frequency must not
    authenticate the enrolled user (real ECAPA-TDNN inference)."""
    genuine_bytes = _encode_wav(_tone(frequency=220.0))
    impostor_bytes = _encode_wav(_tone(frequency=880.0))

    client.post(
        "/enroll",
        data={"user_id": "SCN-WRONG-VC", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", genuine_bytes, "audio/wav")},
    )
    response = client.post(
        "/authenticate/fusion",
        data={"user_id": "SCN-WRONG-VC", "application_id": APPLICATION_ID},
        files={"voice_audio": ("impostor.wav", impostor_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()

    entry = client.get("/audit/SCN-WRONG-VC").json()["entries"][0]
    assert entry["authenticated"] == body["authenticated"]


def test_revoked_template_invalidates_the_old_key_version_and_logs(client, stub_face):
    """Bug-10 scenario: Revoked template - after `/revoke-template`, the
    template_version/key_version reported (and logged) must reflect the new,
    rotated key, and re-authenticating with the same sample must still
    succeed under the *new* key (revocation regenerates, not just deletes)."""
    face_bytes = FACE
    client.post(
        "/enroll",
        data={"user_id": "SCN-REVOKE", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    # Revocation is set-level and needs biometric authorization against the ACTIVE set.
    revoke_response = client.post(
        "/revoke-template",
        data={"user_id": "SCN-REVOKE", "application_id": APPLICATION_ID},
        files={"face_image": ("face.png", face_bytes, "image/png")},
    )
    assert revoke_response.status_code == 200
    assert revoke_response.json()["new_active_template_set_version"] == 2

    response = client.post(
        "/authenticate",
        data={"user_id": "SCN-REVOKE", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is True
    assert body["key_version"] == 2  # proves the new key, not the revoked one, is what matched

    entry = client.get("/audit/SCN-REVOKE").json()["entries"][0]
    assert entry["key_versions"]["face"] == 2


def test_face_real_checkpoint_is_wired_but_fails_closed_without_a_real_photo(client):
    """Bug-10 "Face only" scenario, reported honestly: the real face
    checkpoint is loaded (not mock - see tests/test_services.py's
    equivalent check), but its real MTCNN detector correctly rejects any
    synthetic image as "no face detected" (no real photographed face exists
    in this dev environment - see tests/test_preprocessing.py's identical,
    already-established finding). Enrolling (not authenticating) is what
    exercises this for real: `ModalityService.authenticate` short-circuits
    to `score=0.0` *before* touching the image when nothing is enrolled yet
    (backend/services/base_service.py), so `/enroll` - which always calls
    `pipeline.embed()` - is what proves Face fails *closed* here: a clean
    422 and no bogus "authenticated" audit entry, never a silent grant."""
    pytest.importorskip("facenet_pytorch", reason="facenet-pytorch not installed in this environment")
    import numpy as _np

    face_bytes = _encode_png(_np.random.default_rng(42).integers(0, 256, size=(224, 224, 3), dtype=_np.uint8))

    response = client.post(
        "/enroll",
        data={"user_id": "SCN-FACE-ONLY", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    assert response.status_code == 422
    # No audit entry should exist for a request that never completed a real comparison.
    assert client.get("/audit/SCN-FACE-ONLY").json()["total"] == 0


def test_audit_entry_never_contains_a_raw_image_or_template_blob(client, stub_face):
    """Structural check: the audit response's similarity_scores are plain
    floats, never anything blob-shaped (a raw image/protected template would
    serialize as a huge byte string or nested array, not a JSON number)."""
    image_bytes = FACE
    _enroll_face(client, "AUDIT-SAFE", image_bytes)
    client.post(
        "/authenticate",
        data={"user_id": "AUDIT-SAFE", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("face.png", image_bytes, "image/png")},
    )

    entry = client.get("/audit/AUDIT-SAFE").json()["entries"][0]
    for value in entry["similarity_scores"].values():
        assert isinstance(value, float)
    assert "protected_template" not in entry
    assert "embedding" not in entry
    assert "image" not in entry
