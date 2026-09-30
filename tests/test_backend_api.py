"""Offline, full-stack tests for the FastAPI backend via `TestClient`.

Uses the "face" modality over the deterministic stub embedder from
tests/test_flexible_auth.py (the `stub_face` fixture) - real MTCNN face
detection cannot run on synthetic images - so the HTTP layer, services,
template protection and database are real while the embedding is not. The
last test exercises the real face preprocessing, skipped (this repo's
convention, see tests/test_preprocessing.py) when facenet-pytorch is missing.

Each test gets its own temporary, file-based SQLite database (a real
in-memory `:memory:` DB would be a *different* database per pooled
connection under FastAPI's threadpooled sync routes, breaking a
enroll-then-authenticate flow across two requests) via the `client` fixture,
which also resets every `lru_cache`d settings/engine singleton so tests don't
leak state into each other.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.test_flexible_auth import FACE, _stub_getter

APPLICATION_ID = "capstone-demo"


def _encode_png(image: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    assert ok
    return buffer.tobytes()


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


def _enroll(client: TestClient, image_bytes: bytes, user_id: str = "U001", modality: str = "face"):
    return client.post(
        "/enroll",
        data={"user_id": user_id, "modality": modality, "application_id": APPLICATION_ID},
        files={"image": ("sample.png", image_bytes, "image/png")},
    )


def test_health(client):
    # `client` (not a bare `TestClient(app)`) so this gets its own fresh, empty database - the
    # process-global `app`'s cached engine/session may otherwise point at whatever database an
    # earlier test in this session left with existing templates, which now legitimately trips
    # backend/key_continuity.py's startup check (a real database in that state should fail to
    # start) even though this test only cares that the bare /health endpoint returns 200.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_enroll_returns_success_and_template_metadata(client, stub_face):
    response = _enroll(client, FACE)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["user_id"] == "U001"
    assert body["modality"] == "face"
    assert body["key_version"] == 1
    assert body["template_version"] >= 1


def test_authenticate_after_enroll_succeeds(client, stub_face):
    image_bytes = FACE
    _enroll(client, image_bytes)

    response = client.post(
        "/authenticate",
        data={"user_id": "U001", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is True
    assert body["score"] >= body["threshold"]


def test_authenticate_without_enrollment_returns_enrollment_required(client, stub_face):
    """A submitted modality that was never enrolled is ENROLLMENT_REQUIRED (409) - not authenticated, not a denial."""
    response = client.post(
        "/authenticate",
        data={"user_id": "never-enrolled", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("sample.png", FACE, "image/png")},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "ENROLLMENT_REQUIRED" and body["missing_modalities"] == ["face"]
    assert "authenticated" not in body and "fusion_similarity" not in body


def test_verify_face_matches_authenticate_behavior(client, stub_face):
    image_bytes = FACE
    _enroll(client, image_bytes)

    response = client.post(
        "/verify/face",
        data={"user_id": "U001", "application_id": APPLICATION_ID},
        files={"image": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["authenticated"] is True


def test_revoke_template_rotates_key_version(client, stub_face):
    image_bytes = FACE
    _enroll(client, image_bytes)

    # Revocation is per template SET: it needs biometric authorization against the
    # ACTIVE set, then promotes the oldest STANDBY set (all modalities together).
    response = client.post(
        "/revoke-template",
        data={"user_id": "U001", "application_id": APPLICATION_ID},
        files={"face_image": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["revoked_template_set_version"] == 1
    assert body["new_active_template_set_version"] == 2
    assert body["remaining_standby_template_sets"] == 2

    # Re-authenticating still succeeds (re-derives under the new key_version).
    auth_response = client.post(
        "/authenticate",
        data={"user_id": "U001", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("sample.png", image_bytes, "image/png")},
    )
    assert auth_response.json()["authenticated"] is True


def test_get_user_returns_enrolled_modalities(client, stub_face):
    _enroll(client, FACE)

    response = client.get("/user/U001")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "U001"
    assert len(body["enrolled_modalities"]) == 1
    assert body["enrolled_modalities"][0]["modality"] == "face"


def test_get_user_returns_404_for_unknown_user(client):
    response = client.get("/user/nobody")
    assert response.status_code == 404


def test_delete_user_removes_templates(client, stub_face):
    _enroll(client, FACE)

    delete_response = client.delete("/user/U001")
    assert delete_response.status_code == 200
    # one enrollment = a pool of TEMPLATE_POOL_SIZE (default 4) templates
    assert delete_response.json()["templates_deleted"] == 4

    assert client.get("/user/U001").status_code == 404


def test_delete_user_is_idempotent_for_missing_user(client):
    response = client.delete("/user/nobody")
    assert response.status_code == 200
    assert response.json()["templates_deleted"] == 0


def test_enroll_rejects_unsupported_content_type(client):
    response = client.post(
        "/enroll",
        data={"user_id": "U001", "modality": "face", "application_id": APPLICATION_ID},
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_enroll_rejects_oversized_upload(tmp_path, monkeypatch, stub_face):
    db_path = tmp_path / "test_backend.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "10")
    _reset_caches()

    from backend.main import app

    with TestClient(app) as test_client:
        response = _enroll(test_client, FACE)

    _reset_caches()
    assert response.status_code == 413


def test_enroll_rejects_unsupported_modality(client, stub_face):
    for modality in ("retina", "fingerprint", "iris"):  # iris and fingerprint were removed from the system
        response = _enroll(client, FACE, modality=modality)
        assert response.status_code == 422, modality


def test_removed_modality_verify_routes_no_longer_exist(client):
    for route in ("/verify/iris", "/verify/fingerprint"):
        response = client.post(route, data={"user_id": "U001"}, files={"image": ("s.png", FACE, "image/png")})
        assert response.status_code == 404, route


def test_enroll_rejects_an_undetectable_face_sample_with_a_clean_422(client, random_rgb_image):
    """Real face preprocessing (MTCNN) raises ValueError("No face detected...")
    for a sample with no detectable face - backend/utils.py::call_modality_service
    must turn that into a clean 422, not an unhandled 500."""
    pytest.importorskip("facenet_pytorch", reason="facenet-pytorch not installed in this environment")

    response = _enroll(client, _encode_png(random_rgb_image), modality="face")
    assert response.status_code == 422
    assert "face" in response.json()["detail"]
