"""Offline tests for backend/services/*.py, exercised directly (no HTTP layer).

The enroll/authenticate/revoke roundtrips run the real `ModalityService` for face over the deterministic stub
embedder from tests/test_flexible_auth.py: face's real preprocessing (MTCNN via facenet-pytorch) correctly rejects
any synthetic image as "no face detected" (see
tests/test_preprocessing.py::test_face_preprocessing_requires_facenet_pytorch_and_detects_no_face_on_noise),
so it can't exercise a real roundtrip offline. The real face/voice service getters are checked for caching.

Loading the real face checkpoint needs `facenet-pytorch` - an optional, heavy ML dependency not every dev
environment has installed (see tests/test_preprocessing.py for the same `pytest.importorskip` convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_flexible_auth import _StubPipeline

APPLICATION_ID = "capstone-demo"


@pytest.fixture
def face_image():
    return np.random.default_rng(1).integers(0, 256, size=(300, 300, 3), dtype=np.uint8)


def _get_stub_face_service():
    from backend.config import get_settings
    from backend.services.base_service import ModalityService

    return ModalityService("face", _StubPipeline(512), get_settings())


def _get_face_service():
    pytest.importorskip("facenet_pytorch", reason="facenet-pytorch not installed in this environment")
    from backend.services.face_service import get_face_service

    return get_face_service()


def test_enroll_then_authenticate_with_the_same_image_succeeds(db_session, face_image):
    service = _get_stub_face_service()

    pool = service.enroll(db_session, face_image, user_id="U001", application_id=APPLICATION_ID)
    enrolled = pool[0]
    assert enrolled.modality == "face"
    assert enrolled.key_version == 1
    assert enrolled.is_active is True
    assert len(pool) == 4

    result = service.authenticate(db_session, face_image, user_id="U001", application_id=APPLICATION_ID)
    assert result.authenticated is True
    assert result.score >= result.threshold


def test_authenticate_without_enrollment_fails_closed(db_session, face_image):
    service = _get_stub_face_service()
    result = service.authenticate(db_session, face_image, user_id="never-enrolled", application_id=APPLICATION_ID)
    assert result.authenticated is False
    assert result.score == 0.0


def test_revoke_promotes_the_next_set_and_the_same_biometric_still_authenticates(db_session, face_image):
    from backend.database import crud

    service = _get_stub_face_service()
    pool = service.enroll(db_session, face_image, user_id="U001", application_id=APPLICATION_ID)
    original_bytes = pool[0].protected_template

    revoked, promoted = crud.revoke_active_set_and_promote(db_session, "U001", APPLICATION_ID)
    assert (revoked, promoted) == (1, 2)
    summaries = {s.version: s.status for s in crud.get_template_sets(db_session, "U001", APPLICATION_ID)}
    assert summaries == {1: "REVOKED", 2: "ACTIVE", 3: "STANDBY", 4: "STANDBY"}

    active = crud.get_active_template(db_session, "U001", "face", APPLICATION_ID)
    assert active.template_set_version == 2 and active.key_version == 2 and active.is_active is True
    # a different key over the same embedding: genuinely different template bits
    assert active.protected_template != original_bytes

    result = service.authenticate(db_session, face_image, user_id="U001", application_id=APPLICATION_ID)
    assert result.authenticated is True
    assert result.key_version == 2 and result.template_set_version == 2


def test_revoke_without_a_prior_enrollment_is_not_found(db_session):
    import pytest as _pytest

    from backend.database import crud

    with _pytest.raises(crud.TemplateNotFoundError):
        crud.revoke_active_set_and_promote(db_session, "U004", APPLICATION_ID)
    assert crud.get_user(db_session, "U004") is None


def test_a_different_image_does_not_authenticate(db_session, face_image):
    service = _get_stub_face_service()
    service.enroll(db_session, face_image, user_id="U003", application_id=APPLICATION_ID)
    other = np.random.default_rng(2).integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
    result = service.authenticate(db_session, other, user_id="U003", application_id=APPLICATION_ID)
    assert result.authenticated is False


def test_get_face_service_returns_the_same_cached_instance():
    service = _get_face_service()
    assert service is _get_face_service()


def test_get_voice_service_returns_the_same_cached_instance():
    from backend.services.voice_service import get_voice_service

    assert get_voice_service() is get_voice_service()


def test_unsupported_modality_is_rejected():
    from fastapi import HTTPException

    from backend.services import get_service_for_modality

    for retired in ("fingerprint", "iris"):
        with pytest.raises(HTTPException) as error:
            get_service_for_modality(retired)
        assert error.value.status_code == 422


def test_concurrent_active_template_insert_is_rejected_at_the_db_level(db_session, face_image, monkeypatch):
    """Two racing requests both read 'no template sets yet' before either commits, then both
    try to create the ACTIVE set. Reproduced deterministically by making the second
    `save_modality_templates` see no existing rows, so its insert collides with the first
    call's committed ACTIVE row at the DB level (partial unique indexes on `ProtectedTemplate`).
    That collision, and its ConcurrentEnrollmentError, are what is under test - the patch only
    forces the timing."""
    from backend.database import crud
    from backend.database.crud import ConcurrentEnrollmentError

    _get_stub_face_service().enroll(db_session, face_image, user_id="U005", application_id=APPLICATION_ID)

    monkeypatch.setattr(crud, "get_set_rows", lambda *args, **kwargs: [])
    with pytest.raises(ConcurrentEnrollmentError):
        crud.save_modality_templates(
            db_session,
            user_id="U005",
            application_id=APPLICATION_ID,
            modality="face",
            template_version=1,
            output_bits=128,
            entries={1: crud.PoolEntry(key_version=99, protected_template=b"\x00")},
        )
