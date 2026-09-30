"""The complete face + voice workflow through the real HTTP layer, services, BioHash, database, fusion and audit log.

registration -> face capture (5 poses) -> quality -> embedding -> BioHash -> voice capture -> embedding -> BioHash ->
template storage -> authentication -> Hamming match -> metric estimate -> modality decision -> fusion -> grant/deny ->
audit log; plus face/voice failure, wrong face, wrong voice, revoked template, new template set, invalid upload,
malformed input and alignment/landmark failure.

Embedders are the deterministic stubs of tests/test_flexible_auth.py (SOFTWARE TEST - not a biometric trial); every
other layer is the shipped code.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_flexible_auth import (  # noqa: F401  (client is a fixture)
    APPLICATION_ID,
    FACE,
    OTHER_FACE,
    OTHER_VOICE,
    VOICE,
    _audit,
    _authenticate,
    _db,
    _enroll,
    _fusion_files,
    client,
)

USER = "workflow-user"
BUILDING = "national_data_center"
POSES = ("front", "left", "right", "up", "down")


def _register_face_and_voice(client):
    response = client.post(
        "/enroll",
        data={"user_id": USER, "modality": "face", "application_id": APPLICATION_ID},
        files={f"pose_{p}": (f"{p}.png", FACE, "image/png") for p in POSES},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["poses_valid"] == 5
    _enroll(client, USER, "voice", VOICE)


def _face_voice(client, face=FACE, voice=VOICE):
    return _authenticate(client, USER, BUILDING, face=face, voice=voice)


def test_complete_face_voice_workflow_grants_and_is_audited(client):
    from backend.database import crud

    _register_face_and_voice(client)

    # template storage: 4 sets x 2 modalities, one ACTIVE set, only protected bytes (+ key/set metadata) per row
    rows = crud.get_set_rows(_db(), USER, APPLICATION_ID)
    assert sorted({r.modality for r in rows}) == ["face", "voice"]
    assert len({r.template_set_version for r in rows}) == 4
    assert {r.template_set_version for r in rows if r.is_active} == {1}
    for r in rows:
        assert r.output_bits == 256 and len(bytes(r.protected_template)) == 32  # 256 bits packed
        assert r.key_version >= 1

    body = _face_voice(client).json()
    assert body["status"] == "ACCESS_GRANTED" and body["authenticated"] is True
    assert sorted(body["matched_modalities"]) == ["face", "voice"]
    assert body["template_set_version"] == 1

    entry = _audit(client, USER)[0]
    assert entry["authentication_state"] == "ACCESS_GRANTED"
    assert entry["fusion_policy"] == "ALL_REQUIRED"
    assert sorted(entry["submitted_modalities"]) == ["face", "voice"]
    assert set(entry["key_versions"]) == {"face", "voice"}


@pytest.mark.parametrize("wrong", ["face", "voice"])
def test_a_wrong_face_or_wrong_voice_is_denied_under_all_required(client, wrong):
    _register_face_and_voice(client)
    kwargs = {"face": OTHER_FACE} if wrong == "face" else {"voice": OTHER_VOICE}
    body = _face_voice(client, **kwargs).json()
    assert body["status"] == "ACCESS_DENIED" and body["authenticated"] is False
    assert wrong not in body["matched_modalities"]
    assert _audit(client, USER)[0]["authentication_state"] == "ACCESS_DENIED"


def test_face_failure_and_voice_failure_are_clean_rejections(client):
    _register_face_and_voice(client)
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)  # the stub's "no face"
    import cv2

    ok, png = cv2.imencode(".png", flat)
    response = client.post("/enroll/face/check-pose", data={"pose": "front"}, files={"image": ("f.png", png.tobytes(), "image/png")})
    assert response.status_code == 200 and response.json()["status"] == "NO_FACE" and response.json()["valid"] is False
    # a corrupt WAV: never compared, never a server error, never granted - an audited CAPTURE_ERROR asking for a retry
    response = _authenticate(client, USER, BUILDING, face=FACE, voice=b"RIFF\x00\x00not-a-wave")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ACCESS_DENIED" and body["attempt_assessment"]["modality_statuses"]["voice"] == "CAPTURE_ERROR"
    assert body["attempt_assessment"]["status"] == "RETRY_REQUESTED" and "template_rotation" not in body


def test_revoked_template_fails_and_the_new_set_authenticates(client):
    from backend.database import crud

    _register_face_and_voice(client)
    old = {r.modality: bytes(r.protected_template) for r in crud.get_active_set_rows(_db(), USER, APPLICATION_ID)}
    revoke = client.post("/revoke-template", data={"user_id": USER, "application_id": APPLICATION_ID},
                         files=_fusion_files(face=FACE, voice=VOICE))
    assert revoke.status_code == 200, revoke.text

    body = _face_voice(client).json()
    assert body["status"] == "ACCESS_GRANTED" and body["template_set_version"] == 2
    new = {r.modality: bytes(r.protected_template) for r in crud.get_active_set_rows(_db(), USER, APPLICATION_ID)}
    assert all(new[m] != old[m] for m in old)  # new key -> a different template of the same biometric

    db = _db()  # put the REVOKED templates back into the ACTIVE slots: they no longer match under the active key
    for row in crud.get_active_set_rows(db, USER, APPLICATION_ID):
        row.protected_template = old[row.modality]
    db.commit()
    assert _face_voice(client).json()["status"] == "ACCESS_DENIED"


def test_invalid_upload_and_malformed_input_are_rejected_without_a_server_error(client):
    _register_face_and_voice(client)
    cases = [
        {"face_image": ("f.txt", b"hello", "text/plain"), "voice_audio": ("v.wav", VOICE, "audio/wav")},  # wrong type
        {"face_image": ("f.png", b"\x89PNG\r\n\x1a\nbroken", "image/png"), "voice_audio": ("v.wav", VOICE, "audio/wav")},  # corrupt
    ]
    expected = [415, 200]  # wrong type: a request error; corrupt image: an audited CAPTURE_ERROR (denied, retry)
    for files, code in zip(cases, expected):
        response = client.post("/authenticate/fusion", data={"user_id": USER, "building_id": BUILDING, "application_id": APPLICATION_ID},
                               files=files)
        assert response.status_code == code, response.text
        if code == 200:
            body = response.json()
            assert body["status"] == "ACCESS_DENIED" and body["attempt_assessment"]["modality_statuses"]["face"] == "CAPTURE_ERROR"
    response = client.post("/enroll", data={"user_id": USER, "modality": "retina"}, files={"image": ("f.png", FACE, "image/png")})
    assert 400 <= response.status_code < 500


@pytest.mark.parametrize("error,status", [("AlignmentFailed", "ALIGNMENT_FAILED"), ("LandmarkFailure", "LANDMARK_FAILURE")])
def test_alignment_and_landmark_failures_reach_the_client_as_distinct_statuses(client, monkeypatch, error, status):
    """Aligned-mode pipeline behind the real /enroll/face/check-pose route and ModalityService."""
    import preprocessing.face as pf
    from backend.services import get_service_for_modality
    from embeddings.pipelines import FacePipeline

    pipeline = FacePipeline(checkpoint_path=None, alignment="similarity")

    def boom(image):
        raise getattr(pf, error)("synthetic failure")

    monkeypatch.setattr(pipeline._preprocessor, "detect_and_align", boom)
    monkeypatch.setattr(get_service_for_modality("face"), "pipeline", pipeline)
    response = client.post("/enroll/face/check-pose", data={"pose": "front"}, files={"image": ("f.png", FACE, "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == status and body["valid"] is False and body["detail"]
