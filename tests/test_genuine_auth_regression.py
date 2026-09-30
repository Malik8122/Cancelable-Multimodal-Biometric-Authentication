"""Genuine-user authentication after the multi-template / template-set migration.

Debug findings (see the root-cause report): the matching path was NOT regressed - preprocessing, embeddings,
HKDF/BioHash, fusion and the threshold loader are unchanged since the pre-migration commit and produce identical
similarities. What was wrong: a failed voice consistency check left an ACTIVE voice template behind. These tests pin
both: a genuine user is granted right after enrolling, a wrong user / revoked template / STANDBY template can never
authenticate, and enrollment with a consistency check is all-or-nothing.

Most tests use the deterministic embedder stubs from tests/test_flexible_auth.py (same sample -> same embedding,
different sample -> unrelated embedding) over the real services, database, fusion and HTTP layer. The last test runs
the REAL face and voice models.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pytest

from tests.voice_signals import speech_envelope
from tests.test_flexible_auth import (  # noqa: F401  (client is a fixture)
    APPLICATION_ID,
    BUILDING,
    FACE,
    OTHER_FACE,
    OTHER_VOICE,
    VOICE,
    _audit,
    _authenticate,
    _db,
    _enroll,
    _enroll_user,
    _fusion_files,
    client,
)

USER = "genuine-user"


def _enroll_both(client, user=USER):
    _enroll_user(client, user, face=FACE, voice=VOICE)


# ------------------------------------------------------------------ 1. same-user immediate enroll -> authenticate


def test_a_genuine_user_is_granted_immediately_after_enrolling_both(client):
    _enroll_both(client)
    body = _authenticate(client, USER, BUILDING, face=FACE, voice=VOICE).json()
    assert body["status"] == "ACCESS_GRANTED" and body["authenticated"] is True
    assert body["matched_modalities"] == ["face", "voice"]
    assert body["fusion_similarity"] == pytest.approx(1.0)
    # fusion received both successful modality results
    assert sorted(body["results"]) == ["face", "voice"]
    assert all(r["authenticated"] for r in body["results"].values())
    assert body["template_set_version"] == 1 and body["key_version"] == 1


def test_enrollment_leaves_exactly_one_active_template_per_modality_in_one_set(client):
    from backend.database import crud

    _enroll_both(client)
    rows = crud.get_set_rows(_db(), USER, APPLICATION_ID)
    for modality in ("face", "voice"):
        active = [r for r in rows if r.modality == modality and r.is_active]
        assert len(active) == 1 and active[0].template_status == "ACTIVE"
    assert {r.template_set_version for r in rows if r.is_active} == {1}
    # template_version is the BioHash FORMAT version (1 everywhere); the pool position is template_set_version
    assert {r.template_version for r in rows} == {1}
    assert sorted({r.template_set_version for r in rows}) == [1, 2, 3, 4]
    # key_version == set version after a clean enrollment (no mismatch between the two)
    assert all(r.key_version == r.template_set_version for r in rows)


# ------------------------------------------------------------------ 2. wrong user still denied


def test_a_wrong_biometric_is_denied_in_every_modality(client):
    _enroll_both(client)
    for kwargs in (
        dict(face=OTHER_FACE, voice=VOICE),
        dict(face=FACE, voice=OTHER_VOICE),
    ):
        body = _authenticate(client, USER, BUILDING, **kwargs).json()
        assert body["status"] == "ACCESS_DENIED" and body["authenticated"] is False, kwargs.keys()
    # every modality wrong at once
    body = _authenticate(client, USER, BUILDING, face=OTHER_FACE, voice=OTHER_VOICE).json()
    assert body["status"] == "ACCESS_DENIED" and body["matched_modalities"] == []


def test_another_users_biometrics_do_not_authenticate_this_user(client):
    """Templates are keyed per user: user B presenting user A's samples (or A's under B's id) is denied."""
    _enroll_both(client, "user-a")
    _enroll_user(client, "user-b", face=OTHER_FACE, voice=OTHER_VOICE)
    assert _authenticate(client, "user-b", BUILDING, face=FACE, voice=VOICE).json()["status"] == "ACCESS_DENIED"
    assert _authenticate(client, "user-a", BUILDING, face=OTHER_FACE, voice=OTHER_VOICE).json()["status"] == "ACCESS_DENIED"
    assert _authenticate(client, "user-a", BUILDING, face=FACE, voice=VOICE).json()["status"] == "ACCESS_GRANTED"


# ------------------------------------------------------------------ 3. revoked template still denied


def _revoke(client):
    return client.post(
        "/revoke-template",
        data={"user_id": USER, "application_id": APPLICATION_ID},
        files=_fusion_files(face=FACE, voice=VOICE),
    )


def test_a_revoked_template_set_never_authenticates(client):
    from backend.database import crud

    _enroll_both(client)
    revoked_bytes = {r.modality: bytes(r.protected_template) for r in crud.get_active_set_rows(_db(), USER, APPLICATION_ID)}
    assert _revoke(client).status_code == 200

    # (a) the revoked rows are out of service
    rows = crud.get_set_rows(_db(), USER, APPLICATION_ID)
    assert all(not r.is_active and r.template_status == "REVOKED" for r in rows if r.template_set_version == 1)
    # (b) authentication now compares against set 2 - the same person still passes...
    assert _authenticate(client, USER, BUILDING, face=FACE, voice=VOICE).json()["template_set_version"] == 2
    # (c) ...but the REVOKED templates are unusable: put them in the ACTIVE slots and the genuine user is denied,
    # because the candidate is regenerated under the active key, not the revoked one.
    db = _db()
    for row in crud.get_active_set_rows(db, USER, APPLICATION_ID):
        row.protected_template = revoked_bytes[row.modality]
    db.commit()
    body = _authenticate(client, USER, BUILDING, face=FACE, voice=VOICE).json()
    assert body["status"] == "ACCESS_DENIED" and body["matched_modalities"] == []


def test_a_user_whose_only_set_is_revoked_cannot_authenticate(client):
    from backend.database import crud

    _enroll_both(client)
    db = _db()
    for row in crud.get_set_rows(db, USER, APPLICATION_ID):
        row.is_active, row.template_status, row.template_set_status = False, "REVOKED", "REVOKED"
    db.commit()
    response = _authenticate(client, USER, BUILDING, face=FACE, voice=VOICE)
    assert response.status_code == 409 and response.json()["status"] == "ENROLLMENT_REQUIRED"  # nothing active = not enrolled


# ------------------------------------------------------------------ 4. STANDBY templates are never used


def test_standby_templates_are_never_read_or_compared(client):
    from backend.database import crud

    _enroll_both(client)
    db = _db()
    rng = np.random.default_rng(0)
    for row in crud.get_set_rows(db, USER, APPLICATION_ID):
        if row.template_status == "STANDBY":
            row.protected_template = rng.bytes(len(row.protected_template))  # garbage: would break any comparison
    db.commit()

    # The row each modality is compared with is selected by base_service._active_row from one snapshot of the pool
    # (see ModalityService.authenticate_embedding); spy on that selection point.
    import backend.services.base_service as base_service

    seen = []
    original = base_service._active_row

    def spy(*args, **kwargs):
        row = original(*args, **kwargs)
        if row is not None:
            seen.append((row.modality, row.template_set_version, row.template_status, row.is_active))
        return row

    base_service._active_row = spy
    try:
        body = _authenticate(client, USER, BUILDING, face=FACE, voice=VOICE).json()
    finally:
        base_service._active_row = original
    assert body["status"] == "ACCESS_GRANTED" and body["fusion_similarity"] == pytest.approx(1.0)
    assert seen and all(s[1:] == (1, "ACTIVE", True) for s in seen)  # only ever the ACTIVE set-1 rows


# ------------------------------------------------------------------ 5. voice enrollment is all-or-nothing


def _enroll_voice(client, user, first, second=None):
    files = {"image": ("v.wav", first, "audio/wav")}
    if second is not None:
        files["confirm_image"] = ("v2.wav", second, "audio/wav")
    return client.post("/enroll", data={"user_id": user, "modality": "voice", "application_id": APPLICATION_ID}, files=files)


def _voice_state(client, user):
    from backend.database import crud

    rows = [r for r in crud.get_set_rows(_db(), user, APPLICATION_ID) if r.modality == "voice"]
    enrolled = client.get(f"/user/{user}/enrollment-status", params={"application_id": APPLICATION_ID}).json()["modalities"]["voice"]
    return enrolled, len(rows), sum(1 for r in rows if r.is_active)


def test_a_consistent_pair_of_voice_recordings_enrolls(client):
    response = _enroll_voice(client, "v-user", VOICE, VOICE)
    assert response.status_code == 200 and response.json()["templates_created"] == 4
    assert _voice_state(client, "v-user") == (True, 4, 1)


def test_an_inconsistent_pair_enrolls_nothing_and_cannot_authenticate(client):
    response = _enroll_voice(client, "v-user", VOICE, OTHER_VOICE)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "ENROLLMENT_INCONSISTENT" and "Nothing was enrolled" in body["detail"]
    # no ACTIVE - and no STANDBY - template exists; the modality is NOT ENROLLED
    assert _voice_state(client, "v-user") == (False, 0, 0)
    # a partially enrolled voice template cannot be used: there is nothing to authenticate against
    single = client.post(
        "/authenticate",
        data={"user_id": "v-user", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", VOICE, "audio/wav")},
    ).json()
    assert single["status"] == "ENROLLMENT_REQUIRED" and single["missing_modalities"] == ["voice"]  # not a denial: not enrolled
    profile = client.get("/user/v-user/enrollment-status", params={"application_id": APPLICATION_ID}).json()
    assert profile["modalities"]["voice"] is False and profile["statuses"]["voice"] == "RETRY_REQUIRED"


def test_a_rejected_reenrollment_keeps_the_previous_enrollment(client):
    assert _enroll_voice(client, "v-user", VOICE, VOICE).status_code == 200
    assert _enroll_voice(client, "v-user", OTHER_VOICE, VOICE).status_code == 422  # inconsistent replacement
    assert _voice_state(client, "v-user") == (True, 4, 1)  # untouched: same 4 rows, still one ACTIVE
    ok = client.post(
        "/authenticate",
        data={"user_id": "v-user", "modality": "voice", "application_id": APPLICATION_ID},
        files={"image": ("v.wav", VOICE, "audio/wav")},
    ).json()
    assert ok["status"] == "ACCESS_GRANTED"


def test_a_failed_consistency_check_does_not_touch_other_modalities(client):
    _enroll_user(client, "v-user", face=FACE)
    assert _enroll_voice(client, "v-user", VOICE, OTHER_VOICE).status_code == 422
    status = client.get("/user/v-user/enrollment-status", params={"application_id": APPLICATION_ID}).json()["modalities"]
    assert status == {"face": True, "voice": False, "hand": False}


def test_single_sample_enrollment_is_unchanged(client):
    assert _enroll_voice(client, "v-user", VOICE).status_code == 200
    assert _voice_state(client, "v-user") == (True, 4, 1)


# ------------------------------------------------------------------ 6. debug stage logging is debug-only and never leaks embeddings


def test_debug_stage_logging_only_with_debug_scores_and_never_embeddings(client, caplog, monkeypatch):
    _enroll_both(client)
    with caplog.at_level(logging.INFO):
        _authenticate(client, USER, BUILDING, face=FACE, voice=VOICE)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert text.count("AUTH-DEBUG") == 2 and "FUSION-DEBUG" in text
    for needle in ("embedding_dim=", "biohash_bits=256", "hamming=", "threshold=", "entering_fusion=['face', 'voice']",
                   "rejected_before_fusion=[]", "fusion_distance=", "state=ACCESS_GRANTED"):
        assert needle in text, needle

    caplog.clear()
    monkeypatch.setenv("DEBUG_SCORES", "false")
    from backend.config import get_settings

    get_settings.cache_clear()
    from backend.services.face_service import get_face_service  # noqa: F401  (stubbed by the client fixture)

    with caplog.at_level(logging.INFO):
        body = _authenticate(client, USER, BUILDING, face=FACE, voice=VOICE).json()
    assert "AUTH-DEBUG" not in "\n".join(r.getMessage() for r in caplog.records)
    assert {"results", "score", "threshold", "distance"}.isdisjoint(body)  # no per-modality values; fusion_distance is a fusion value


# ------------------------------------------------------------------ 7. the REAL models, end to end


def test_real_models_genuine_user_is_granted_and_a_wrong_voice_is_denied(tmp_path, monkeypatch):
    """Face (real portrait) and voice (real ECAPA model): enroll, then authenticate
    immediately with the same files, with a mild recapture, and with a wrong voice."""
    pytest.importorskip("facenet_pytorch")
    pytest.importorskip("torchvision")
    matplotlib = pytest.importorskip("matplotlib")
    import cv2
    import io

    from fastapi.testclient import TestClient
    from scipy.io import wavfile

    portrait = pathlib.Path(matplotlib.__file__).parent / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    if not portrait.exists():
        pytest.skip("no real face image available")
    image = cv2.imread(str(portrait))
    face = cv2.imencode(".jpg", image)[1].tobytes()
    face_recapture = cv2.imencode(".jpg", cv2.convertScaleAbs(image, alpha=1.04, beta=6), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    def tone(freq):
        t = np.linspace(0, 3.0, 48000, endpoint=False)
        wave = 0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)) * speech_envelope(t)
        buffer = io.BytesIO()
        wavfile.write(buffer, 16000, (wave * 32767).astype(np.int16))
        return buffer.getvalue()

    def chirp():
        t = np.linspace(0, 3.0, 48000, endpoint=False)
        buffer = io.BytesIO()
        wavfile.write(buffer, 16000, (0.3 * speech_envelope(t) * np.sin(2 * np.pi * (200 + 400 * t) * t) * 32767).astype(np.int16))
        return buffer.getvalue()

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'real.db'}")
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()
    from backend.main import app

    def files(f, v):
        return {"face_image": ("f.jpg", f, "image/jpeg"), "voice_audio": ("v.wav", v, "audio/wav")}

    with TestClient(app) as c:
        for modality, sample, ctype in (("face", face, "image/jpeg"), ("voice", tone(220), "audio/wav")):
            r = c.post("/enroll", data={"user_id": "real", "modality": modality, "application_id": APPLICATION_ID}, files={"image": ("s", sample, ctype)})
            assert r.status_code == 200, r.text
        data = {"user_id": "real", "application_id": APPLICATION_ID, "building_id": BUILDING}
        same = c.post("/authenticate/fusion", data=data, files=files(face, tone(220))).json()
        assert same["status"] == "ACCESS_GRANTED" and same["fusion_similarity"] == pytest.approx(1.0)
        assert sorted(same["results"]) == ["face", "voice"] and all(v["authenticated"] for v in same["results"].values())
        recapture = c.post("/authenticate/fusion", data=data, files=files(face_recapture, tone(220))).json()
        assert recapture["status"] == "ACCESS_GRANTED", recapture["results"]
        # A clearly different sound. (Not another pure tone: the real ECAPA model scores any two tones ~0.99, so a tone is not a
        # meaningful "other speaker" - a chirp, which it does separate, is.)
        wrong = c.post("/authenticate/fusion", data=data, files=files(face, chirp())).json()
        assert wrong["status"] == "ACCESS_DENIED" and wrong["matched_modalities"] == ["face"]
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()
