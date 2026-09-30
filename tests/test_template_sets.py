"""Template-set architecture (docs/MULTI_TEMPLATE_ARCHITECTURE.md).

A user's credential is a pool of template SETS; each set holds one template per
enrolled modality; exactly one set is ACTIVE; revoke / activate / generate move
whole sets and require biometric authorization against the ACTIVE set.

Offline: face and voice with deterministic stub embedders (tests/test_flexible_auth.py - face detection cannot
run on synthetic images), each test on its own SQLite file.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.test_flexible_auth import FACE, OTHER_FACE, VOICE, _stub_getter

APPLICATION_ID = "capstone-demo"
USER = "SET-U1"

PUBLIC_KEYS = {
    "status", "authentication_state", "fusion_distance", "active_template_set",
    "user_id", "authenticated", "fusion_similarity", "fusion_threshold", "fusion_policy", "matched_modalities",
    "modalities_used", "template_set_version", "key_version", "authentication_time_ms",
    "display_name",  # the user's name - present because these requests are granted
}
FORBIDDEN_KEYS = {
    "score", "threshold", "distance", "results", "fused_score", "failed_modalities", "modality",
    "template_version", "template_versions", "key_versions", "face_similarity", "voice_similarity",
}


_CONTENT_TYPE = {"face": ("s.png", "image/png"), "voice": ("s.wav", "audio/wav")}


def _reset_caches():
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory
    from backend.services import face_service, voice_service

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    face_service.get_face_service.cache_clear()  # the cached services hold the Settings they were built with
    voice_service.get_voice_service.cache_clear()


@pytest.fixture(autouse=True)
def stub_embedders(monkeypatch):
    import backend.services.face_service as face_service
    import backend.services.voice_service as voice_service

    monkeypatch.setattr(face_service, "get_face_service", _stub_getter("face", 512))
    monkeypatch.setattr(voice_service, "get_voice_service", _stub_getter("voice", 192))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'sets.db'}")
    _reset_caches()
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
    _reset_caches()


def _db():
    from backend.database.session import get_session_factory

    return get_session_factory()()


def _enroll(client, sample, modality="face", user=USER):
    filename, content_type = _CONTENT_TYPE[modality]
    return client.post(
        "/enroll",
        data={"user_id": user, "modality": modality, "application_id": APPLICATION_ID},
        files={"image": (filename, sample, content_type)},
    )


def _enroll_both(client, user=USER):
    assert _enroll(client, FACE, "face", user).status_code == 200
    assert _enroll(client, VOICE, "voice", user).status_code == 200


def _auth(client, sample, modality="face", user=USER):
    filename, content_type = _CONTENT_TYPE[modality]
    return client.post(
        "/authenticate",
        data={"user_id": user, "modality": modality, "application_id": APPLICATION_ID},
        files={"image": (filename, sample, content_type)},
    )


def _captures(face=FACE, voice=VOICE):
    """Authorization captures for a user enrolled in both modalities."""
    files = {}
    if face is not None:
        files["face_image"] = ("f.png", face, "image/png")
    if voice is not None:
        files["voice_audio"] = ("v.wav", voice, "audio/wav")
    return files


def _revoke(client, user=USER, files=None):
    return client.post(
        "/revoke-template",
        data={"user_id": user, "application_id": APPLICATION_ID},
        files=_captures() if files is None else files,
    )


def _sets(client, user=USER):
    response = client.get(f"/templates/{user}", params={"application_id": APPLICATION_ID})
    assert response.status_code == 200, response.text
    return response.json()


def _statuses(client, user=USER):
    return {s["template_set_version"]: s["status"] for s in _sets(client, user)["sets"]}


# --- enrollment ------------------------------------------------------------------------------


def test_enrollment_creates_four_sets_and_each_contains_every_enrolled_modality(client):
    first = _enroll(client, FACE, "face").json()
    assert first["templates_created"] == 4
    assert first["active_template_set_version"] == 1
    assert first["standby_template_set_versions"] == [2, 3, 4]
    assert "protected_template" not in first

    second = _enroll(client, VOICE, "voice").json()  # joins the same four sets
    assert second["templates_created"] == 4 and second["active_template_set_version"] == 1

    body = _sets(client)
    assert body["active_template_set_version"] == 1 and body["standby_count"] == 3 and body["pool_size"] == 4
    assert [s["status"] for s in body["sets"]] == ["ACTIVE", "STANDBY", "STANDBY", "STANDBY"]
    for s in body["sets"]:
        assert s["modalities"] == ["face", "voice"]
        assert set(s["key_versions"]) == {"face", "voice"}
    assert "protected_template" not in str(body)


def test_templates_in_different_sets_are_cryptographically_different(client):
    from backend.database import crud

    _enroll_both(client)
    rows = crud.get_set_rows(_db(), USER, APPLICATION_ID)
    assert len(rows) == 8
    assert len({bytes(r.protected_template) for r in rows}) == 8
    for modality in ("face", "voice"):
        assert sorted(r.key_version for r in rows if r.modality == modality) == [1, 2, 3, 4]
    assert len({r.template_group_id for r in rows}) == 4  # one group id per set, shared by its modalities


def test_set_pool_size_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'p.db'}")
    monkeypatch.setenv("TEMPLATE_POOL_SIZE", "2")
    _reset_caches()
    from backend.main import app

    with TestClient(app) as test_client:
        body = _enroll(test_client, FACE).json()
    _reset_caches()
    assert body["templates_created"] == 2 and body["standby_template_set_versions"] == [2]


def test_database_allows_only_one_active_row_per_modality_and_one_live_row_per_set_modality(client):
    from sqlalchemy.exc import IntegrityError

    from backend.database import crud

    _enroll_both(client)
    db = _db()
    standby = [r for r in crud.get_set_rows(db, USER, APPLICATION_ID) if r.template_set_version == 2 and r.modality == "face"][0]
    standby.is_active = True
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    row = [r for r in crud.get_set_rows(db, USER, APPLICATION_ID) if r.template_set_version == 3 and r.modality == "face"][0]
    db.add(
        type(row)(
            user_id=USER, application_id=APPLICATION_ID, modality="face", template_version=1, key_version=99,
            output_bits=256, protected_template=b"\x01" * 32, is_active=False, template_status="STANDBY",
            template_set_version=3, template_set_status="STANDBY", template_index=3,
        )
    )
    with pytest.raises(IntegrityError):  # a second live face template in set 3
        db.commit()


# --- authentication --------------------------------------------------------------------------


def test_authentication_matches_only_the_active_set(client):
    from backend.database import crud

    _enroll_both(client)
    db = _db()
    rng = np.random.default_rng(0)
    for row in crud.get_set_rows(db, USER, APPLICATION_ID):
        if row.template_set_version != 1:  # overwrite every STANDBY template
            row.protected_template = rng.bytes(len(row.protected_template))
    db.commit()

    fused = client.post(
        "/authenticate/fusion",
        data={"user_id": USER, "application_id": APPLICATION_ID},
        files={"voice_audio": ("v.wav", VOICE, "audio/wav")},
    ).json()
    assert fused["authenticated"] is True and fused["template_set_version"] == 1
    body = _auth(client, FACE).json()
    assert body["authenticated"] is True and body["fusion_similarity"] == pytest.approx(1.0)
    assert body["template_set_version"] == 1 and body["key_version"] == 1


def test_wrong_biometric_is_rejected(client):
    _enroll(client, FACE)
    assert _auth(client, OTHER_FACE).json()["authenticated"] is False


def test_authentication_fails_closed_if_active_rows_come_from_different_sets(tmp_path, monkeypatch):
    """Corrupt the ACTIVE set on purpose: voice's active row moves to set 2. Authentication
    must refuse (500, never a grant) rather than compare against a mixture of sets."""
    from backend.database import crud

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'm.db'}")
    _reset_caches()
    from backend.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        _enroll_both(test_client)
        db = _db()
        rows = crud.get_set_rows(db, USER, APPLICATION_ID)
        fp1 = next(r for r in rows if r.modality == "voice" and r.template_set_version == 1)
        fp2 = next(r for r in rows if r.modality == "voice" and r.template_set_version == 2)
        fp1.template_status, fp1.is_active = "STANDBY", False
        db.flush()
        fp2.template_status, fp2.is_active = "ACTIVE", True
        db.commit()
        response = _auth(test_client, FACE, "face")
    _reset_caches()
    assert response.status_code == 500 and "authenticated" not in response.json()


# --- revocation (whole set, authorized) -------------------------------------------------------


def test_revocation_moves_every_modality_to_the_next_set_together(client):
    _enroll_both(client)
    body = _revoke(client).json()
    assert body == {
        "success": True, "user_id": USER, "revoked_template_set_version": 1,
        "new_active_template_set_version": 2, "remaining_standby_template_sets": 2,
    }
    assert _statuses(client) == {1: "REVOKED", 2: "ACTIVE", 3: "STANDBY", 4: "STANDBY"}
    revoked = _sets(client)["sets"][0]
    assert revoked["revoked_at"] and _sets(client)["sets"][1]["activated_at"]

    # Both modalities now authenticate under set 2 / key 2 - the same person, no re-enrollment.
    for image, modality in ((FACE, "face"), (VOICE, "voice")):
        result = _auth(client, image, modality).json()
        assert result["authenticated"] is True
        assert result["template_set_version"] == 2 and result["key_version"] == 2


def test_revoked_set_rows_never_stay_active(client):
    from backend.database import crud

    _enroll_both(client)
    _revoke(client)
    rows = crud.get_set_rows(_db(), USER, APPLICATION_ID)
    assert {r.template_set_version for r in rows if r.is_active} == {2}
    assert all(r.template_status == "REVOKED" and not r.is_active for r in rows if r.template_set_version == 1)
    assert {r.modality for r in rows if r.is_active} == {"face", "voice"}


def test_revocation_requires_biometric_authorization(client):
    _enroll_both(client)
    # no captures / a missing modality / a wrong biometric / an extra modality: all 403, nothing changes
    assert _revoke(client, files={}).status_code == 403
    assert _revoke(client, files=_captures(voice=None)).status_code == 403
    assert _revoke(client, files=_captures(face=OTHER_FACE)).status_code == 403
    assert _statuses(client) == {1: "ACTIVE", 2: "STANDBY", 3: "STANDBY", 4: "STANDBY"}
    assert _revoke(client).status_code == 200


def test_failed_authorization_attempts_cannot_exhaust_the_pool_and_are_audited(client):
    _enroll_both(client)
    for _ in range(5):
        assert _revoke(client, files=_captures(face=OTHER_FACE)).status_code == 403
    assert _statuses(client)[1] == "ACTIVE"
    for _ in range(3):
        assert _revoke(client).status_code == 200  # all three real revocations are still available

    from backend.database import audit

    rows = audit.get_history_for_user(_db(), USER, limit=50)
    management = [r for r in rows if r.fusion_policy == "TEMPLATE_MANAGEMENT"]
    assert len(management) == 8 and sum(1 for r in management if not r.authenticated) == 5
    assert all(r.template_set_version is not None and r.template_set_status == "ACTIVE" for r in management)


def test_pool_exhaustion_returns_409_and_leaves_the_active_set_usable(client):
    _enroll_both(client)
    for _ in range(3):
        assert _revoke(client).status_code == 200
    response = _revoke(client)
    assert response.status_code == 409
    assert response.json()["detail"] == "Template set pool exhausted. Re-enrollment required."
    assert _statuses(client) == {1: "REVOKED", 2: "REVOKED", 3: "REVOKED", 4: "ACTIVE"}
    assert _auth(client, FACE).json()["authenticated"] is True


def test_revoke_unknown_user_is_404(client):
    assert _revoke(client, user="ghost").status_code == 404


# --- activate / generate ---------------------------------------------------------------------


def test_activate_promotes_a_specific_standby_set_after_authorization(client):
    _enroll_both(client)
    url = f"/templates/{USER}/activate/3"
    assert client.post(url, data={"application_id": APPLICATION_ID}, files=_captures(face=OTHER_FACE)).status_code == 403
    response = client.post(url, data={"application_id": APPLICATION_ID}, files=_captures())
    assert response.status_code == 200
    assert response.json()["new_active_template_set_version"] == 3
    assert response.json()["previous_active_template_set_version"] == 1
    assert _statuses(client) == {1: "REVOKED", 2: "STANDBY", 3: "ACTIVE", 4: "STANDBY"}
    assert _auth(client, FACE).json()["template_set_version"] == 3
    # only STANDBY sets can be activated
    assert client.post(f"/templates/{USER}/activate/1", data={"application_id": APPLICATION_ID}, files=_captures()).status_code == 404


def test_generate_adds_a_complete_standby_set_only_when_there_is_room(client):
    _enroll_both(client)

    def generate(files=None):
        return client.post(
            f"/templates/{USER}/generate", data={"application_id": APPLICATION_ID}, files=_captures() if files is None else files
        )

    assert generate().status_code == 409  # pool already holds 4 live sets
    for _ in range(3):
        _revoke(client)
    assert generate(_captures(face=OTHER_FACE)).status_code == 403  # a stranger cannot seed a set

    response = generate()
    assert response.status_code == 200
    assert response.json()["new_template_set_version"] == 5 and response.json()["standby_template_set_versions"] == [5]
    sets = {s["template_set_version"]: s for s in _sets(client)["sets"]}
    assert sets[5]["status"] == "STANDBY" and sets[5]["modalities"] == ["face", "voice"]
    assert sets[5]["key_versions"] == {"face": 5, "voice": 5}  # keys continue, never reused
    assert _revoke(client).status_code == 200  # usable again: set 5 becomes ACTIVE
    assert _auth(client, FACE).json()["template_set_version"] == 5


# --- enrolling modalities into existing sets -------------------------------------------------


def test_a_modality_enrolled_later_joins_only_the_live_sets(client):
    assert _enroll(client, FACE, "face").status_code == 200
    assert _revoke(client, files=_captures(voice=None)).status_code == 200  # face-only user: set 1 -> 2
    joined = _enroll(client, VOICE, "voice").json()
    assert joined["templates_created"] == 3 and joined["active_template_set_version"] == 2
    sets = {s["template_set_version"]: s for s in _sets(client)["sets"]}
    assert sets[1]["status"] == "REVOKED" and sets[1]["modalities"] == ["face"]
    assert all(sets[v]["modalities"] == ["face", "voice"] for v in (2, 3, 4))
    assert _auth(client, VOICE, "voice").json()["template_set_version"] == 2


def test_reenrolling_a_modality_replaces_its_templates_inside_the_same_sets(client):
    from backend.database import crud

    _enroll_both(client)
    again = _enroll(client, FACE, "face").json()
    assert again["templates_created"] == 4 and again["active_template_set_version"] == 1
    assert _statuses(client) == {1: "ACTIVE", 2: "STANDBY", 3: "STANDBY", 4: "STANDBY"}  # no extra sets
    rows = crud.get_set_rows(_db(), USER, APPLICATION_ID)
    live_face = sorted(r.key_version for r in rows if r.modality == "face" and r.template_status != "REVOKED")
    assert live_face == [5, 6, 7, 8]  # fresh key versions, never reused
    assert _auth(client, FACE).json()["authenticated"] is True


# --- public API vs internal scores -----------------------------------------------------------


def test_public_responses_expose_one_decision_one_fusion_similarity_and_one_set_version(tmp_path, monkeypatch):
    monkeypatch.setenv("DEBUG_SCORES", "false")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'pub.db'}")
    _reset_caches()
    from backend.main import app

    with TestClient(app) as test_client:
        _enroll(test_client, VOICE, "voice")
        single = _auth(test_client, VOICE, "voice").json()
        fused = test_client.post(
            "/authenticate/fusion",
            data={"user_id": USER, "application_id": APPLICATION_ID},
            files={"voice_audio": ("v.wav", VOICE, "audio/wav")},
        ).json()
        audit = test_client.get(f"/audit/{USER}").json()["entries"][0]
        stored = _db().execute(
            __import__("sqlalchemy").text(
                "SELECT voice_similarity, fusion_similarity, template_set_version, template_set_status FROM audit_logs"
            )
        ).all()
    _reset_caches()

    for body in (single, fused):
        assert set(body) == PUBLIC_KEYS, set(body) ^ PUBLIC_KEYS
        assert not (FORBIDDEN_KEYS & set(body))
        assert body["fusion_policy"] == "ALL_REQUIRED" and body["matched_modalities"] == ["voice"]
        assert body["template_set_version"] == 1
    # hidden from the audit API, kept in the table together with the set version/status
    assert audit["similarity_scores"] is None and audit["voice_similarity"] is None
    assert audit["template_set_version"] == 1 and audit["template_set_status"] == "ACTIVE"
    assert len(stored) == 2
    assert all(r[0] == pytest.approx(1.0) and r[1] == pytest.approx(1.0) and r[2] == 1 and r[3] == "ACTIVE" for r in stored)


def test_debug_mode_adds_the_internal_scores(client):
    _enroll(client, VOICE, "voice")
    body = _auth(client, VOICE, "voice").json()
    assert body["score"] == pytest.approx(1.0) and "results" in body and body["fusion_distance"] == pytest.approx(0.0)


def test_a_submitted_modality_that_is_not_enrolled_is_enrollment_required(client):
    """Face is enrolled, voice is not. Submitting both -> 409 (voice is not authenticated, nothing is
    evaluated); submitting just the enrolled face -> granted, fused over that one modality."""
    _enroll(client, FACE, "face")
    both = client.post("/authenticate/fusion", data={"user_id": USER, "application_id": APPLICATION_ID}, files=_captures())
    assert both.status_code == 409 and both.json()["status"] == "ENROLLMENT_REQUIRED"
    assert both.json()["missing_modalities"] == ["voice"]
    only = client.post(
        "/authenticate/fusion", data={"user_id": USER, "application_id": APPLICATION_ID}, files=_captures(voice=None)
    ).json()
    assert only["authentication_state"] == "ACCESS_GRANTED" and only["modalities_used"] == ["face"]


# --- security validation ---------------------------------------------------------------------


def _validate(rows, stored=None, key_version=1, pool_size=4):
    from backend.security_validation import validate_template_pool
    from template_protection.hkdf_keys import derive_key

    stored = stored or next(r for r in rows if r.is_active and r.modality == "face")
    key = derive_key("s", application_id=APPLICATION_ID, user_id=USER, modality="face", key_version=key_version)
    return validate_template_pool(rows, stored=stored, key=key, pool_size=pool_size)


def _rows(client):
    from backend.database import crud

    _enroll_both(client)
    return crud.get_set_rows(_db(), USER, APPLICATION_ID)


def test_pool_validation_passes_for_a_healthy_pool(client):
    assert all(_validate(_rows(client)).values())


def test_pool_validation_detects_each_broken_invariant(client):
    rows = _rows(client)
    face = {r.template_set_version: r for r in rows if r.modality == "face"}
    voice = {r.template_set_version: r for r in rows if r.modality == "voice"}

    assert not _validate(rows, key_version=2)["key_version_matches_template"]
    assert not _validate(rows, pool_size=3)["pool_size_correct"]

    voice[1].is_active = False  # only ONE modality's row still active: rows are inconsistent
    assert not _validate(rows)["exactly_one_active_set"]
    voice[1].is_active = True
    voice[2].is_active, voice[2].template_status = True, "ACTIVE"  # voice active in two sets
    checks = _validate(rows)
    assert not checks["exactly_one_active_set"]
    voice[2].is_active, voice[2].template_status = False, "STANDBY"

    original = face[2].protected_template
    face[2].protected_template = face[1].protected_template  # standby duplicating the active template
    assert not _validate(rows)["standby_unique"]
    face[2].protected_template = original

    face[3].template_set_status = "ACTIVE"  # row says STANDBY, set says ACTIVE
    assert not _validate(rows)["set_status_consistent"]
    face[3].template_set_status = "STANDBY"

    face[1].template_status = "REVOKED"  # a revoked row flagged active
    assert not _validate(rows)["revoked_never_active"]


def test_authentication_fails_closed_when_the_pool_is_corrupt(tmp_path, monkeypatch):
    from backend.database import crud

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'c.db'}")
    _reset_caches()
    from backend.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        _enroll(test_client, FACE)
        db = _db()
        rows = crud.get_set_rows(db, USER, APPLICATION_ID)
        rows[1].key_version = rows[2].key_version  # two live face templates under one key version
        db.commit()
        response = _auth(test_client, FACE)
    _reset_caches()
    assert response.status_code == 500 and "authenticated" not in response.json()


# --- migration -------------------------------------------------------------------------------

_AUDIT_DDL = (
    "CREATE TABLE audit_logs (audit_id VARCHAR PRIMARY KEY, timestamp DATETIME, user_id VARCHAR, building_id VARCHAR, "
    "modality_list JSON, similarity_scores JSON, thresholds_used JSON, fusion_score FLOAT, fusion_policy VARCHAR, "
    "authenticated BOOLEAN, latency_ms INTEGER, template_versions JSON, key_versions JSON)"
)
_USERS_DDL = "CREATE TABLE users (id VARCHAR PRIMARY KEY, username VARCHAR, created_at DATETIME)"
_LEGACY_TEMPLATES_DDL = (
    "CREATE TABLE protected_templates (template_id VARCHAR PRIMARY KEY, user_id VARCHAR, modality VARCHAR, "
    "application_id VARCHAR, template_version INTEGER, key_version INTEGER, output_bits INTEGER, "
    "protected_template BLOB, is_active BOOLEAN, created_at DATETIME)"
)
_POOL_TEMPLATES_DDL = _LEGACY_TEMPLATES_DDL[:-1] + (
    ", template_status VARCHAR, activation_time DATETIME, revoked_time DATETIME, revoked_reason VARCHAR, "
    "template_group_id VARCHAR, template_index INTEGER)"
)
_ACTIVE_INDEX = (
    "CREATE UNIQUE INDEX ux_one_active_template_per_context ON protected_templates (user_id, modality, application_id) "
    "WHERE is_active = 1"
)


def _legacy_db(tmp_path, monkeypatch, templates_ddl):
    import sqlite3

    db_file = tmp_path / "legacy.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    _reset_caches()
    connection = sqlite3.connect(db_file)
    for statement in (_USERS_DDL, templates_ddl, _ACTIVE_INDEX, _AUDIT_DDL):
        connection.execute(statement)
    connection.execute("INSERT INTO users VALUES (?, NULL, '2026-01-01 00:00:00')", (USER,))
    return connection


def _seed_matching_master_secret_fingerprint():
    """Record a MASTER_SECRET fingerprint matching this suite's fixed test secret.

    These tests insert legacy `protected_templates` rows directly via raw SQL to simulate a real
    pre-migration production database, then start the app to prove `upgrade_schema()`'s migration
    logic. That's a separate concern from `backend/key_continuity.py`'s startup check (see
    tests/test_key_continuity.py for that) - without this, a legacy database with existing
    templates and no recorded fingerprint would correctly, but irrelevantly to what these tests
    are about, hit that check's hard-fail. Call after closing the raw `sqlite3` connection above,
    before starting `TestClient(app)`.
    """
    from backend.database import crud
    from backend.database.migration import upgrade_schema
    from backend.database.session import get_engine, get_session_factory
    from backend.secret_fingerprint import fingerprint_master_secret

    upgrade_schema(get_engine())  # creates master_secret_fingerprint; idempotent, the app repeats this at its own startup
    with get_session_factory()() as db:
        crud.initialize_master_secret_fingerprint(db, fingerprint_master_secret("unit-test-master-secret-not-for-production"))


def test_pre_pool_single_template_database_migrates_into_set_one_and_keeps_authenticating(tmp_path, monkeypatch):
    from backend.services.face_service import get_face_service
    from template_protection.biohash import generate_template
    from template_protection.hkdf_keys import derive_key
    from template_protection.utils import pack_bits

    connection = _legacy_db(tmp_path, monkeypatch, _LEGACY_TEMPLATES_DDL)
    embedding = get_face_service().pipeline.embed(cv2.cvtColor(cv2.imdecode(np.frombuffer(FACE, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    key = derive_key("unit-test-master-secret-not-for-production", application_id=APPLICATION_ID, user_id=USER, modality="face", key_version=3)
    bits = pack_bits(generate_template(embedding, key, output_bits=256))
    connection.execute("INSERT INTO protected_templates VALUES ('old-active', ?, 'face', ?, 1, 3, 256, ?, 1, '2026-01-01 00:00:00')", (USER, APPLICATION_ID, bits))
    connection.execute("INSERT INTO protected_templates VALUES ('old-dead', ?, 'face', ?, 1, 2, 256, ?, 0, '2025-12-01 00:00:00')", (USER, APPLICATION_ID, bits))
    connection.commit()
    connection.close()
    _seed_matching_master_secret_fingerprint()

    from backend.main import app

    with TestClient(app) as test_client:  # startup runs upgrade_schema()
        body = _sets(test_client)
        assert body["active_template_set_version"] == 1 and body["standby_count"] == 0
        assert [(s["template_set_version"], s["status"]) for s in body["sets"]] == [(1, "ACTIVE")]

        auth = _auth(test_client, FACE).json()  # the pre-migration template still authenticates
        assert auth["authenticated"] is True and auth["key_version"] == 3 and auth["template_set_version"] == 1
        assert _revoke(test_client, files=_captures(voice=None)).status_code == 409  # a pool of one set

        generated = test_client.post(
            f"/templates/{USER}/generate", data={"application_id": APPLICATION_ID}, files=_captures(voice=None)
        )
        assert generated.status_code == 200 and generated.json()["new_template_set_version"] == 2
        assert _revoke(test_client, files=_captures(voice=None)).json()["new_active_template_set_version"] == 2
        assert _auth(test_client, FACE).json()["key_version"] == 4  # continues after the highest legacy key
    _reset_caches()


def test_previous_per_modality_pools_are_converted_to_sets_even_if_modalities_drifted(tmp_path, monkeypatch):
    connection = _legacy_db(tmp_path, monkeypatch, _POOL_TEMPLATES_DDL)
    rows = []
    # face was revoked once (active at T2); voice never was (active at T1): drifted pools.
    layout = {
        "face": {1: "REVOKED", 2: "ACTIVE", 3: "STANDBY", 4: "STANDBY"},
        "voice": {1: "ACTIVE", 2: "STANDBY", 3: "STANDBY", 4: "STANDBY"},
    }
    for modality, per_index in layout.items():
        for index, status in per_index.items():
            rows.append(
                (f"{modality}-{index}", USER, modality, APPLICATION_ID, 1, index, 256, bytes([index]) * 32,
                 1 if status == "ACTIVE" else 0, f"2026-01-0{index} 00:00:00", status,
                 "2026-01-05 00:00:00" if status == "ACTIVE" else None, "2026-01-04 00:00:00" if status == "REVOKED" else None,
                 "revoked by user" if status == "REVOKED" else None, f"g-{modality}", index)
            )
    connection.executemany("INSERT INTO protected_templates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()
    _seed_matching_master_secret_fingerprint()

    from backend.main import app

    with TestClient(app) as test_client:
        body = _sets(test_client)
        assert body["active_template_set_version"] == 2
        assert {s["template_set_version"]: s["status"] for s in body["sets"]} == {1: "REVOKED", 2: "ACTIVE", 3: "STANDBY", 4: "STANDBY"}
        assert all(s["modalities"] == ["face", "voice"] for s in body["sets"])
        from backend.database import crud

        active = crud.get_active_set_rows(_db(), USER, APPLICATION_ID)
        assert sorted((r.modality, r.template_set_version) for r in active) == [("face", 2), ("voice", 2)]
        drifted = next(r for r in crud.get_set_rows(_db(), USER, APPLICATION_ID) if r.template_id == "voice-1")
        assert drifted.template_status == "REVOKED" and drifted.revoked_reason == "migrated to set-level activation"
    _reset_caches()


def test_upgrade_schema_is_idempotent(tmp_path, monkeypatch):
    from backend.database.migration import upgrade_schema
    from backend.database.session import get_engine

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'i.db'}")
    _reset_caches()
    first = upgrade_schema(get_engine())
    second = upgrade_schema(get_engine())
    _reset_caches()
    assert first["columns_added"] == [] and second == {
        "columns_added": [], "legacy_templates_backfilled": 0, "template_set_rows_backfilled": 0,
        "retired_modalities_removed": {"rows_deleted": {"protected_templates": 0, "enrollment_events": 0}, "columns_dropped": []},
    }


def test_upgrade_removes_retired_iris_and_fingerprint_data_and_keeps_face_and_voice(tmp_path, monkeypatch):
    """Iris and fingerprint were removed from the system: `upgrade_schema` deletes their templates and drops the
    audit column that held fingerprint scores. Face and voice rows (and old audit rows) are untouched."""
    connection = _legacy_db(tmp_path, monkeypatch, _POOL_TEMPLATES_DDL)
    connection.execute("ALTER TABLE audit_logs ADD COLUMN fingerprint_similarity FLOAT")
    connection.execute(
        "INSERT INTO audit_logs (audit_id, user_id, authenticated, latency_ms, fingerprint_similarity) VALUES ('a1', ?, 1, 5, 0.9)",
        (USER,),
    )
    rows = [
        (f"{modality}-1", USER, modality, APPLICATION_ID, 1, 1, 256, bytes([n]) * 32, 1, "2026-01-01 00:00:00", "ACTIVE",
         "2026-01-01 00:00:00", None, None, "g-1", 1)
        for n, modality in enumerate(("face", "voice", "fingerprint", "iris"), start=1)
    ]
    connection.executemany("INSERT INTO protected_templates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()

    from sqlalchemy import inspect, text

    from backend.database.migration import upgrade_schema
    from backend.database.session import get_engine

    result = upgrade_schema(get_engine())
    engine = get_engine()
    with engine.connect() as db:
        modalities = sorted(m for (m,) in db.execute(text("SELECT modality FROM protected_templates")))
        audit_ids = [a for (a,) in db.execute(text("SELECT audit_id FROM audit_logs"))]
    columns = {c["name"] for c in inspect(engine).get_columns("audit_logs")}
    _reset_caches()
    assert modalities == ["face", "voice"]
    assert audit_ids == ["a1"] and "fingerprint_similarity" not in columns
    assert result["retired_modalities_removed"] == {
        "rows_deleted": {"protected_templates": 2, "enrollment_events": 0},
        "columns_dropped": ["audit_logs.fingerprint_similarity"],
    }
