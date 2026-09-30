"""Synchronized cancelable-template rotation - the security response to a SUSPICIOUS authentication attempt.

These tests exercise the rotation MECHANISM, so `rotation_on` also sets MAX_MODALITY_RETRIES=0: the first high-quality
mismatch already escalates. WHEN rotation is triggered under the default retry policy (capture errors, quality failures
and isolated mismatches never rotate) is tested in tests/test_authentication_failure_handling.py.

Real HTTP layer, services, BioHash, database and fusion policy; deterministic stub embedders (tests/test_flexible_auth.py).
SOFTWARE TEST - not a biometric trial.
"""

from __future__ import annotations

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
    _enroll_user,
    client,
)

A, B = "user-a", "user-b"
BUILDING = "national_data_center"


@pytest.fixture
def rotation_on(monkeypatch):
    monkeypatch.setenv("TEMPLATE_ROTATION_ON_FAILED_AUTH", "true")
    monkeypatch.setenv("MAX_MODALITY_RETRIES", "0")  # escalate on the first high-quality mismatch (mechanism tests)

SUSPICIOUS_FACE = "suspicious authentication attempt: repeated high-quality face verification mismatch"


def _reload_settings():
    """Apply an env change made after the `client` fixture started (Settings are cached per process)."""
    from backend.config import get_settings

    get_settings.cache_clear()


def _sets(user):
    """{set_version: (status, {modality: key_version}, {modality: template bytes})}"""
    from backend.database import crud

    out = {}
    for r in crud.get_set_rows(_db(), user, APPLICATION_ID):
        status, keys, blobs = out.setdefault(r.template_set_version, (r.template_set_status, {}, {}))
        keys[r.modality] = r.key_version
        blobs[r.modality] = bytes(r.protected_template)
    return out


def _active(user):
    return next(v for v, (status, _, _) in _sets(user).items() if status == "ACTIVE")


def test_1_successful_authentication_does_not_rotate(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _sets(A)
    body = _authenticate(client, A, BUILDING, face=FACE, voice=VOICE).json()
    assert body["status"] == "ACCESS_GRANTED" and "template_rotation" not in body
    assert _sets(A) == before
    entry = _audit(client, A)[0]
    assert entry["template_rotation_triggered"] is False and entry["template_rotation_status"] == "NOT_APPLICABLE"


def test_2_impostor_face_is_denied_and_every_enrolled_modality_moves_t1_to_t2(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    before = _sets(A)
    assert _active(A) == 1
    body = _authenticate(client, A, BUILDING, face=OTHER_FACE).json()  # B's face presented on A's account
    assert body["status"] == "ACCESS_DENIED" and body["authenticated"] is False
    rot = body["template_rotation"]
    assert rot == {"status": "ROTATED", "triggered": True, "active_set_before": 1, "active_set_after": 2,
                   "modalities": ["face", "voice"], "remaining_standby_sets": 2, "reason": SUSPICIOUS_FACE}
    assert body["attempt_assessment"]["status"] == "SUSPICIOUS_ATTEMPT"
    after = _sets(A)
    assert after[1][0] == "REVOKED" and after[2][0] == "ACTIVE"
    assert set(after[2][1]) == {"face", "voice"}  # the whole set moved together
    # No secret material leaves the backend.
    assert not any(k in str(rot).lower() for k in ("seed", "secret", "projection", "template_bytes"))
    entry = _audit(client, A)[0]
    assert (entry["authentication_state"], entry["active_set_before"], entry["active_set_after"],
            entry["template_rotation_triggered"], entry["template_rotation_status"]) == ("ACCESS_DENIED", 1, 2, True, "ROTATED")
    assert entry["failed_modalities"] == ["face"]
    # TEST 6: the promoted templates are the ones generated at ENROLLMENT - byte-identical to the pre-attempt STANDBY
    # set 2 - so nothing derived from the impostor sample was stored.
    assert after[2][2] == before[2][2]
    assert sum(len(b) for _, _, b in after.values()) == sum(len(b) for _, _, b in before.values())  # no new rows


def test_3_legitimate_user_still_authenticates_after_rotation(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    _authenticate(client, A, BUILDING, face=OTHER_FACE)
    body = _authenticate(client, A, BUILDING, face=FACE, voice=VOICE).json()
    assert body["status"] == "ACCESS_GRANTED" and body["template_set_version"] == 2
    body = _authenticate(client, A, BUILDING, face=FACE).json()
    assert body["status"] == "ACCESS_GRANTED" and body["template_set_version"] == 2


def test_4_impostor_remains_rejected_after_rotation(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    _enroll_user(client, B, face=OTHER_FACE, voice=OTHER_VOICE)
    _authenticate(client, A, BUILDING, face=OTHER_FACE)
    body = _authenticate(client, A, BUILDING, face=OTHER_FACE, voice=OTHER_VOICE).json()  # B's biometrics, A's account
    assert body["status"] == "ACCESS_DENIED" and body["template_rotation"]["active_set_before"] == 2
    assert body["template_rotation"]["active_set_after"] == 3
    # B's own account is untouched by attempts on A's account.
    assert _active(B) == 1 and _authenticate(client, B, BUILDING, face=OTHER_FACE).json()["status"] == "ACCESS_GRANTED"


def test_5_face_only_account_rotates_only_face(rotation_on, client):
    _enroll_user(client, A, face=FACE)
    body = _authenticate(client, A, BUILDING, face=OTHER_FACE).json()
    assert body["template_rotation"]["modalities"] == ["face"]
    after = _sets(A)
    assert all(set(keys) == {"face"} for _, keys, _ in after.values())  # no voice fabricated


def test_face_voice_account_shifts_together(rotation_on, client):
    _enroll_user(client, A, face=FACE, voice=VOICE)
    body = _authenticate(client, A, BUILDING, face=OTHER_FACE, voice=VOICE).json()  # ALL_REQUIRED: face fails -> denied
    assert body["status"] == "ACCESS_DENIED"
    assert body["template_rotation"]["modalities"] == ["face", "voice"]
    assert set(_sets(A)[2][1]) == {"face", "voice"}


def test_noisy_voice_failure_is_retried_not_rotated_and_is_not_called_an_attack(monkeypatch, client):
    monkeypatch.setenv("TEMPLATE_ROTATION_ON_FAILED_AUTH", "true")  # default retry policy
    _reload_settings()
    _enroll_user(client, A, face=FACE, voice=VOICE)
    body = _authenticate(client, A, BUILDING, face=FACE, voice=OTHER_VOICE).json()
    assert body["status"] == "ACCESS_DENIED"  # ALL_REQUIRED still decides the attempt
    assert "template_rotation" not in body and _active(A) == 1
    assessment = body["attempt_assessment"]
    assert assessment["status"] == "RETRY_REQUESTED" and assessment["retry_modalities"] == ["voice"]
    assert "attack" not in str(assessment).lower()  # no claim about intent


def test_unenrolled_modality_is_enrollment_required_and_never_rotates(rotation_on, client):
    _enroll_user(client, A, face=FACE)
    response = _authenticate(client, A, BUILDING, face=FACE, voice=VOICE)  # voice not enrolled
    assert response.status_code == 409
    assert _active(A) == 1 and len(_sets(A)) == 4


def test_pool_exhaustion_stops_rotating_and_keeps_the_user_able_to_authenticate(rotation_on, client):
    _enroll_user(client, A, face=FACE)
    for expected_after in (2, 3, 4):
        assert _authenticate(client, A, BUILDING, face=OTHER_FACE).json()["template_rotation"]["active_set_after"] == expected_after
    body = _authenticate(client, A, BUILDING, face=OTHER_FACE).json()
    assert body["template_rotation"]["status"] == "POOL_EXHAUSTED" and body["template_rotation"]["triggered"] is False
    assert body["template_rotation"]["active_set_before"] == body["template_rotation"]["active_set_after"] == 4
    assert _authenticate(client, A, BUILDING, face=FACE).json()["status"] == "ACCESS_GRANTED"


def test_rotation_is_off_by_default(monkeypatch, client):
    monkeypatch.setenv("MAX_MODALITY_RETRIES", "0")  # escalate at once: the security response itself is off
    _reload_settings()
    _enroll_user(client, A, face=FACE)
    body = _authenticate(client, A, BUILDING, face=OTHER_FACE).json()
    assert body["status"] == "ACCESS_DENIED" and "template_rotation" not in body
    assert _active(A) == 1
    assert _audit(client, A)[0]["template_rotation_status"] == "DISABLED"


# ----------------------------------------------------------------------------- concurrency (file-backed SQLite)


def _assert_one_active_set_shared_by_all(user, modalities):
    from backend.database import crud

    rows = crud.get_set_rows(_db(), user, APPLICATION_ID)
    active = {(r.modality, r.template_set_version) for r in rows if r.is_active}
    assert {m for m, _ in active} == set(modalities) and len({v for _, v in active}) == 1, active
    by_set = {}
    for r in rows:
        by_set.setdefault(r.template_set_version, set()).add(r.template_set_status)
    assert all(len(s) == 1 for s in by_set.values()), by_set  # every set is uniformly ACTIVE / STANDBY / REVOKED
    assert sum(s == {"ACTIVE"} for s in by_set.values()) == 1


def test_concurrent_rotations_of_the_same_set_rotate_exactly_once(rotation_on, tmp_path):
    """N simultaneous DENIED attempts evaluated against set 1 -> exactly one T1 -> T2 transition, no skipped set."""
    import threading

    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.config import Settings
    from backend.database import crud
    from backend.database.models import Base
    from backend.services.authentication import rotate_after_failed_authentication
    from backend.services.base_service import ModalityService

    class _P:
        is_mock = False

        def embed(self, raw):
            v = np.random.default_rng(int(raw)).standard_normal(512)
            return v / np.linalg.norm(v)

    for trial in range(3):
        engine = create_engine(f"sqlite:///{tmp_path / f'race{trial}.db'}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        settings = Settings(master_secret="race", template_pool_size=4, template_rotation_on_failed_auth=True)
        db = factory()
        for m in ("face", "voice"):
            service = ModalityService(m, _P(), settings)
            service.capture_gate = False  # the stub "recording" is a seed, not audio
            service.enroll(db, 1, "u", "app")
        db.close()
        n, barrier, out = 8, threading.Barrier(8), []

        def worker():
            s = factory()
            barrier.wait()
            try:
                out.append(rotate_after_failed_authentication(s, settings, "u", "app", 1))
            finally:
                s.close()

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(r.status for r in out) == ["ALREADY_ROTATED"] * (n - 1) + ["ROTATED"]
        assert all((r.active_set_before, r.active_set_after) == (1, 2) for r in out)
        s = factory()
        rows = crud.get_set_rows(s, "u", "app")
        assert {r.modality: r.template_set_version for r in rows if r.is_active} == {"face": 2, "voice": 2}
        assert {v: rows_[0].template_set_status for v, rows_ in crud._group_by_set(rows).items()} == {
            1: "REVOKED", 2: "ACTIVE", 3: "STANDBY", 4: "STANDBY"}
        s.close()
        engine.dispose()


def test_concurrent_denied_http_requests_rotate_once_and_audit_every_attempt(rotation_on, client):
    import threading

    _enroll_user(client, A, face=FACE, voice=VOICE)
    n, barrier, bodies = 6, threading.Barrier(6), []

    def attempt():
        barrier.wait()
        bodies.append(_authenticate(client, A, BUILDING, face=OTHER_FACE).json())

    threads = [threading.Thread(target=attempt) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(b["status"] == "ACCESS_DENIED" for b in bodies)
    statuses = sorted(b["template_rotation"]["status"] for b in bodies)
    # Requests evaluated against set 1 rotate it once; a request that started after the rotation was evaluated against
    # set 2 and may legitimately rotate that one - either way every rotation retires the set that attempt was checked on.
    rotations = [b["template_rotation"] for b in bodies if b["template_rotation"]["status"] == "ROTATED"]
    assert set(statuses) <= {"ROTATED", "ALREADY_ROTATED"}
    assert sorted(r["active_set_before"] for r in rotations) == list(range(1, len(rotations) + 1))  # no set skipped
    _assert_one_active_set_shared_by_all(A, ("face", "voice"))
    entries = _audit(client, A)
    assert len(entries) == n  # exactly one audit row per attempt - no duplicates, none lost
    assert sum(e["template_rotation_status"] == "ROTATED" for e in entries) == len(rotations)
    assert _authenticate(client, A, BUILDING, face=FACE, voice=VOICE).json()["status"] == "ACCESS_GRANTED"


def test_legitimate_multimodal_attempt_racing_impostor_rotations_always_gets_a_decision(rotation_on, client):
    """A genuine face+voice attempt in the middle of concurrent rotations is evaluated against ONE snapshot of the
    pool: it gets a clean decision (never a 500 from a torn read), and the one-active-set invariant always holds."""
    import threading

    _enroll_user(client, A, face=FACE, voice=VOICE)
    barrier, responses = threading.Barrier(6), []

    def impostor():
        barrier.wait()
        responses.append(("impostor", _authenticate(client, A, BUILDING, face=OTHER_FACE, voice=OTHER_VOICE)))

    def genuine():
        barrier.wait()
        responses.append(("genuine", _authenticate(client, A, BUILDING, face=FACE, voice=VOICE)))

    threads = [threading.Thread(target=impostor) for _ in range(5)] + [threading.Thread(target=genuine)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r.status_code == 200 for _, r in responses), [(who, r.status_code, r.text[:120]) for who, r in responses]
    assert all(r.json()["status"] == "ACCESS_DENIED" for who, r in responses if who == "impostor")
    genuine_body = next(r.json() for who, r in responses if who == "genuine")
    assert genuine_body["status"] == "ACCESS_GRANTED"  # its own snapshot's set, whichever that was
    _assert_one_active_set_shared_by_all(A, ("face", "voice"))
    assert len(_audit(client, A)) == 6


# ----------------------------------------------------------------------------- manual paths share the same primitive


def _revoke_request(client, user=A):
    from tests.test_flexible_auth import _fusion_files

    return client.post("/revoke-template", data={"user_id": user, "application_id": APPLICATION_ID},
                       files=_fusion_files(face=FACE, voice=VOICE))


def test_manual_revoke_still_needs_biometric_authorization_and_rotates_all_modalities_together(client):
    from tests.test_flexible_auth import _fusion_files

    _enroll_user(client, A, face=FACE, voice=VOICE)
    denied = client.post("/revoke-template", data={"user_id": A, "application_id": APPLICATION_ID},
                         files=_fusion_files(face=OTHER_FACE, voice=VOICE))
    assert denied.status_code == 403 and _active(A) == 1  # unchanged security requirement
    body = _revoke_request(client).json()
    assert (body["revoked_template_set_version"], body["new_active_template_set_version"]) == (1, 2)
    _assert_one_active_set_shared_by_all(A, ("face", "voice"))


def test_concurrent_authorized_revokes_never_double_rotate_or_create_two_active_sets(client):
    """Several AUTHORIZED /revoke-template requests at once: each retires only the set it authorized against, so the
    successes form a gap-free chain (1->2, 2->3, ...); a request whose authorized set was already retired gets 409."""
    import threading

    _enroll_user(client, A, face=FACE, voice=VOICE)
    n, barrier, responses = 6, threading.Barrier(6), []

    def revoke():
        barrier.wait()
        responses.append(_revoke_request(client))

    threads = [threading.Thread(target=revoke) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    codes = sorted(r.status_code for r in responses)
    assert set(codes) <= {200, 409}, [(r.status_code, r.text[:120]) for r in responses]
    ok = sorted((r.json()["revoked_template_set_version"], r.json()["new_active_template_set_version"]) for r in responses if r.status_code == 200)
    assert ok and ok == [(v, v + 1) for v in range(1, len(ok) + 1)]  # no double rotation of a set, no skipped set
    assert _active(A) == 1 + len(ok)
    _assert_one_active_set_shared_by_all(A, ("face", "voice"))
    assert _authenticate(client, A, BUILDING, face=FACE, voice=VOICE).json()["status"] == "ACCESS_GRANTED"


def test_concurrent_authorized_activations_promote_exactly_one_set(client):
    import threading

    from tests.test_flexible_auth import _fusion_files

    _enroll_user(client, A, face=FACE, voice=VOICE)
    barrier, responses = threading.Barrier(4), []

    def activate(version):
        barrier.wait()
        responses.append(client.post(f"/templates/{A}/activate/{version}", data={"application_id": APPLICATION_ID},
                                     files=_fusion_files(face=FACE, voice=VOICE)))

    threads = [threading.Thread(target=activate, args=(v,)) for v in (3, 3, 4, 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(r.status_code for r in responses) <= {200, 404, 409}, [(r.status_code, r.text[:120]) for r in responses]
    ok = [r.json() for r in responses if r.status_code == 200]
    assert ok and all(o["previous_active_template_set_version"] != o["new_active_template_set_version"] for o in ok)
    _assert_one_active_set_shared_by_all(A, ("face", "voice"))
    # successes form a chain: each one retired the set the previous one activated (or the original set 1)
    chain = sorted(ok, key=lambda o: o["previous_active_template_set_version"])
    assert chain[0]["previous_active_template_set_version"] == 1
    assert all(x["new_active_template_set_version"] == y["previous_active_template_set_version"] for x, y in zip(chain, chain[1:]))
    assert _active(A) == chain[-1]["new_active_template_set_version"]


def test_every_rotation_path_uses_the_one_compare_and_swap_primitive(monkeypatch, rotation_on, client):
    from backend.database import crud

    calls = []
    original = crud.rotate_active_set_if_current

    def spy(*args, **kwargs):
        calls.append(args[3] if len(args) > 3 else kwargs.get("expected_active"))
        return original(*args, **kwargs)

    monkeypatch.setattr(crud, "rotate_active_set_if_current", spy)
    _enroll_user(client, A, face=FACE, voice=VOICE)
    _authenticate(client, A, BUILDING, face=OTHER_FACE)  # automatic: 1 -> 2
    _revoke_request(client)  # manual revoke: 2 -> 3
    from tests.test_flexible_auth import _fusion_files

    client.post(f"/templates/{A}/activate/4", data={"application_id": APPLICATION_ID},
                files=_fusion_files(face=FACE, voice=VOICE))  # manual activation: 3 -> 4
    assert calls == [1, 2, 3]  # each path passed the set it evaluated/authorized against
    assert _active(A) == 4
